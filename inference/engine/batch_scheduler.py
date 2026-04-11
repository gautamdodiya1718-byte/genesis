"""
inference/engine/batch_scheduler.py
--------------------------------------
Batch generation scheduler for throughput-optimized inference.

Problem: On CPU, loading a model once and running N prompts sequentially
is much faster than N separate load→generate→unload cycles.

The BatchScheduler:
  1. Accepts GenerationRequests via submit()
  2. Groups requests by model_key (same model = one batch)
  3. Executes each model's queue in order, reusing the loaded pipeline
  4. Calls result_callback with each completed GenerationResult
  5. Supports priority queue (higher priority requests run first)
  6. Supports timeout-based auto-flush (don't wait indefinitely for a full batch)

Batch strategies:
  eager    — run each request immediately (no batching, lowest latency)
  grouped  — group by model, run each group together (better throughput)
  windowed — collect requests for window_s seconds, then run (best GPU util)

Usage (async API server):
    scheduler = BatchScheduler(pipeline, strategy="grouped")
    scheduler.start()
    job_id = scheduler.submit(request, callback=on_complete)
    scheduler.stop()

Usage (offline bulk generation):
    scheduler = BatchScheduler(pipeline, strategy="eager")
    results = scheduler.run_sync([req1, req2, req3])
"""
from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

from inference.engine.model_router import GenerationRequest
from inference.engine.optimized_pipeline import OptimizedPipeline, GenerationResult

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    QUEUED     = "queued"
    RUNNING    = "running"
    DONE       = "done"
    FAILED     = "failed"
    CANCELLED  = "cancelled"


@dataclass
class BatchJob:
    job_id:   str
    request:  GenerationRequest
    priority: int = 0           # higher = runs first
    submitted_at: float = field(default_factory=time.time)
    started_at:   Optional[float] = None
    finished_at:  Optional[float] = None
    status: JobStatus = JobStatus.QUEUED
    result: Optional[GenerationResult] = None
    callback: Optional[Callable[[GenerationResult], None]] = None

    def __lt__(self, other: "BatchJob") -> bool:
        # Priority queue: highest priority, then FIFO
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.submitted_at < other.submitted_at

    @property
    def wait_s(self) -> Optional[float]:
        if self.started_at:
            return self.started_at - self.submitted_at
        return None

    @property
    def duration_s(self) -> Optional[float]:
        if self.started_at and self.finished_at:
            return self.finished_at - self.started_at
        return None


@dataclass
class SchedulerStats:
    total_submitted: int = 0
    total_completed: int = 0
    total_failed: int = 0
    total_images: int = 0
    mean_wait_s: float = 0.0
    mean_duration_s: float = 0.0
    queue_depth: int = 0
    active_model: Optional[str] = None


class BatchScheduler:
    """
    Background job scheduler for batched image generation.

    Thread-safe: submit() can be called from any thread.
    Worker thread runs continuously until stop() is called.
    """

    def __init__(
        self,
        pipeline: OptimizedPipeline,
        strategy: str = "grouped",      # eager | grouped | windowed
        window_s: float = 2.0,          # for windowed strategy
        max_queue_size: int = 100,
        max_concurrent_model: int = 1,  # how many jobs run per model group
    ):
        self.pipeline   = pipeline
        self.strategy   = strategy
        self.window_s   = window_s
        self.max_queue  = max_queue_size

        self._pq: queue.PriorityQueue = queue.PriorityQueue(maxsize=max_queue_size)
        self._jobs: Dict[str, BatchJob] = {}
        self._jobs_lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._stats = SchedulerStats()
        self._waits:     List[float] = []
        self._durations: List[float] = []

    # ── Public API ─────────────────────────────────────────────

    def start(self) -> None:
        """Start background worker thread."""
        if self._worker and self._worker.is_alive():
            return
        self._stop_evt.clear()
        self._worker = threading.Thread(
            target=self._worker_loop, daemon=True, name="genesis-batch-worker"
        )
        self._worker.start()
        logger.info(f"BatchScheduler started | strategy={self.strategy}")

    def stop(self, wait: bool = True, timeout: float = 30.0) -> None:
        """Stop worker thread. If wait=True, waits for queue to drain."""
        self._stop_evt.set()
        if wait and self._worker:
            self._worker.join(timeout=timeout)
        logger.info("BatchScheduler stopped")

    def submit(
        self,
        request: GenerationRequest,
        priority: int = 0,
        callback: Optional[Callable[[GenerationResult], None]] = None,
    ) -> str:
        """
        Submit a generation request to the scheduler.
        Returns job_id. Result delivered via callback (if provided).
        """
        job_id = request.request_id or str(uuid.uuid4())[:8]
        job = BatchJob(
            job_id=job_id,
            request=request,
            priority=priority,
            callback=callback,
        )
        with self._jobs_lock:
            self._jobs[job_id] = job
        try:
            self._pq.put((job,), block=False)
            self._stats.total_submitted += 1
        except queue.Full:
            logger.warning(f"Queue full — rejecting job {job_id}")
            job.status = JobStatus.FAILED
            job.result = GenerationResult(
                images=[], request_id=job_id,
                model_key="none", steps=0, guidance=0,
                duration_s=0, width=0, height=0, seed=None,
                error="Queue full",
            )
        return job_id

    def get_job(self, job_id: str) -> Optional[BatchJob]:
        with self._jobs_lock:
            return self._jobs.get(job_id)

    def wait_for(self, job_id: str, timeout: float = 120.0) -> Optional[GenerationResult]:
        """Block until a job completes or timeout. Returns result or None."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = self.get_job(job_id)
            if job and job.status in (JobStatus.DONE, JobStatus.FAILED):
                return job.result
            time.sleep(0.1)
        return None

    def cancel(self, job_id: str) -> bool:
        """Cancel a queued job. Has no effect if already running."""
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job and job.status == JobStatus.QUEUED:
                job.status = JobStatus.CANCELLED
                return True
        return False

    def run_sync(self, requests: List[GenerationRequest]) -> List[GenerationResult]:
        """
        Run a list of requests synchronously (blocking). No worker thread needed.
        Grouped by model for efficiency: runs all requests for same model together.
        """
        results: Dict[str, GenerationResult] = {}

        if self.strategy == "grouped":
            # Group by routed model_key
            groups: Dict[str, List[GenerationRequest]] = {}
            for req in requests:
                decision = self.pipeline.router.route(req)
                groups.setdefault(decision.model_key, []).append(req)

            for model_key, group in groups.items():
                logger.info(f"  Running {len(group)} jobs on {model_key}")
                for req in group:
                    result = self.pipeline.generate(req)
                    results[req.request_id] = result
        else:
            # Eager: run each immediately
            for req in requests:
                results[req.request_id] = self.pipeline.generate(req)

        return [results[req.request_id] for req in requests]

    def stats(self) -> SchedulerStats:
        self._stats.queue_depth = self._pq.qsize()
        if self._waits:
            self._stats.mean_wait_s = sum(self._waits[-50:]) / len(self._waits[-50:])
        if self._durations:
            self._stats.mean_duration_s = sum(self._durations[-50:]) / len(self._durations[-50:])
        return self._stats

    # ── Worker ─────────────────────────────────────────────────

    def _worker_loop(self) -> None:
        logger.info("Batch worker started")
        while not self._stop_evt.is_set():
            if self.strategy == "windowed":
                self._run_windowed()
            elif self.strategy == "grouped":
                self._run_grouped_cycle()
            else:
                self._run_eager_cycle()
        logger.info("Batch worker exited")

    def _run_eager_cycle(self) -> None:
        """Process one job immediately."""
        try:
            (job,) = self._pq.get(timeout=0.5)
        except queue.Empty:
            return
        if job.status == JobStatus.CANCELLED:
            return
        self._execute_job(job)

    def _run_grouped_cycle(self) -> None:
        """
        Drain the queue into groups by model, run each group.
        Allows loading a model once for multiple prompts.
        """
        # Collect all currently queued jobs
        batch: List[BatchJob] = []
        try:
            while True:
                (job,) = self._pq.get_nowait()
                if job.status != JobStatus.CANCELLED:
                    batch.append(job)
        except queue.Empty:
            pass

        if not batch:
            time.sleep(0.1)
            return

        # Group by routed model
        groups: Dict[str, List[BatchJob]] = {}
        for job in batch:
            decision = self.pipeline.router.route(job.request)
            groups.setdefault(decision.model_key, []).append(job)

        for model_key, group in groups.items():
            logger.info(f"Executing group: {len(group)} jobs on {model_key}")
            for job in sorted(group):  # priority sort within group
                self._execute_job(job)

    def _run_windowed(self) -> None:
        """Collect jobs for window_s seconds, then run all."""
        collected: List[BatchJob] = []
        deadline = time.time() + self.window_s
        while time.time() < deadline:
            try:
                (job,) = self._pq.get(timeout=max(0.01, deadline - time.time()))
                if job.status != JobStatus.CANCELLED:
                    collected.append(job)
            except queue.Empty:
                break

        for job in sorted(collected):
            self._execute_job(job)

    def _execute_job(self, job: BatchJob) -> None:
        job.status     = JobStatus.RUNNING
        job.started_at = time.time()
        self._stats.active_model = job.request.model_hint or "auto"

        # Track wait time
        if job.wait_s is not None:
            self._waits.append(job.wait_s)

        try:
            result = self.pipeline.generate(job.request)
            job.result     = result
            job.status     = JobStatus.DONE
            job.finished_at = time.time()
            self._stats.total_completed += 1
            self._stats.total_images    += len(result.images)
            if job.duration_s is not None:
                self._durations.append(job.duration_s)

            if job.callback:
                try:
                    job.callback(result)
                except Exception as e:
                    logger.warning(f"Callback error for {job.job_id}: {e}")

        except Exception as e:
            logger.error(f"Job {job.job_id} failed: {e}", exc_info=True)
            job.status      = JobStatus.FAILED
            job.finished_at = time.time()
            job.result = GenerationResult(
                images=[], request_id=job.job_id,
                model_key="", steps=0, guidance=0,
                duration_s=0, width=0, height=0,
                seed=None, error=str(e),
            )
            self._stats.total_failed += 1

        finally:
            self._stats.active_model = None

    def print_stats(self) -> None:
        s = self.stats()
        print(f"\n  BatchScheduler | strategy={self.strategy}")
        print(f"  submitted={s.total_submitted}  completed={s.total_completed}  "
              f"failed={s.total_failed}  images={s.total_images}")
        print(f"  mean_wait={s.mean_wait_s:.2f}s  mean_gen={s.mean_duration_s:.2f}s  "
              f"queue_depth={s.queue_depth}\n")
