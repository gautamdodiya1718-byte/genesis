"""
training/orchestrator.py
--------------------------
Training lifecycle orchestrator for the Genesis self-improving pipeline.

Responsibilities:
  1. Monitor dataset for updates (new samples / new version)
  2. Snapshot dataset → versioned training dataset
  3. Trigger VAE training (if enabled + criteria met)
  4. Trigger diffusion U-Net training
  5. Run post-training evaluation (benchmark)
  6. Link eval results to checkpoint
  7. Update model registry with deployment decision
  8. Schedule next retrain cycle

Schedule:
  - Dataset refresh: every N hours (default: 24h)
  - Model retrain:   every M days  (default: 14d)
  - Evaluation:      after every training run

All state is persisted to orchestrator_state.json for crash recovery.
"""
from __future__ import annotations
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_STATE_FILE = "orchestrator_state.json"


class OrchestratorState:
    """Persisted state for the training orchestrator."""

    def __init__(
        self,
        last_dataset_snapshot: Optional[str] = None,
        last_vae_train_step: int = 0,
        last_diffusion_train_step: int = 0,
        last_eval_run_id: Optional[str] = None,
        last_retrain_time: float = 0.0,
        last_dataset_check_time: float = 0.0,
        current_dataset_version: Optional[str] = None,
        n_cycles: int = 0,
        total_samples_trained: int = 0,
    ):
        self.last_dataset_snapshot     = last_dataset_snapshot
        self.last_vae_train_step       = last_vae_train_step
        self.last_diffusion_train_step = last_diffusion_train_step
        self.last_eval_run_id          = last_eval_run_id
        self.last_retrain_time         = last_retrain_time
        self.last_dataset_check_time   = last_dataset_check_time
        self.current_dataset_version   = current_dataset_version
        self.n_cycles                  = n_cycles
        self.total_samples_trained     = total_samples_trained

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "OrchestratorState":
        return cls(**{k: d.get(k) for k in cls.__dict__.keys()
                      if not k.startswith("_")})

    @classmethod
    def from_file(cls, path: Path) -> "OrchestratorState":
        if path.exists():
            try:
                return cls.from_dict(json.loads(path.read_text()))
            except Exception:
                pass
        return cls()

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))


class TrainingOrchestrator:
    """
    Controls the self-improvement training loop.

    The orchestrator coordinates dataset versioning, training, evaluation,
    and governance decisions. It does NOT implement training itself —
    it delegates to existing training modules (VAETrainer, DiffusionTrainer).

    Usage (minimal — headless automation):
        orch = TrainingOrchestrator(cfg, dataset_versioning, checkpoint_manager)
        orch.run_forever()          # blocks, runs training loop

    Usage (single cycle — scriptable):
        orch = TrainingOrchestrator(cfg, dataset_versioning, checkpoint_manager)
        orch.run_cycle()
    """

    def __init__(
        self,
        cfg,
        dataset_versioning,           # DatasetVersioning instance
        checkpoint_manager,           # CheckpointManager instance
        benchmark_runner=None,        # BenchmarkRunner (optional)
        model_registry=None,          # ModelRegistry (optional)
        vae_trainer_factory: Optional[Callable] = None,
        diffusion_trainer_factory: Optional[Callable] = None,
        state_dir: str = "outputs/orchestrator",
        # Schedule
        dataset_refresh_hours: float = 24.0,
        retrain_interval_days: float = 14.0,
        min_new_samples_to_retrain: int = 500,
        eval_after_train: bool = True,
    ):
        self.cfg = cfg
        self.dataset_versioning   = dataset_versioning
        self.checkpoint_manager   = checkpoint_manager
        self.benchmark_runner     = benchmark_runner
        self.model_registry       = model_registry
        self.vae_trainer_factory  = vae_trainer_factory
        self.diffusion_trainer_factory = diffusion_trainer_factory
        self.eval_after_train     = eval_after_train

        # Schedule
        self.dataset_refresh_secs  = dataset_refresh_hours * 3600
        self.retrain_interval_secs = retrain_interval_days * 86400
        self.min_new_samples       = min_new_samples_to_retrain

        self.state_dir  = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._state_path = self.state_dir / _STATE_FILE
        self.state = OrchestratorState.from_file(self._state_path)

        logger.info(
            f"Orchestrator initialized | cycle={self.state.n_cycles} "
            f"last_retrain={self.state.last_retrain_time:.0f}"
        )

    def _save_state(self) -> None:
        self.state.save(self._state_path)

    # ── Decision logic ────────────────────────────────────────

    def _should_refresh_dataset(self) -> bool:
        elapsed = time.time() - self.state.last_dataset_check_time
        return elapsed >= self.dataset_refresh_secs

    def _should_retrain(self, new_sample_count: int) -> bool:
        """
        Trigger training if:
          - Enough time has passed since last retrain, AND
          - Enough new samples are available
        """
        elapsed = time.time() - self.state.last_retrain_time
        time_ok    = elapsed >= self.retrain_interval_secs
        samples_ok = new_sample_count >= self.min_new_samples
        logger.info(
            f"Retrain check: time_ok={time_ok} ({elapsed/3600:.1f}h elapsed) "
            f"samples_ok={samples_ok} ({new_sample_count} new samples)"
        )
        return time_ok and samples_ok

    # ── Step: Dataset snapshot ────────────────────────────────

    def step_snapshot_dataset(self) -> Optional[str]:
        """
        Create a new dataset snapshot if the source dataset has grown.
        Returns new version name, or None if skipped.
        """
        cfg_ds = self.cfg.get("dataset", {})
        source_root = cfg_ds.get("root", "outputs/dataset")

        # Count current samples in source
        source_index = Path(source_root) / "index.jsonl"
        if not source_index.exists():
            logger.warning(f"No index.jsonl at {source_root}")
            return None

        with open(source_index) as f:
            n_source = sum(1 for line in f if line.strip())

        prev_snap = self.dataset_versioning.get_latest()
        n_prev    = prev_snap.n_samples if prev_snap else 0
        n_new     = n_source - n_prev

        logger.info(
            f"Dataset: source={n_source} prev_snap={n_prev} new={n_new}"
        )

        if not self._should_retrain(n_new):
            logger.info("Skipping snapshot — retrain criteria not met")
            return None

        cfg_snap = self.cfg.get("orchestrator", {}).get("snapshot", {})
        snap = self.dataset_versioning.create_snapshot(
            source_dataset_root=source_root,
            filter_min_quality=cfg_snap.get("min_quality", 0.3),
            max_samples=cfg_snap.get("max_samples", None),
            description=f"Auto-snapshot at cycle {self.state.n_cycles}",
        )

        self.state.current_dataset_version = snap.version
        self.state.last_dataset_check_time = time.time()
        self._save_state()
        logger.info(f"Created snapshot: {snap.version} ({snap.n_samples} samples)")
        return snap.version

    # ── Step: VAE training ────────────────────────────────────

    def step_train_vae(
        self, dataset_version: str,
    ) -> Optional[str]:
        """
        Train VAE on the given dataset snapshot.
        Returns checkpoint ID on success, None if skipped/failed.
        """
        cfg_orch = self.cfg.get("orchestrator", {})
        if not cfg_orch.get("train_vae", False):
            logger.info("VAE training disabled in orchestrator config")
            return None

        if self.vae_trainer_factory is None:
            logger.warning("No vae_trainer_factory provided")
            return None

        snap = self.dataset_versioning.get_version(dataset_version)
        if snap is None or not snap.exists():
            logger.error(f"Snapshot {dataset_version} not found")
            return None

        logger.info(f"Starting VAE training on {dataset_version} ({snap.n_samples} samples)")
        t0 = time.time()

        try:
            trainer = self.vae_trainer_factory(
                cfg=self.cfg,
                index_path=str(snap.index_path),
                images_dir=str(snap.images_dir),
            )
            checkpoint_path = trainer.train()
        except Exception as e:
            logger.error(f"VAE training failed: {e}", exc_info=True)
            return None

        duration_h = (time.time() - t0) / 3600
        logger.info(f"VAE training done in {duration_h:.1f}h → {checkpoint_path}")
        return checkpoint_path

    # ── Step: Diffusion training ──────────────────────────────

    def step_train_diffusion(
        self,
        dataset_version: str,
        vae_checkpoint: Optional[str] = None,
    ) -> Optional[str]:
        """
        Train diffusion U-Net on the given dataset snapshot.
        Returns checkpoint ID on success.
        """
        if self.diffusion_trainer_factory is None:
            logger.warning("No diffusion_trainer_factory provided")
            return None

        snap = self.dataset_versioning.get_version(dataset_version)
        if snap is None or not snap.exists():
            logger.error(f"Snapshot {dataset_version} not found")
            return None

        logger.info(f"Starting diffusion training on {dataset_version}")
        t0 = time.time()

        try:
            trainer = self.diffusion_trainer_factory(
                cfg=self.cfg,
                index_path=str(snap.index_path),
                images_dir=str(snap.images_dir),
                vae_checkpoint=vae_checkpoint,
            )
            checkpoint_id = trainer.train()
        except Exception as e:
            logger.error(f"Diffusion training failed: {e}", exc_info=True)
            return None

        duration_h = (time.time() - t0) / 3600
        logger.info(f"Diffusion training done in {duration_h:.1f}h → {checkpoint_id}")

        self.state.last_retrain_time = time.time()
        self.state.last_diffusion_train_step = (
            self.checkpoint_manager.latest().step
            if self.checkpoint_manager.latest() else 0
        )
        self._save_state()
        return checkpoint_id

    # ── Step: Evaluation ──────────────────────────────────────

    def step_evaluate(
        self, model_version: str, checkpoint_id: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Run benchmark evaluation for a model version.
        Returns evaluation summary dict.
        """
        if self.benchmark_runner is None:
            logger.info("No benchmark_runner — skipping evaluation")
            return None

        logger.info(f"Running benchmark evaluation for {model_version}")
        try:
            result = self.benchmark_runner.run(
                model_version=model_version,
                run_id=f"auto_{model_version}_{int(time.time())}",
            )
        except Exception as e:
            logger.error(f"Benchmark failed: {e}", exc_info=True)
            return None

        # Link eval to checkpoint
        if checkpoint_id:
            self.checkpoint_manager.link_evaluation(
                checkpoint_id=checkpoint_id,
                eval_run_id=result.run_id,
                clip_score=result.mean_clip_score,
                aesthetic_score=result.mean_aesthetic_score,
                fid_score=result.fid_score,
            )

        # Register with model registry
        if self.model_registry is not None:
            try:
                self.model_registry.update_eval(
                    version=model_version,
                    eval_run_id=result.run_id,
                    clip_score=result.mean_clip_score,
                    aesthetic_score=result.mean_aesthetic_score,
                    fid_score=result.fid_score,
                )
            except Exception as e:
                logger.warning(f"Registry update failed: {e}")

        self.state.last_eval_run_id = result.run_id
        self._save_state()
        return result.to_dict()

    # ── Single cycle ──────────────────────────────────────────

    def run_cycle(self) -> dict:
        """
        Run one complete training cycle:
          1. Check for dataset updates
          2. Snapshot if criteria met
          3. Train (VAE + diffusion)
          4. Evaluate
          5. Update registry

        Returns summary dict.
        """
        self.state.n_cycles += 1
        cycle_id = f"cycle_{self.state.n_cycles:04d}"
        t0 = time.time()
        logger.info(f"\n{'='*50}")
        logger.info(f"ORCHESTRATOR CYCLE {cycle_id}")
        logger.info(f"{'='*50}")

        summary = {
            "cycle_id": cycle_id,
            "started_at": t0,
            "dataset_version": None,
            "vae_checkpoint": None,
            "diffusion_checkpoint": None,
            "eval_result": None,
            "actions_taken": [],
        }

        # Step 1: Dataset snapshot
        if self._should_refresh_dataset():
            version = self.step_snapshot_dataset()
            if version:
                summary["dataset_version"] = version
                summary["actions_taken"].append(f"snapshot:{version}")
        else:
            # Use existing latest snapshot
            latest = self.dataset_versioning.get_latest()
            if latest:
                summary["dataset_version"] = latest.version
            logger.info(f"Using existing snapshot: {summary['dataset_version']}")

        dataset_version = summary["dataset_version"]

        # Step 2: VAE training (optional)
        if dataset_version:
            vae_ckpt = self.step_train_vae(dataset_version)
            if vae_ckpt:
                summary["vae_checkpoint"] = vae_ckpt
                summary["actions_taken"].append(f"vae_train:{vae_ckpt}")

        # Step 3: Diffusion training
        if dataset_version:
            diff_ckpt = self.step_train_diffusion(
                dataset_version, summary.get("vae_checkpoint")
            )
            if diff_ckpt:
                summary["diffusion_checkpoint"] = diff_ckpt
                summary["actions_taken"].append(f"diffusion_train:{diff_ckpt}")

        # Step 4: Evaluation
        if self.eval_after_train and summary.get("diffusion_checkpoint"):
            cfg_orch = self.cfg.get("orchestrator", {})
            model_version = cfg_orch.get(
                "model_version",
                f"genesis_cycle{self.state.n_cycles}"
            )
            eval_result = self.step_evaluate(
                model_version=model_version,
                checkpoint_id=summary["diffusion_checkpoint"],
            )
            if eval_result:
                summary["eval_result"] = eval_result
                summary["actions_taken"].append(f"eval:{eval_result.get('run_id')}")

        summary["duration_s"] = round(time.time() - t0, 1)
        summary["completed_at"] = time.time()

        # Persist cycle log
        log_path = self.state_dir / f"{cycle_id}.json"
        log_path.write_text(json.dumps(summary, indent=2))

        logger.info(
            f"Cycle {cycle_id} done in {summary['duration_s']:.0f}s | "
            f"actions: {', '.join(summary['actions_taken']) or 'none'}"
        )
        return summary

    # ── Continuous loop ───────────────────────────────────────

    def run_forever(
        self,
        check_interval_seconds: float = 3600.0,
        max_cycles: Optional[int] = None,
    ) -> None:
        """
        Run the orchestration loop indefinitely (or until max_cycles).
        check_interval_seconds: how often to poll for dataset updates.
        """
        logger.info(
            f"Orchestrator starting | check_interval={check_interval_seconds/3600:.1f}h "
            f"retrain_interval={self.retrain_interval_secs/86400:.1f}d"
        )
        cycle = 0
        while True:
            if max_cycles and cycle >= max_cycles:
                logger.info(f"Reached max_cycles={max_cycles}. Stopping.")
                break
            self.run_cycle()
            cycle += 1
            logger.info(f"Sleeping {check_interval_seconds/3600:.1f}h...")
            time.sleep(check_interval_seconds)

    def status(self) -> dict:
        return {
            "n_cycles": self.state.n_cycles,
            "current_dataset_version": self.state.current_dataset_version,
            "last_retrain": self.state.last_retrain_time,
            "last_eval": self.state.last_eval_run_id,
            "n_checkpoints": len(self.checkpoint_manager.list()),
            "n_dataset_versions": len(self.dataset_versioning.list_versions()),
        }
