"""
evaluation/benchmark_runner.py
--------------------------------
Standardised benchmark pipeline: generate → evaluate → store results.

Standard benchmark prompts cover 5 categories:
  portrait, cityscape, animal, landscape, art

Each run:
  1. Generate N images per prompt using the active generator
  2. Compute FID vs reference set (if available)
  3. Compute CLIP score (prompt-image alignment)
  4. Compute aesthetic scores
  5. Persist results JSON to eval_cache/
"""
from __future__ import annotations
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from PIL import Image

logger = logging.getLogger(__name__)

_DEFAULT_PROMPTS: List[Dict] = [
    {"id": "portrait_01",   "category": "portrait",
     "prompt": "portrait of a woman, natural light, photorealistic, 4k"},
    {"id": "portrait_02",   "category": "portrait",
     "prompt": "elderly craftsman at work, detailed, cinematic"},
    {"id": "cityscape_01",  "category": "cityscape",
     "prompt": "cyberpunk city at night, neon lights, rain reflections"},
    {"id": "cityscape_02",  "category": "cityscape",
     "prompt": "futuristic skyline at sunset, golden hour, detailed"},
    {"id": "animal_01",     "category": "animal",
     "prompt": "golden retriever dog, studio photography, sharp focus"},
    {"id": "animal_02",     "category": "animal",
     "prompt": "wolf in a snowy forest, wildlife photography, dramatic"},
    {"id": "landscape_01",  "category": "landscape",
     "prompt": "mountain landscape at sunrise, misty valleys, photorealistic"},
    {"id": "landscape_02",  "category": "landscape",
     "prompt": "tropical beach, turquoise water, palm trees, HDR"},
    {"id": "art_01",        "category": "art",
     "prompt": "oil painting of a dragon in a stormy sea, fantasy art"},
    {"id": "art_02",        "category": "art",
     "prompt": "watercolor illustration of a forest fairy, detailed"},
]


@dataclass
class PromptResult:
    prompt_id: str
    prompt: str
    category: str
    images: List[str] = field(default_factory=list)   # saved paths
    clip_score: float = 0.0
    aesthetic_score: float = 0.0
    generation_time_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "prompt_id": self.prompt_id, "prompt": self.prompt,
            "category": self.category, "n_images": len(self.images),
            "images": self.images, "clip_score": round(self.clip_score, 3),
            "aesthetic_score": round(self.aesthetic_score, 3),
            "generation_time_s": round(self.generation_time_s, 2),
        }


@dataclass
class BenchmarkResult:
    run_id: str
    model_version: str
    timestamp: float = field(default_factory=time.time)
    prompt_results: List[PromptResult] = field(default_factory=list)
    fid_score: Optional[float] = None
    mean_clip_score: float = 0.0
    mean_aesthetic_score: float = 0.0
    total_time_s: float = 0.0
    config: dict = field(default_factory=dict)

    def compute_aggregates(self) -> None:
        if self.prompt_results:
            self.mean_clip_score = sum(
                r.clip_score for r in self.prompt_results
            ) / len(self.prompt_results)
            self.mean_aesthetic_score = sum(
                r.aesthetic_score for r in self.prompt_results
            ) / len(self.prompt_results)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "model_version": self.model_version,
            "timestamp": self.timestamp,
            "fid_score": self.fid_score,
            "mean_clip_score": round(self.mean_clip_score, 3),
            "mean_aesthetic_score": round(self.mean_aesthetic_score, 3),
            "total_time_s": round(self.total_time_s, 2),
            "config": self.config,
            "prompt_results": [r.to_dict() for r in self.prompt_results],
        }

    def summary(self) -> str:
        return (
            f"Benchmark [{self.run_id}] v={self.model_version} "
            f"FID={self.fid_score:.2f if self.fid_score else 'n/a'} "
            f"CLIP={self.mean_clip_score:.2f} "
            f"Aesthetic={self.mean_aesthetic_score:.2f} "
            f"time={self.total_time_s:.0f}s"
        )


class BenchmarkRunner:
    """
    Orchestrates generation + evaluation for model benchmarking.

    Usage:
        runner = BenchmarkRunner(cfg, generator, clip_scorer, aesthetic_scorer)
        result = runner.run("v0.2.0")
    """

    def __init__(
        self,
        cfg,
        generator,               # AdaptiveGenerator or CustomPipeline
        clip_scorer=None,        # CLIPScorer (optional)
        aesthetic_scorer=None,   # AestheticScorer (optional)
        fid_scorer=None,         # FIDScorer (optional)
        output_dir: str = "outputs/benchmarks",
        prompts: Optional[List[Dict]] = None,
        images_per_prompt: int = 2,
        reference_dir: Optional[str] = None,
    ):
        self.cfg = cfg
        self.generator = generator
        self.clip_scorer = clip_scorer
        self.aesthetic_scorer = aesthetic_scorer
        self.fid_scorer = fid_scorer
        self.output_dir = Path(output_dir)
        self.prompts = prompts or _DEFAULT_PROMPTS
        self.images_per_prompt = images_per_prompt
        self.reference_dir = reference_dir

    def run(
        self,
        model_version: str,
        run_id: Optional[str] = None,
        max_prompts: Optional[int] = None,
    ) -> BenchmarkResult:
        """
        Execute full benchmark for a model version.
        Returns BenchmarkResult with all metrics.
        """
        import uuid
        run_id = run_id or f"bench_{model_version}_{int(time.time())}"
        run_dir = self.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        prompts = self.prompts[:max_prompts] if max_prompts else self.prompts
        result = BenchmarkResult(
            run_id=run_id, model_version=model_version,
            config={"images_per_prompt": self.images_per_prompt,
                    "n_prompts": len(prompts)}
        )

        logger.info(f"Benchmark [{run_id}] | {len(prompts)} prompts | v={model_version}")

        # ── Generate images ────────────────────────────────────
        all_gen_paths: List[str] = []
        for entry in prompts:
            prompt_id = entry["id"]
            prompt    = entry["prompt"]
            category  = entry.get("category", "unknown")
            prompt_dir = run_dir / prompt_id
            prompt_dir.mkdir(exist_ok=True)

            t_gen = time.time()
            try:
                images = self.generator.generate(
                    prompt=prompt, num_images=self.images_per_prompt
                )
            except Exception as e:
                logger.error(f"Generation failed [{prompt_id}]: {e}")
                images = []

            saved_paths = []
            for i, img in enumerate(images):
                p = str(prompt_dir / f"{i:03d}.png")
                try:
                    img.save(p)
                    saved_paths.append(p)
                    all_gen_paths.append(p)
                except Exception:
                    pass

            gen_time = time.time() - t_gen

            # Evaluate this prompt's images
            pr = PromptResult(
                prompt_id=prompt_id, prompt=prompt,
                category=category, images=saved_paths,
                generation_time_s=gen_time,
            )

            if self.clip_scorer and saved_paths:
                try:
                    scores = self.clip_scorer.score_batch(
                        [prompt] * len(saved_paths), saved_paths
                    )
                    pr.clip_score = sum(scores) / len(scores)
                except Exception as e:
                    logger.warning(f"CLIP score failed: {e}")

            if self.aesthetic_scorer and saved_paths:
                try:
                    scores = self.aesthetic_scorer.score_batch(saved_paths)
                    pr.aesthetic_score = sum(scores) / len(scores)
                except Exception as e:
                    logger.warning(f"Aesthetic score failed: {e}")

            result.prompt_results.append(pr)
            logger.info(
                f"  [{prompt_id}] clip={pr.clip_score:.2f} "
                f"aes={pr.aesthetic_score:.2f} {gen_time:.0f}s"
            )

        # ── FID (needs reference set) ──────────────────────────
        if self.fid_scorer and self.reference_dir and len(all_gen_paths) >= 4:
            try:
                # Collect generated images into a temp dir for FID
                gen_dir = str(run_dir / "_all_generated")
                os.makedirs(gen_dir, exist_ok=True)
                for i, p in enumerate(all_gen_paths):
                    import shutil
                    shutil.copy(p, os.path.join(gen_dir, f"{i:06d}.png"))
                result.fid_score = self.fid_scorer.compute(
                    real_dir=self.reference_dir,
                    generated_dir=gen_dir,
                )
            except Exception as e:
                logger.warning(f"FID failed: {e}")

        result.total_time_s = time.time() - t0
        result.compute_aggregates()

        # ── Save results ───────────────────────────────────────
        results_path = run_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info(f"Benchmark complete → {results_path}")
        logger.info(result.summary())
        return result

    def load_result(self, run_id: str) -> Optional[BenchmarkResult]:
        p = self.output_dir / run_id / "results.json"
        if not p.exists():
            return None
        with open(p) as f:
            data = json.load(f)
        r = BenchmarkResult(
            run_id=data["run_id"],
            model_version=data["model_version"],
            timestamp=data["timestamp"],
            fid_score=data.get("fid_score"),
            mean_clip_score=data["mean_clip_score"],
            mean_aesthetic_score=data["mean_aesthetic_score"],
            total_time_s=data["total_time_s"],
            config=data.get("config", {}),
        )
        r.prompt_results = [
            PromptResult(
                prompt_id=pr["prompt_id"], prompt=pr["prompt"],
                category=pr["category"], images=pr["images"],
                clip_score=pr["clip_score"],
                aesthetic_score=pr["aesthetic_score"],
                generation_time_s=pr["generation_time_s"],
            )
            for pr in data.get("prompt_results", [])
        ]
        return r

    def list_runs(self) -> List[str]:
        if not self.output_dir.exists():
            return []
        return sorted(
            d.name for d in self.output_dir.iterdir()
            if d.is_dir() and (d / "results.json").exists()
        )
