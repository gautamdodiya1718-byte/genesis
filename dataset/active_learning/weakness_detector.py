"""
dataset/active_learning/weakness_detector.py
----------------------------------------------
Analyzes evaluation results to find model weaknesses by category.

Reads benchmark BenchmarkResult JSON files and identifies:
  - Categories with below-threshold CLIP scores
  - Categories with below-threshold aesthetic scores
  - Categories showing regression vs. prior runs
  - Prompt types with highest variance (unstable generation)

Output is a WeaknessReport consumed by dataset_expander.py
to drive targeted crawling.

Weakness taxonomy:
  semantic   — model doesn't understand the concept (low CLIP)
  aesthetic  — model generates ugly/blurry results (low aesthetic)
  regression — metric dropped since last eval (delta analysis)
  variance   — highly inconsistent outputs within category
"""
from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Per-category thresholds — below these values = weakness
_DEFAULT_THRESHOLDS = {
    "clip_score":       22.0,
    "aesthetic_score":   4.0,
    "clip_variance_max": 8.0,   # std dev above this = unstable category
}

_SEVERITY_LEVELS = {
    "critical": 0,   # score < threshold * 0.70
    "high":     1,   # score < threshold * 0.85
    "medium":   2,   # score < threshold
    "low":      3,   # slight below threshold or high variance
}


@dataclass
class CategoryWeakness:
    category: str
    weakness_type: str     # semantic | aesthetic | regression | variance
    severity: str          # critical | high | medium | low
    score: float
    threshold: float
    delta: Optional[float] = None   # vs. baseline (negative = regression)
    prompt_ids: List[str] = field(default_factory=list)
    notes: str = ""

    @property
    def deficit(self) -> float:
        """How far below threshold. Larger = worse."""
        return max(0.0, self.threshold - self.score)

    @property
    def priority(self) -> float:
        """Higher = needs more data collection attention."""
        severity_weights = {"critical": 4.0, "high": 3.0, "medium": 2.0, "low": 1.0}
        return severity_weights.get(self.severity, 1.0) * (1.0 + self.deficit)

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "weakness_type": self.weakness_type,
            "severity": self.severity,
            "score": round(self.score, 3),
            "threshold": self.threshold,
            "delta": round(self.delta, 3) if self.delta is not None else None,
            "deficit": round(self.deficit, 3),
            "priority": round(self.priority, 3),
            "prompt_ids": self.prompt_ids,
            "notes": self.notes,
        }


@dataclass
class WeaknessReport:
    run_id: str
    model_version: str
    baseline_run_id: Optional[str]
    weaknesses: List[CategoryWeakness] = field(default_factory=list)
    strong_categories: List[str] = field(default_factory=list)
    overall_clip: float = 0.0
    overall_aesthetic: float = 0.0
    total_categories: int = 0

    @property
    def weak_categories(self) -> List[str]:
        return list({w.category for w in self.weaknesses})

    @property
    def critical_categories(self) -> List[str]:
        return list({w.category for w in self.weaknesses if w.severity == "critical"})

    def top_weaknesses(self, n: int = 5) -> List[CategoryWeakness]:
        return sorted(self.weaknesses, key=lambda w: w.priority, reverse=True)[:n]

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "model_version": self.model_version,
            "baseline_run_id": self.baseline_run_id,
            "overall_clip": round(self.overall_clip, 3),
            "overall_aesthetic": round(self.overall_aesthetic, 3),
            "total_categories": self.total_categories,
            "weak_categories": self.weak_categories,
            "strong_categories": self.strong_categories,
            "weaknesses": [w.to_dict() for w in self.weaknesses],
        }

    def print_summary(self) -> None:
        W = 60
        print(f"\n{'='*W}")
        print(f"  Weakness Report — {self.model_version}")
        print(f"{'='*W}")
        print(f"  Overall CLIP={self.overall_clip:.2f}  AES={self.overall_aesthetic:.2f}")
        print(f"  Weak categories: {', '.join(self.weak_categories) or 'none'}")
        print(f"\n  Top weaknesses by priority:")
        for w in self.top_weaknesses(8):
            icon = {"critical":"🔴","high":"🟠","medium":"🟡","low":"🟢"}.get(w.severity,"?")
            delta_str = f" Δ{w.delta:+.2f}" if w.delta is not None else ""
            print(f"  {icon} [{w.severity:<8}] {w.category:<18} "
                  f"{w.weakness_type:<12} score={w.score:.2f}{delta_str}")
        print(f"{'='*W}\n")


def _severity(score: float, threshold: float) -> str:
    if score < threshold * 0.70:  return "critical"
    if score < threshold * 0.85:  return "high"
    if score < threshold:          return "medium"
    return "low"


class WeaknessDetector:
    """
    Parses benchmark results and produces a WeaknessReport.

    Usage:
        detector = WeaknessDetector(benchmarks_dir="outputs/benchmarks")
        report = detector.analyze(run_id="bench_v0.2.0_1234567")
        # Or compare two runs:
        report = detector.analyze_with_baseline(
            current_run_id="bench_v0.2.0_...",
            baseline_run_id="bench_v0.1.0_...",
        )
    """

    def __init__(
        self,
        benchmarks_dir: str = "outputs/benchmarks",
        thresholds: Optional[Dict[str, float]] = None,
    ):
        self.benchmarks_dir = Path(benchmarks_dir)
        self.thresholds = {**_DEFAULT_THRESHOLDS, **(thresholds or {})}

    def _load(self, run_id: str) -> Optional[dict]:
        p = self.benchmarks_dir / run_id / "results.json"
        if not p.exists():
            logger.error(f"Benchmark not found: {p}")
            return None
        return json.loads(p.read_text())

    def _latest_run(self) -> Optional[dict]:
        if not self.benchmarks_dir.exists():
            return None
        runs = sorted(
            [d for d in self.benchmarks_dir.iterdir()
             if (d / "results.json").exists()],
            key=lambda d: json.loads((d / "results.json").read_text()).get("timestamp", 0),
            reverse=True,
        )
        if not runs:
            return None
        return json.loads((runs[0] / "results.json").read_text())

    def analyze(
        self,
        run_id: Optional[str] = None,
        baseline_run_id: Optional[str] = None,
    ) -> Optional[WeaknessReport]:
        """
        Analyze a single benchmark run (optionally vs baseline).
        Loads latest run if run_id is None.
        """
        data = self._load(run_id) if run_id else self._latest_run()
        if data is None:
            return None
        baseline = self._load(baseline_run_id) if baseline_run_id else None
        return self._build_report(data, baseline)

    def analyze_with_baseline(
        self,
        current_run_id: str,
        baseline_run_id: str,
    ) -> Optional[WeaknessReport]:
        current  = self._load(current_run_id)
        baseline = self._load(baseline_run_id)
        if current is None:
            return None
        return self._build_report(current, baseline)

    def _build_report(self, data: dict, baseline: Optional[dict]) -> WeaknessReport:
        report = WeaknessReport(
            run_id=data.get("run_id", "unknown"),
            model_version=data.get("model_version", "unknown"),
            baseline_run_id=baseline.get("run_id") if baseline else None,
            overall_clip=data.get("mean_clip_score", 0.0),
            overall_aesthetic=data.get("mean_aesthetic_score", 0.0),
        )

        # Group prompt results by category
        cat_clips:   Dict[str, List[Tuple[str, float]]] = {}
        cat_aes:     Dict[str, List[Tuple[str, float]]] = {}
        for pr in data.get("prompt_results", []):
            cat = pr.get("category", "unknown")
            pid = pr.get("prompt_id", "")
            cat_clips.setdefault(cat, []).append((pid, pr.get("clip_score", 0.0)))
            cat_aes.setdefault(cat, []).append((pid, pr.get("aesthetic_score", 0.0)))

        report.total_categories = len(cat_clips)

        # Baseline category scores for delta computation
        base_cat_clips: Dict[str, float] = {}
        base_cat_aes:   Dict[str, float] = {}
        if baseline:
            for pr in baseline.get("prompt_results", []):
                cat = pr.get("category", "unknown")
                base_cat_clips.setdefault(cat, [])
                base_cat_clips[cat] = pr.get("clip_score", 0.0)
                base_cat_aes.setdefault(cat, [])
                base_cat_aes[cat] = pr.get("aesthetic_score", 0.0)

        clip_thresh = self.thresholds["clip_score"]
        aes_thresh  = self.thresholds["aesthetic_score"]
        var_thresh  = self.thresholds["clip_variance_max"]

        for cat, clip_pairs in cat_clips.items():
            aes_pairs  = cat_aes.get(cat, [])
            clip_vals  = [v for _, v in clip_pairs]
            aes_vals   = [v for _, v in aes_pairs]
            pids       = [pid for pid, _ in clip_pairs]
            mean_clip  = statistics.mean(clip_vals) if clip_vals else 0.0
            mean_aes   = statistics.mean(aes_vals)  if aes_vals  else 0.0
            clip_std   = statistics.stdev(clip_vals) if len(clip_vals) > 1 else 0.0

            is_weak = False

            # Semantic weakness (CLIP)
            if mean_clip < clip_thresh:
                delta = None
                if cat in base_cat_clips:
                    delta = mean_clip - base_cat_clips[cat]
                report.weaknesses.append(CategoryWeakness(
                    category=cat,
                    weakness_type="semantic",
                    severity=_severity(mean_clip, clip_thresh),
                    score=mean_clip,
                    threshold=clip_thresh,
                    delta=delta,
                    prompt_ids=pids,
                    notes=f"CLIP {mean_clip:.2f} < {clip_thresh:.2f}",
                ))
                is_weak = True

            # Aesthetic weakness
            if mean_aes < aes_thresh:
                delta = None
                if cat in base_cat_aes:
                    delta = mean_aes - base_cat_aes[cat]
                report.weaknesses.append(CategoryWeakness(
                    category=cat,
                    weakness_type="aesthetic",
                    severity=_severity(mean_aes, aes_thresh),
                    score=mean_aes,
                    threshold=aes_thresh,
                    delta=delta,
                    prompt_ids=pids,
                    notes=f"Aesthetic {mean_aes:.2f} < {aes_thresh:.2f}",
                ))
                is_weak = True

            # Regression vs baseline
            if baseline and cat in base_cat_clips:
                delta = mean_clip - base_cat_clips[cat]
                if delta < -2.0:  # meaningful regression
                    report.weaknesses.append(CategoryWeakness(
                        category=cat,
                        weakness_type="regression",
                        severity=_severity(mean_clip, base_cat_clips[cat]),
                        score=mean_clip,
                        threshold=base_cat_clips[cat],
                        delta=delta,
                        prompt_ids=pids,
                        notes=f"CLIP dropped {delta:+.2f} vs baseline",
                    ))
                    is_weak = True

            # Variance (unstable category)
            if clip_std > var_thresh:
                report.weaknesses.append(CategoryWeakness(
                    category=cat,
                    weakness_type="variance",
                    severity="medium" if clip_std < var_thresh * 1.5 else "high",
                    score=clip_std,
                    threshold=var_thresh,
                    prompt_ids=pids,
                    notes=f"CLIP std={clip_std:.2f} (unstable generation)",
                ))
                is_weak = True

            if not is_weak:
                report.strong_categories.append(cat)

        return report

    def save_report(self, report: WeaknessReport, output_path: str) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info(f"Weakness report → {output_path}")
