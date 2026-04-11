"""
evaluation/evaluation_report.py
---------------------------------
Structured evaluation report comparing model versions over time.

Reads BenchmarkResult JSON files, computes deltas between versions,
flags regressions, produces human-readable + machine-readable output.

Usage:
    reporter = EvaluationReporter("outputs/benchmarks")
    report = reporter.compare("v0.1.0", "v0.2.0")
    reporter.print_report(report)
    reporter.save_report(report, "outputs/eval_reports/v0.1_vs_v0.2.json")
"""
from __future__ import annotations
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class MetricDelta:
    metric: str
    baseline_value: Optional[float]
    current_value: Optional[float]
    delta: Optional[float]       # current - baseline (or flipped for FID)
    delta_pct: Optional[float]   # percentage change
    improved: Optional[bool]     # True = better, False = worse, None = n/a
    is_regression: bool = False  # True if performance dropped below threshold

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "baseline": self.baseline_value,
            "current": self.current_value,
            "delta": round(self.delta, 4) if self.delta is not None else None,
            "delta_pct": round(self.delta_pct, 2) if self.delta_pct is not None else None,
            "improved": self.improved,
            "is_regression": self.is_regression,
        }


@dataclass
class PerPromptDelta:
    prompt_id: str
    category: str
    clip_delta: float
    aesthetic_delta: float
    baseline_clip: float
    current_clip: float

    def to_dict(self) -> dict:
        return {
            "prompt_id": self.prompt_id,
            "category": self.category,
            "clip_delta": round(self.clip_delta, 3),
            "aesthetic_delta": round(self.aesthetic_delta, 3),
            "baseline_clip": round(self.baseline_clip, 3),
            "current_clip": round(self.current_clip, 3),
        }


@dataclass
class ComparisonReport:
    baseline_version: str
    current_version:  str
    baseline_run_id:  str
    current_run_id:   str
    generated_at:     float = field(default_factory=time.time)
    metric_deltas:    List[MetricDelta] = field(default_factory=list)
    per_prompt:       List[PerPromptDelta] = field(default_factory=list)
    regressions:      List[str] = field(default_factory=list)
    improvements:     List[str] = field(default_factory=list)
    verdict:          str = "unknown"  # improved | regressed | neutral | mixed
    notes:            List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "baseline_version": self.baseline_version,
            "current_version":  self.current_version,
            "baseline_run_id":  self.baseline_run_id,
            "current_run_id":   self.current_run_id,
            "generated_at":     self.generated_at,
            "verdict":          self.verdict,
            "regressions":      self.regressions,
            "improvements":     self.improvements,
            "notes":            self.notes,
            "metric_deltas":    [d.to_dict() for d in self.metric_deltas],
            "per_prompt":       [p.to_dict() for p in self.per_prompt],
        }


@dataclass
class SingleVersionReport:
    """Standalone report for a single model version (no baseline)."""
    version: str
    run_id:  str
    generated_at: float = field(default_factory=time.time)
    fid_score: Optional[float] = None
    mean_clip_score: float = 0.0
    mean_aesthetic_score: float = 0.0
    total_images: int = 0
    per_category: Dict[str, dict] = field(default_factory=dict)
    grade: str = "unknown"  # A/B/C/D/F based on composite

    def to_dict(self) -> dict:
        return {
            "version": self.version, "run_id": self.run_id,
            "generated_at": self.generated_at,
            "fid_score": self.fid_score,
            "mean_clip_score": round(self.mean_clip_score, 3),
            "mean_aesthetic_score": round(self.mean_aesthetic_score, 3),
            "total_images": self.total_images,
            "grade": self.grade,
            "per_category": self.per_category,
        }


# ── Thresholds for regression detection ───────────────────────

_REGRESSION_THRESHOLDS = {
    "clip_score": {
        "min_delta": -2.0,   # >2 point drop = regression
        "pct_drop_alert": -5.0,
    },
    "aesthetic_score": {
        "min_delta": -0.3,
        "pct_drop_alert": -5.0,
    },
    "fid_score": {
        "max_delta": 5.0,    # FID: increase is bad
        "pct_rise_alert": 15.0,
    },
}


def _grade(clip: float, aesthetic: float,
           fid: Optional[float]) -> str:
    score = clip * 0.4 + aesthetic * 10 * 0.4
    if fid is not None:
        fid_score = max(0, 10 - fid / 5) * 0.2
        score += fid_score
    if score >= 8:  return "A"
    if score >= 6:  return "B"
    if score >= 4:  return "C"
    if score >= 2:  return "D"
    return "F"


# ── EvaluationReporter ────────────────────────────────────────

class EvaluationReporter:
    """
    Reads benchmark results from disk and produces comparison reports.
    """

    def __init__(self, benchmarks_dir: str = "outputs/benchmarks",
                 reports_dir: str = "outputs/eval_reports"):
        self.benchmarks_dir = Path(benchmarks_dir)
        self.reports_dir    = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def _load_run(self, run_id: str) -> Optional[dict]:
        p = self.benchmarks_dir / run_id / "results.json"
        if not p.exists():
            logger.error(f"Benchmark run not found: {p}")
            return None
        with open(p) as f:
            return json.load(f)

    def _latest_run_for_version(self, version: str) -> Optional[dict]:
        """Find most recent benchmark run for a given model version."""
        candidates = []
        if not self.benchmarks_dir.exists():
            return None
        for d in self.benchmarks_dir.iterdir():
            rp = d / "results.json"
            if not rp.exists():
                continue
            with open(rp) as f:
                data = json.load(f)
            if data.get("model_version") == version:
                candidates.append((data["timestamp"], data))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][1]

    def list_versions(self) -> List[Tuple[str, str, float]]:
        """Return [(version, run_id, timestamp)] sorted by time."""
        results = []
        if not self.benchmarks_dir.exists():
            return results
        for d in self.benchmarks_dir.iterdir():
            rp = d / "results.json"
            if not rp.exists():
                continue
            with open(rp) as f:
                data = json.load(f)
            results.append((
                data.get("model_version", "unknown"),
                data.get("run_id", d.name),
                data.get("timestamp", 0.0),
            ))
        results.sort(key=lambda x: x[2], reverse=True)
        return results

    def single_version_report(
        self, version: str, run_id: Optional[str] = None
    ) -> Optional[SingleVersionReport]:
        """Generate a standalone report for one model version."""
        data = (self._load_run(run_id) if run_id
                else self._latest_run_for_version(version))
        if data is None:
            return None

        per_cat: Dict[str, Dict] = {}
        total_images = 0
        for pr in data.get("prompt_results", []):
            cat = pr.get("category", "unknown")
            if cat not in per_cat:
                per_cat[cat] = {"clip_scores": [], "aesthetic_scores": [],
                                "n_prompts": 0, "n_images": 0}
            per_cat[cat]["clip_scores"].append(pr["clip_score"])
            per_cat[cat]["aesthetic_scores"].append(pr["aesthetic_score"])
            per_cat[cat]["n_prompts"] += 1
            per_cat[cat]["n_images"] += pr.get("n_images", 0)
            total_images += pr.get("n_images", 0)

        cat_summary = {}
        for cat, v in per_cat.items():
            cat_summary[cat] = {
                "mean_clip": round(sum(v["clip_scores"]) / len(v["clip_scores"]), 3),
                "mean_aesthetic": round(
                    sum(v["aesthetic_scores"]) / len(v["aesthetic_scores"]), 3),
                "n_prompts": v["n_prompts"],
                "n_images": v["n_images"],
            }

        fid = data.get("fid_score")
        clip = data.get("mean_clip_score", 0.0)
        aes  = data.get("mean_aesthetic_score", 0.0)

        return SingleVersionReport(
            version=data.get("model_version", version),
            run_id=data.get("run_id", "unknown"),
            fid_score=fid,
            mean_clip_score=clip,
            mean_aesthetic_score=aes,
            total_images=total_images,
            per_category=cat_summary,
            grade=_grade(clip, aes, fid),
        )

    def compare(
        self,
        baseline_version: str,
        current_version: str,
        baseline_run_id: Optional[str] = None,
        current_run_id: Optional[str] = None,
    ) -> Optional[ComparisonReport]:
        """
        Compare two model versions.
        Loads most recent benchmark run for each unless run_id specified.
        """
        baseline_data = (self._load_run(baseline_run_id) if baseline_run_id
                         else self._latest_run_for_version(baseline_version))
        current_data  = (self._load_run(current_run_id)  if current_run_id
                         else self._latest_run_for_version(current_version))

        if baseline_data is None:
            logger.error(f"No benchmark data for baseline: {baseline_version}")
            return None
        if current_data is None:
            logger.error(f"No benchmark data for current: {current_version}")
            return None

        report = ComparisonReport(
            baseline_version=baseline_data.get("model_version", baseline_version),
            current_version=current_data.get("model_version", current_version),
            baseline_run_id=baseline_data.get("run_id", ""),
            current_run_id=current_data.get("run_id", ""),
        )

        def _delta_metric(name: str, b_val, c_val,
                          higher_is_better: bool = True) -> MetricDelta:
            if b_val is None or c_val is None:
                return MetricDelta(name, b_val, c_val, None, None, None)
            raw_delta = c_val - b_val
            pct = (raw_delta / max(abs(b_val), 1e-9)) * 100
            improved = raw_delta > 0 if higher_is_better else raw_delta < 0
            # Regression detection
            is_reg = False
            if name in _REGRESSION_THRESHOLDS:
                t = _REGRESSION_THRESHOLDS[name]
                if higher_is_better:
                    is_reg = raw_delta < t.get("min_delta", -999)
                else:
                    is_reg = raw_delta > t.get("max_delta", 999)
            return MetricDelta(name, b_val, c_val, raw_delta, pct, improved, is_reg)

        # ── Top-level metrics ──────────────────────────────────
        report.metric_deltas.append(_delta_metric(
            "clip_score",
            baseline_data.get("mean_clip_score"),
            current_data.get("mean_clip_score"),
            higher_is_better=True,
        ))
        report.metric_deltas.append(_delta_metric(
            "aesthetic_score",
            baseline_data.get("mean_aesthetic_score"),
            current_data.get("mean_aesthetic_score"),
            higher_is_better=True,
        ))
        report.metric_deltas.append(_delta_metric(
            "fid_score",
            baseline_data.get("fid_score"),
            current_data.get("fid_score"),
            higher_is_better=False,
        ))

        # ── Per-prompt comparison ──────────────────────────────
        b_prompts = {pr["prompt_id"]: pr
                     for pr in baseline_data.get("prompt_results", [])}
        c_prompts = {pr["prompt_id"]: pr
                     for pr in current_data.get("prompt_results", [])}

        for pid, c_pr in c_prompts.items():
            if pid not in b_prompts:
                continue
            b_pr = b_prompts[pid]
            report.per_prompt.append(PerPromptDelta(
                prompt_id=pid,
                category=c_pr.get("category", "unknown"),
                clip_delta=c_pr["clip_score"] - b_pr["clip_score"],
                aesthetic_delta=c_pr["aesthetic_score"] - b_pr["aesthetic_score"],
                baseline_clip=b_pr["clip_score"],
                current_clip=c_pr["clip_score"],
            ))

        # ── Regressions + improvements ─────────────────────────
        for d in report.metric_deltas:
            if d.is_regression:
                report.regressions.append(
                    f"{d.metric}: {d.baseline_value:.3f} → {d.current_value:.3f} "
                    f"({d.delta:+.3f}, {d.delta_pct:+.1f}%)"
                )
            elif d.improved:
                report.improvements.append(
                    f"{d.metric}: {d.baseline_value:.3f} → {d.current_value:.3f} "
                    f"({d.delta:+.3f})"
                )

        # ── Verdict ────────────────────────────────────────────
        n_improved   = sum(1 for d in report.metric_deltas if d.improved is True)
        n_regressed  = sum(1 for d in report.metric_deltas if d.is_regression)
        n_applicable = sum(1 for d in report.metric_deltas if d.improved is not None)

        if n_applicable == 0:
            report.verdict = "insufficient_data"
        elif n_regressed > 0 and n_improved == 0:
            report.verdict = "regressed"
        elif n_regressed > 0 and n_improved > 0:
            report.verdict = "mixed"
        elif n_improved == n_applicable:
            report.verdict = "improved"
        elif n_improved > 0:
            report.verdict = "mostly_improved"
        else:
            report.verdict = "neutral"

        if report.regressions:
            report.notes.append(
                f"ALERT: {len(report.regressions)} metric(s) regressed. "
                f"Review before promoting to production."
            )
        if report.verdict in ("improved", "mostly_improved"):
            report.notes.append("Model shows improvement over baseline.")

        return report

    def save_report(
        self,
        report: ComparisonReport,
        output_path: Optional[str] = None,
    ) -> str:
        if output_path is None:
            fname = (f"compare_{report.baseline_version}_vs_"
                     f"{report.current_version}_{int(report.generated_at)}.json")
            output_path = str(self.reports_dir / fname)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info(f"Report saved → {output_path}")
        return output_path

    def print_report(self, report: ComparisonReport) -> None:
        W = 60
        print(f"\n{'='*W}")
        print(f"  GENESIS EVALUATION REPORT")
        print(f"  Baseline : {report.baseline_version}  ({report.baseline_run_id})")
        print(f"  Current  : {report.current_version}  ({report.current_run_id})")
        print(f"  Verdict  : {report.verdict.upper()}")
        print(f"{'='*W}")

        print("\n  METRICS:")
        for d in report.metric_deltas:
            if d.delta is None:
                print(f"    {d.metric:<22} n/a")
                continue
            arrow = "↑" if d.improved else "↓"
            flag  = " ⚠ REGRESSION" if d.is_regression else ""
            print(f"    {d.metric:<22} {d.baseline_value:.3f} → {d.current_value:.3f}"
                  f"  ({d.delta:+.3f}) {arrow}{flag}")

        if report.regressions:
            print("\n  ⚠ REGRESSIONS:")
            for r in report.regressions:
                print(f"    - {r}")

        if report.improvements:
            print("\n  ✓ IMPROVEMENTS:")
            for r in report.improvements:
                print(f"    + {r}")

        print("\n  PER-CATEGORY (CLIP delta):")
        cat_deltas: Dict[str, List[float]] = {}
        for pp in report.per_prompt:
            cat_deltas.setdefault(pp.category, []).append(pp.clip_delta)
        for cat, deltas in sorted(cat_deltas.items()):
            mean = sum(deltas) / len(deltas)
            print(f"    {cat:<20} {mean:+.3f}")

        if report.notes:
            print("\n  NOTES:")
            for n in report.notes:
                print(f"    {n}")
        print(f"{'='*W}\n")

    def print_single(self, report: SingleVersionReport) -> None:
        W = 60
        print(f"\n{'='*W}")
        print(f"  GENESIS EVAL: {report.version}  [Grade: {report.grade}]")
        print(f"{'='*W}")
        print(f"  CLIP Score   : {report.mean_clip_score:.3f}")
        print(f"  Aesthetic    : {report.mean_aesthetic_score:.3f}")
        if report.fid_score is not None:
            print(f"  FID          : {report.fid_score:.2f}")
        print(f"  Total Images : {report.total_images}")
        print("\n  BY CATEGORY:")
        for cat, v in sorted(report.per_category.items()):
            print(f"    {cat:<20} clip={v['mean_clip']:.3f} "
                  f"aes={v['mean_aesthetic']:.3f}")
        print(f"{'='*W}\n")
