"""
scripts/pipeline.py
--------------------
Self-improving pipeline entry point.

Wires all four new systems together into the Genesis feedback loop:

  DATA ENGINE → DATA QUALITY ENGINE → DATA STORAGE
        ↓
  MODEL TRAINING ENGINE → EVALUATION ENGINE → FEEDBACK ENGINE

Usage:
    # Run one cycle of the self-improvement loop
    python scripts/pipeline.py --mode cycle

    # Run continuous loop
    python scripts/pipeline.py --mode loop --interval 24

    # Run just evaluation on the current best checkpoint
    python scripts/pipeline.py --mode eval --model_version v0.2.0

    # Snapshot current dataset as a training version
    python scripts/pipeline.py --mode snapshot --version v3

    # Print governance audit log
    python scripts/pipeline.py --mode audit
"""
from __future__ import annotations
import argparse
import logging
import os
import sys
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import GenesisConfig
from core.logger import setup_logger
from core.model_registry import ModelRegistry
from core.model_versioning import ModelVersioningHub
from core.governance import ModelGovernance

from dataset.intelligence.deduplicator import Deduplicator
from dataset.intelligence.quality_filter import QualityFilter
from dataset.intelligence.embedding_index import EmbeddingIndex
from dataset.intelligence.metadata_manager import MetadataManager

from training.dataset_versioning import DatasetVersioning
from training.checkpoint_manager import CheckpointManager
from training.orchestrator import TrainingOrchestrator

from evaluation.clip_score import CLIPScorer
from evaluation.aesthetic_score import AestheticScorer
from evaluation.benchmark_runner import BenchmarkRunner
from evaluation.evaluation_report import EvaluationReporter

logger = logging.getLogger(__name__)


def build_pipeline(cfg):
    """Instantiate all Genesis subsystems from config."""
    device = cfg.system.get("device", "cpu")
    out    = cfg.system.get("output_dir", "outputs")

    # ── Dataset Intelligence ───────────────────────────────────
    deduplicator = Deduplicator(
        phash_threshold=cfg.get_nested("quality_filter.phash_threshold", 0.95),
        embedding_threshold=cfg.get_nested("quality_filter.embedding_threshold", 0.97),
        use_embedding_tier=cfg.get_nested("quality_filter.use_embedding_tier", False),
        device=device,
    )
    quality_filter = QualityFilter.from_config(cfg)
    embedding_index = EmbeddingIndex(
        index_dir=os.path.join(out, "embedding_index"),
        device=device,
    )
    metadata_manager = MetadataManager(
        db_path=os.path.join(out, "metadata", "dataset.db")
    )

    # ── Training Orchestration ─────────────────────────────────
    dataset_versioning = DatasetVersioning(
        versioned_root=os.path.join(out, "dataset_versions")
    )
    checkpoint_manager = CheckpointManager(
        checkpoint_root=os.path.join(out, "checkpoints", "diffusion"),
        max_to_keep=cfg.get_nested("training.max_checkpoints_to_keep", 5),
    )

    # ── Evaluation ─────────────────────────────────────────────
    clip_scorer      = CLIPScorer(device=device)
    aesthetic_scorer = AestheticScorer(device=device,
                                       use_laion=cfg.get_nested("eval.use_laion", False))
    benchmark_runner = BenchmarkRunner(
        cfg=cfg,
        generator=None,  # set below if running generation evals
        clip_scorer=clip_scorer,
        aesthetic_scorer=aesthetic_scorer,
        output_dir=os.path.join(out, "benchmarks"),
        images_per_prompt=cfg.get_nested("eval.images_per_prompt", 2),
        reference_dir=cfg.get_nested("eval.reference_dir", None),
    )
    eval_reporter = EvaluationReporter(
        benchmarks_dir=os.path.join(out, "benchmarks"),
        reports_dir=os.path.join(out, "eval_reports"),
    )

    # ── Model Governance ───────────────────────────────────────
    registry = ModelRegistry(
        registry_root=os.path.join(out, "models")
    )
    versioning_hub = ModelVersioningHub(
        lineage_dir=os.path.join(out, "models", "lineage")
    )
    governance = ModelGovernance(
        registry=registry,
        versioning_hub=versioning_hub,
        governance_dir=os.path.join(out, "governance"),
        mode=cfg.get_nested("governance.mode", "auto"),
        gate_config={
            "min_clip_score":     cfg.get_nested("governance.min_clip_score", 20.0),
            "min_aesthetic_score": cfg.get_nested("governance.min_aesthetic_score", 3.5),
            "max_fid_score":       cfg.get_nested("governance.max_fid_score", 80.0),
        },
    )

    # ── Orchestrator ───────────────────────────────────────────
    orchestrator = TrainingOrchestrator(
        cfg=cfg,
        dataset_versioning=dataset_versioning,
        checkpoint_manager=checkpoint_manager,
        benchmark_runner=benchmark_runner,
        model_registry=registry,
        state_dir=os.path.join(out, "orchestrator"),
        dataset_refresh_hours=cfg.get_nested("orchestrator.refresh_hours", 24.0),
        retrain_interval_days=cfg.get_nested("orchestrator.retrain_days", 14.0),
        min_new_samples_to_retrain=cfg.get_nested(
            "orchestrator.min_new_samples", 500),
    )

    return {
        "deduplicator": deduplicator,
        "quality_filter": quality_filter,
        "embedding_index": embedding_index,
        "metadata_manager": metadata_manager,
        "dataset_versioning": dataset_versioning,
        "checkpoint_manager": checkpoint_manager,
        "clip_scorer": clip_scorer,
        "aesthetic_scorer": aesthetic_scorer,
        "benchmark_runner": benchmark_runner,
        "eval_reporter": eval_reporter,
        "registry": registry,
        "versioning_hub": versioning_hub,
        "governance": governance,
        "orchestrator": orchestrator,
    }


def cmd_dedup(args, cfg, sys_):
    image_dir = args.image_dir or os.path.join(
        cfg.dataset.get("root", "outputs/dataset"), "images"
    )
    report = sys_["deduplicator"].run(
        image_dir=image_dir,
        report_path=args.report,
    )
    print(report.summary())


def cmd_quality_filter(args, cfg, sys_):
    import json
    index_path = args.index or os.path.join(
        cfg.dataset.get("root", "outputs/dataset"), "index.jsonl"
    )
    items = []
    with open(index_path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    accepted, rejected = sys_["quality_filter"].filter_batch(items)
    print(f"Accepted: {len(accepted)}, Rejected: {len(rejected)}")


def cmd_snapshot(args, cfg, sys_):
    source = args.source or cfg.dataset.get("root", "outputs/dataset")
    snap = sys_["dataset_versioning"].create_snapshot(
        source_dataset_root=source,
        version_name=args.version,
        filter_min_quality=args.min_quality,
        description=args.description or "",
    )
    print(f"Created snapshot: {snap.version} ({snap.n_samples} samples)")
    sys_["dataset_versioning"].print_versions()


def cmd_cycle(args, cfg, sys_):
    summary = sys_["orchestrator"].run_cycle()
    print(f"\nCycle complete: {summary}")


def cmd_loop(args, cfg, sys_):
    interval = args.interval * 3600
    sys_["orchestrator"].run_forever(
        check_interval_seconds=interval,
        max_cycles=args.max_cycles,
    )


def cmd_eval(args, cfg, sys_):
    version = args.model_version or "unknown"
    result  = sys_["orchestrator"].step_evaluate(version)
    if result:
        print(f"\nEvaluation: CLIP={result.get('mean_clip_score', 0):.3f} "
              f"AES={result.get('mean_aesthetic_score', 0):.3f}")
        # Compare to latest report
        versions = sys_["eval_reporter"].list_versions()
        if len(versions) >= 2:
            report = sys_["eval_reporter"].compare(
                versions[-2][0], versions[-1][0]
            )
            if report:
                sys_["eval_reporter"].print_report(report)


def cmd_audit(args, cfg, sys_):
    sys_["governance"].print_audit_log(n=20)
    sys_["registry"].print_registry()
    sys_["checkpoint_manager"].print_table()
    sys_["dataset_versioning"].print_versions()


def cmd_status(args, cfg, sys_):
    import json
    print(json.dumps(sys_["orchestrator"].status(), indent=2))


def main():
    parser = argparse.ArgumentParser(description="Genesis v0.3 Self-Improving Pipeline")
    parser.add_argument("--mode", default="status",
                        choices=["cycle","loop","eval","snapshot",
                                 "dedup","filter","audit","status"])
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--image_dir", default=None)
    parser.add_argument("--index", default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--model_version", default=None)
    parser.add_argument("--min_quality", type=float, default=0.3)
    parser.add_argument("--description", default="")
    parser.add_argument("--interval", type=float, default=24.0,
                        help="Loop interval in hours")
    parser.add_argument("--max_cycles", type=int, default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()

    setup_logger(level=args.log_level)
    cfg = GenesisConfig.load(args.config)
    sys_ = build_pipeline(cfg)

    dispatch = {
        "cycle":    cmd_cycle,
        "loop":     cmd_loop,
        "eval":     cmd_eval,
        "snapshot": cmd_snapshot,
        "dedup":    cmd_dedup,
        "filter":   cmd_quality_filter,
        "audit":    cmd_audit,
        "status":   cmd_status,
    }
    dispatch[args.mode](args, cfg, sys_)


if __name__ == "__main__":
    main()
