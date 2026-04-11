"""
training/checkpoint_manager.py
--------------------------------
Model checkpoint management: versioning, rollback, evaluation linking.

Checkpoint registry stored as JSON alongside checkpoint files.
Supports:
  - Saving checkpoints with full metadata
  - Loading best checkpoint by metric
  - Rollback to any prior checkpoint
  - Linking evaluation results to checkpoints
  - Automatic pruning of old checkpoints

Checkpoint directory layout:
  <checkpoint_root>/
    registry.json          — master index
    step_00005000/         — checkpoint at step 5000
      unet.pt              — model weights
      optimizer.pt         — optimizer state (optional)
      metadata.json        — step, loss, dataset_version, etc.
    step_00010000/
      ...
"""
from __future__ import annotations
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

_REGISTRY_FILE = "registry.json"
_BEST_METRICS  = {  # higher_is_better per metric
    "clip_score":       True,
    "aesthetic_score":  True,
    "fid_score":        False,
    "val_loss":         False,
}


class CheckpointMetadata:
    __slots__ = (
        "checkpoint_id", "step", "epoch", "model_type", "loss",
        "val_loss", "dataset_version", "eval_run_id",
        "eval_clip_score", "eval_aesthetic_score", "eval_fid_score",
        "created_at", "is_best", "tags", "notes",
    )

    def __init__(
        self,
        checkpoint_id: str,
        step: int,
        epoch: int = 0,
        model_type: str = "diffusion",
        loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        dataset_version: str = "unknown",
        eval_run_id: Optional[str] = None,
        eval_clip_score: Optional[float] = None,
        eval_aesthetic_score: Optional[float] = None,
        eval_fid_score: Optional[float] = None,
        created_at: Optional[float] = None,
        is_best: bool = False,
        tags: Optional[List[str]] = None,
        notes: str = "",
    ):
        self.checkpoint_id        = checkpoint_id
        self.step                 = step
        self.epoch                = epoch
        self.model_type           = model_type
        self.loss                 = loss
        self.val_loss             = val_loss
        self.dataset_version      = dataset_version
        self.eval_run_id          = eval_run_id
        self.eval_clip_score      = eval_clip_score
        self.eval_aesthetic_score = eval_aesthetic_score
        self.eval_fid_score       = eval_fid_score
        self.created_at           = created_at or time.time()
        self.is_best              = is_best
        self.tags                 = tags or []
        self.notes                = notes

    def to_dict(self) -> dict:
        return {s: getattr(self, s) for s in self.__slots__}

    @classmethod
    def from_dict(cls, d: dict) -> "CheckpointMetadata":
        return cls(**{k: d.get(k) for k in cls.__slots__})


class CheckpointManager:
    """
    Manages model checkpoints with versioning, evaluation linking, and rollback.

    Usage:
        cm = CheckpointManager("outputs/checkpoints/diffusion")
        cm.save(model, optimizer, step=5000, loss=0.42,
                dataset_version="v2")
        best = cm.best("eval_clip_score")
        cm.rollback(best.checkpoint_id, model)
    """

    def __init__(
        self,
        checkpoint_root: str,
        max_to_keep: int = 5,
    ):
        self.root = Path(checkpoint_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep
        self._registry: Dict[str, CheckpointMetadata] = {}
        self._load_registry()

    # ── Registry ──────────────────────────────────────────────

    def _registry_path(self) -> Path:
        return self.root / _REGISTRY_FILE

    def _load_registry(self) -> None:
        rp = self._registry_path()
        if rp.exists():
            try:
                data = json.loads(rp.read_text())
                for cid, meta in data.items():
                    self._registry[cid] = CheckpointMetadata.from_dict(meta)
                logger.info(f"Loaded registry: {len(self._registry)} checkpoints")
            except Exception as e:
                logger.warning(f"Registry load failed: {e}")

    def _save_registry(self) -> None:
        data = {cid: m.to_dict() for cid, m in self._registry.items()}
        self._registry_path().write_text(json.dumps(data, indent=2))

    # ── Save ──────────────────────────────────────────────────

    def save(
        self,
        model: torch.nn.Module,
        step: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        model_type: str = "diffusion",
        loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        dataset_version: str = "unknown",
        extra_state: Optional[dict] = None,
        tags: Optional[List[str]] = None,
        notes: str = "",
    ) -> CheckpointMetadata:
        """
        Save model checkpoint + metadata.
        Returns CheckpointMetadata for this checkpoint.
        """
        cid    = f"step_{step:08d}"
        cdir   = self.root / cid
        cdir.mkdir(exist_ok=True)

        # Save weights
        torch.save(model.state_dict(), cdir / "model.pt")

        # Save optimizer state
        if optimizer is not None:
            torch.save(optimizer.state_dict(), cdir / "optimizer.pt")

        # Save extra state (e.g. EMA, scaler)
        if extra_state:
            torch.save(extra_state, cdir / "extra_state.pt")

        meta = CheckpointMetadata(
            checkpoint_id=cid,
            step=step, epoch=epoch, model_type=model_type,
            loss=loss, val_loss=val_loss, dataset_version=dataset_version,
            tags=tags, notes=notes,
        )
        (cdir / "metadata.json").write_text(
            json.dumps(meta.to_dict(), indent=2)
        )

        self._registry[cid] = meta
        self._save_registry()

        logger.info(
            f"Checkpoint saved: {cid} | step={step} "
            f"loss={loss:.4f if loss else 'n/a'} "
            f"dataset={dataset_version}"
        )

        # Prune old checkpoints
        self._maybe_prune()
        return meta

    # ── Load ──────────────────────────────────────────────────

    def load(
        self,
        model: torch.nn.Module,
        checkpoint_id: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu",
        strict: bool = True,
    ) -> CheckpointMetadata:
        """Load weights from a checkpoint into model (in-place)."""
        cdir = self.root / checkpoint_id
        weights_path = cdir / "model.pt"
        if not weights_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

        state = torch.load(str(weights_path), map_location=device)
        model.load_state_dict(state, strict=strict)

        if optimizer is not None:
            opt_path = cdir / "optimizer.pt"
            if opt_path.exists():
                optimizer.load_state_dict(
                    torch.load(str(opt_path), map_location=device)
                )

        meta = self._registry.get(checkpoint_id)
        if meta is None:
            # Try loading from disk
            mp = cdir / "metadata.json"
            if mp.exists():
                meta = CheckpointMetadata.from_dict(json.loads(mp.read_text()))
        logger.info(f"Loaded checkpoint: {checkpoint_id}")
        return meta

    def load_latest(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu",
    ) -> Optional[CheckpointMetadata]:
        """Load the most recent checkpoint by step."""
        latest = self.latest()
        if latest is None:
            return None
        return self.load(model, latest.checkpoint_id, optimizer, device)

    # ── Query ─────────────────────────────────────────────────

    def list(self) -> List[CheckpointMetadata]:
        return sorted(self._registry.values(), key=lambda m: m.step)

    def latest(self) -> Optional[CheckpointMetadata]:
        metas = self.list()
        return metas[-1] if metas else None

    def best(self, metric: str = "eval_clip_score") -> Optional[CheckpointMetadata]:
        """
        Return the checkpoint with the best value of a given metric.
        Supports: eval_clip_score, eval_aesthetic_score, eval_fid_score, val_loss.
        """
        higher_is_better = _BEST_METRICS.get(metric.split("eval_")[-1], True)
        candidates = [
            m for m in self._registry.values()
            if getattr(m, metric, None) is not None
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda m: (
            getattr(m, metric) if higher_is_better
            else -getattr(m, metric)
        ))

    def get(self, checkpoint_id: str) -> Optional[CheckpointMetadata]:
        return self._registry.get(checkpoint_id)

    # ── Rollback ──────────────────────────────────────────────

    def rollback(
        self,
        checkpoint_id: str,
        model: torch.nn.Module,
        device: str = "cpu",
    ) -> CheckpointMetadata:
        """Roll back model to a specific checkpoint (for production deployments)."""
        logger.warning(f"Rolling back to checkpoint: {checkpoint_id}")
        return self.load(model, checkpoint_id, device=device)

    # ── Evaluation linking ────────────────────────────────────

    def link_evaluation(
        self,
        checkpoint_id: str,
        eval_run_id: str,
        clip_score: Optional[float] = None,
        aesthetic_score: Optional[float] = None,
        fid_score: Optional[float] = None,
    ) -> None:
        """Associate an evaluation run result with a checkpoint."""
        meta = self._registry.get(checkpoint_id)
        if meta is None:
            logger.warning(f"Cannot link eval: checkpoint {checkpoint_id} not found")
            return
        meta.eval_run_id          = eval_run_id
        meta.eval_clip_score      = clip_score
        meta.eval_aesthetic_score = aesthetic_score
        meta.eval_fid_score       = fid_score
        meta.is_best = False  # recomputed in best()
        self._save_registry()
        logger.info(
            f"Linked eval [{eval_run_id}] to {checkpoint_id}: "
            f"clip={clip_score} aes={aesthetic_score} fid={fid_score}"
        )

    # ── Pruning ───────────────────────────────────────────────

    def _maybe_prune(self) -> None:
        """Keep only the latest max_to_keep checkpoints (plus best-eval ones)."""
        all_sorted = self.list()  # ascending by step
        if len(all_sorted) <= self.max_to_keep:
            return

        # Always keep latest max_to_keep
        to_keep = set(m.checkpoint_id for m in all_sorted[-self.max_to_keep:])

        # Always keep best eval checkpoint
        best = self.best("eval_clip_score")
        if best:
            to_keep.add(best.checkpoint_id)

        to_delete = [m for m in all_sorted
                     if m.checkpoint_id not in to_keep]
        for m in to_delete:
            cdir = self.root / m.checkpoint_id
            if cdir.exists():
                shutil.rmtree(str(cdir))
                logger.debug(f"Pruned checkpoint: {m.checkpoint_id}")
            del self._registry[m.checkpoint_id]

        self._save_registry()

    def prune_all_except(self, keep_ids: List[str]) -> int:
        """Manually prune all checkpoints except specified IDs."""
        deleted = 0
        for cid in list(self._registry.keys()):
            if cid not in keep_ids:
                cdir = self.root / cid
                if cdir.exists():
                    shutil.rmtree(str(cdir))
                del self._registry[cid]
                deleted += 1
        self._save_registry()
        return deleted

    def stats(self) -> dict:
        metas = self.list()
        eval_linked = [m for m in metas if m.eval_run_id is not None]
        return {
            "n_checkpoints": len(metas),
            "latest_step": metas[-1].step if metas else None,
            "n_eval_linked": len(eval_linked),
            "root": str(self.root),
            "checkpoints": [
                {"id": m.checkpoint_id, "step": m.step,
                 "clip": m.eval_clip_score, "fid": m.eval_fid_score}
                for m in metas
            ],
        }

    def print_table(self) -> None:
        metas = self.list()
        print(f"\n{'='*70}")
        print(f"  Checkpoints ({self.root})")
        print(f"{'='*70}")
        hdr = f"{'ID':<20} {'Step':>8} {'Loss':>8} {'CLIP':>7} {'FID':>7} {'AES':>7}"
        print(f"  {hdr}")
        print(f"  {'-'*65}")
        for m in metas:
            best = self.best("eval_clip_score")
            star = " *" if (best and best.checkpoint_id == m.checkpoint_id) else ""
            print(
                f"  {m.checkpoint_id:<20} {m.step:>8,} "
                f"{f'{m.loss:.4f}' if m.loss else 'n/a':>8} "
                f"{f'{m.eval_clip_score:.2f}' if m.eval_clip_score else 'n/a':>7} "
                f"{f'{m.eval_fid_score:.1f}' if m.eval_fid_score else 'n/a':>7} "
                f"{f'{m.eval_aesthetic_score:.2f}' if m.eval_aesthetic_score else 'n/a':>7}"
                f"{star}"
            )
        print(f"{'='*70}\n")
