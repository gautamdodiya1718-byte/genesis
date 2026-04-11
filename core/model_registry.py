"""
core/model_registry.py
------------------------
Central model registry for Genesis. Tracks all model artifacts,
their versions, evaluation results, and deployment status.

Each model type (diffusion, vae, caption, embedding) is registered
independently. The registry is the single source of truth for which
model version is currently in production.

Registry storage: JSON file (one per model type).
  models/
    diffusion/
      registry.json
    vae/
      registry.json
    caption/
      registry.json
    embedding/
      registry.json

Usage:
    registry = ModelRegistry("models/")
    registry.register("diffusion", "v0.2.0",
                      checkpoint_path="outputs/checkpoints/step_10000",
                      dataset_version="v2")
    current = registry.get_production("diffusion")
    registry.promote("diffusion", "v0.2.0")
"""
from __future__ import annotations
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_MODEL_TYPES = {"diffusion", "vae", "caption", "embedding", "custom"}


class ModelEntry:
    """One registered model version."""

    def __init__(
        self,
        version: str,
        model_type: str,
        checkpoint_path: str = "",
        dataset_version: str = "unknown",
        created_at: Optional[float] = None,
        status: str = "staged",       # staged | production | archived | failed
        eval_run_id: Optional[str]  = None,
        eval_clip_score: Optional[float] = None,
        eval_aesthetic_score: Optional[float] = None,
        eval_fid_score: Optional[float] = None,
        promoted_at: Optional[float] = None,
        promoted_by: str = "auto",
        notes: str = "",
        tags: Optional[List[str]] = None,
        parent_version: Optional[str] = None,
        training_steps: int = 0,
    ):
        self.version              = version
        self.model_type           = model_type
        self.checkpoint_path      = checkpoint_path
        self.dataset_version      = dataset_version
        self.created_at           = created_at or time.time()
        self.status               = status
        self.eval_run_id          = eval_run_id
        self.eval_clip_score      = eval_clip_score
        self.eval_aesthetic_score = eval_aesthetic_score
        self.eval_fid_score       = eval_fid_score
        self.promoted_at          = promoted_at
        self.promoted_by          = promoted_by
        self.notes                = notes
        self.tags                 = tags or []
        self.parent_version       = parent_version
        self.training_steps       = training_steps

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "ModelEntry":
        return cls(**{k: d[k] for k in d if hasattr(cls.__init__, "__code__")
                      and k in cls.__init__.__code__.co_varnames})

    def eval_summary(self) -> str:
        parts = []
        if self.eval_clip_score is not None:
            parts.append(f"CLIP={self.eval_clip_score:.3f}")
        if self.eval_aesthetic_score is not None:
            parts.append(f"AES={self.eval_aesthetic_score:.3f}")
        if self.eval_fid_score is not None:
            parts.append(f"FID={self.eval_fid_score:.2f}")
        return " ".join(parts) if parts else "no eval"


class ModelRegistry:
    """
    Central registry for all Genesis model versions.
    Thread-safe for single-process use (file-level locking not implemented).
    """

    def __init__(self, registry_root: str = "models"):
        self.root = Path(registry_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._caches: Dict[str, Dict[str, ModelEntry]] = {}

    # ── IO ────────────────────────────────────────────────────

    def _registry_path(self, model_type: str) -> Path:
        d = self.root / model_type
        d.mkdir(parents=True, exist_ok=True)
        return d / "registry.json"

    def _load(self, model_type: str) -> Dict[str, ModelEntry]:
        if model_type in self._caches:
            return self._caches[model_type]
        p = self._registry_path(model_type)
        entries: Dict[str, ModelEntry] = {}
        if p.exists():
            try:
                data = json.loads(p.read_text())
                for v, d in data.items():
                    entries[v] = ModelEntry.from_dict(d)
            except Exception as e:
                logger.warning(f"Registry load failed [{model_type}]: {e}")
        self._caches[model_type] = entries
        return entries

    def _save(self, model_type: str, entries: Dict[str, ModelEntry]) -> None:
        p = self._registry_path(model_type)
        data = {v: e.to_dict() for v, e in entries.items()}
        p.write_text(json.dumps(data, indent=2))
        self._caches[model_type] = entries

    # ── Registration ──────────────────────────────────────────

    def register(
        self,
        model_type: str,
        version: str,
        checkpoint_path: str = "",
        dataset_version: str = "unknown",
        notes: str = "",
        tags: Optional[List[str]] = None,
        parent_version: Optional[str] = None,
        training_steps: int = 0,
    ) -> ModelEntry:
        """
        Register a new model version as 'staged'.
        Staged models are not in production until explicitly promoted.
        """
        entries = self._load(model_type)
        if version in entries:
            logger.warning(f"Version {version} already registered for {model_type}")
            return entries[version]

        entry = ModelEntry(
            version=version, model_type=model_type,
            checkpoint_path=checkpoint_path,
            dataset_version=dataset_version,
            status="staged",
            notes=notes, tags=tags, parent_version=parent_version,
            training_steps=training_steps,
        )
        entries[version] = entry
        self._save(model_type, entries)
        logger.info(f"Registered {model_type} v{version} (staged)")
        return entry

    # ── Eval ──────────────────────────────────────────────────

    def update_eval(
        self,
        model_type: str = "diffusion",
        version: Optional[str] = None,
        eval_run_id: Optional[str] = None,
        clip_score: Optional[float] = None,
        aesthetic_score: Optional[float] = None,
        fid_score: Optional[float] = None,
    ) -> None:
        """Link evaluation results to a registered model version."""
        entries = self._load(model_type)
        if version is None:
            # Update latest staged/production entry
            for v in sorted(entries.keys(), reverse=True):
                if entries[v].status in ("staged", "production"):
                    version = v
                    break
        if version is None or version not in entries:
            logger.warning(f"Cannot update eval: {model_type} v{version} not found")
            return
        e = entries[version]
        e.eval_run_id          = eval_run_id
        e.eval_clip_score      = clip_score
        e.eval_aesthetic_score = aesthetic_score
        e.eval_fid_score       = fid_score
        self._save(model_type, entries)
        logger.info(f"Updated eval for {model_type} v{version}: {e.eval_summary()}")

    # ── Promotion ─────────────────────────────────────────────

    def promote(
        self,
        model_type: str,
        version: str,
        promoted_by: str = "auto",
        require_eval: bool = True,
    ) -> ModelEntry:
        """
        Promote a staged model to production.

        The previously production model is archived.
        If require_eval=True (default), the model must have evaluation results.
        """
        entries = self._load(model_type)
        if version not in entries:
            raise ValueError(f"{model_type} v{version} not registered")

        entry = entries[version]
        if require_eval and entry.eval_run_id is None:
            raise RuntimeError(
                f"Cannot promote {model_type} v{version}: no evaluation results. "
                f"Run benchmark first or pass require_eval=False."
            )

        # Archive current production model
        for v, e in entries.items():
            if e.status == "production":
                e.status = "archived"
                logger.info(f"Archived {model_type} v{v}")

        # Promote new version
        entry.status      = "production"
        entry.promoted_at = time.time()
        entry.promoted_by = promoted_by
        self._save(model_type, entries)
        logger.info(
            f"PROMOTED {model_type} v{version} to production | "
            f"{entry.eval_summary()}"
        )
        return entry

    def demote(self, model_type: str, version: str) -> None:
        """Demote production model back to staged (emergency rollback)."""
        entries = self._load(model_type)
        if version not in entries:
            raise ValueError(f"{model_type} v{version} not registered")
        entries[version].status = "staged"
        self._save(model_type, entries)
        logger.warning(f"Demoted {model_type} v{version} from production")

    def mark_failed(self, model_type: str, version: str, reason: str = "") -> None:
        entries = self._load(model_type)
        if version in entries:
            entries[version].status = "failed"
            entries[version].notes += f" [FAILED: {reason}]"
            self._save(model_type, entries)

    # ── Query ─────────────────────────────────────────────────

    def get(self, model_type: str, version: str) -> Optional[ModelEntry]:
        return self._load(model_type).get(version)

    def get_production(self, model_type: str) -> Optional[ModelEntry]:
        for e in self._load(model_type).values():
            if e.status == "production":
                return e
        return None

    def list_versions(self, model_type: str) -> List[ModelEntry]:
        return sorted(self._load(model_type).values(),
                       key=lambda e: e.created_at)

    def all_model_types(self) -> List[str]:
        return [d.name for d in self.root.iterdir() if d.is_dir()]

    def stats(self) -> dict:
        result = {}
        for mt in self.all_model_types():
            entries = self._load(mt)
            prod = self.get_production(mt)
            result[mt] = {
                "n_versions": len(entries),
                "production": prod.version if prod else None,
                "production_eval": prod.eval_summary() if prod else None,
            }
        return result

    def print_registry(self, model_type: Optional[str] = None) -> None:
        types = [model_type] if model_type else self.all_model_types()
        W = 70
        for mt in types:
            entries = self.list_versions(mt)
            print(f"\n{'='*W}")
            print(f"  {mt.upper()} Registry")
            print(f"{'='*W}")
            for e in entries:
                prod_flag = " ← PRODUCTION" if e.status == "production" else ""
                print(f"  v{e.version:<15} [{e.status:<10}] "
                      f"ds={e.dataset_version:<6} "
                      f"{e.eval_summary()}{prod_flag}")
        print()
