"""
core/model_versioning.py
--------------------------
Model version tracking with semantic versioning and lineage graph.

Provides:
  - Semantic version management (MAJOR.MINOR.PATCH)
  - Training lineage (parent → child chains)
  - Version comparison and ordering
  - Automated version bumping on training events

Version semantics for Genesis:
  MAJOR — architecture change (new U-Net, new VAE, new text encoder)
  MINOR — new training data version, significant eval improvement
  PATCH — fine-tune, hyperparameter change, minor eval improvement

Lineage is stored as a DAG in versions/lineage.json.
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
class SemanticVersion:
    major: int = 0
    minor: int = 1
    patch: int = 0
    pre:   str = ""   # e.g. "alpha", "beta", "rc1"

    def bump_major(self) -> "SemanticVersion":
        return SemanticVersion(self.major + 1, 0, 0)

    def bump_minor(self) -> "SemanticVersion":
        return SemanticVersion(self.major, self.minor + 1, 0)

    def bump_patch(self) -> "SemanticVersion":
        return SemanticVersion(self.major, self.minor, self.patch + 1)

    def __str__(self) -> str:
        base = f"{self.major}.{self.minor}.{self.patch}"
        return f"{base}-{self.pre}" if self.pre else base

    def __lt__(self, other: "SemanticVersion") -> bool:
        return self.tuple() < other.tuple()

    def tuple(self) -> Tuple[int, int, int]:
        return (self.major, self.minor, self.patch)

    @classmethod
    def parse(cls, s: str) -> "SemanticVersion":
        s = s.lstrip("v")
        pre = ""
        if "-" in s:
            s, pre = s.split("-", 1)
        parts = s.split(".")
        major = int(parts[0]) if len(parts) > 0 else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2]) if len(parts) > 2 else 0
        return cls(major, minor, patch, pre)

    def to_dict(self) -> dict:
        return {"major": self.major, "minor": self.minor,
                "patch": self.patch, "pre": self.pre}

    @classmethod
    def from_dict(cls, d: dict) -> "SemanticVersion":
        return cls(d.get("major", 0), d.get("minor", 1),
                   d.get("patch", 0), d.get("pre", ""))


@dataclass
class VersionNode:
    """Node in the model version lineage graph."""
    version:         str
    model_type:      str
    created_at:      float = field(default_factory=time.time)
    parent_version:  Optional[str]  = None
    dataset_version: str  = "unknown"
    training_event:  str  = "train"   # train | finetune | retrain | eval_only
    eval_clip_score: Optional[float] = None
    eval_fid_score:  Optional[float] = None
    notes:           str  = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "VersionNode":
        return cls(**{k: d.get(k) for k in
                      ("version","model_type","created_at","parent_version",
                       "dataset_version","training_event","eval_clip_score",
                       "eval_fid_score","notes")})


class VersionTracker:
    """
    Tracks model version lineage for a single model type.

    Usage:
        tracker = VersionTracker("models/lineage", "diffusion")
        v = tracker.current_version()           # SemanticVersion
        new_v = tracker.bump("minor", ...)      # "0.2.0"
        tracker.print_lineage()
    """

    def __init__(self, lineage_dir: str, model_type: str):
        self.model_type = model_type
        self.lineage_dir = Path(lineage_dir)
        self.lineage_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.lineage_dir / f"{model_type}_lineage.json"
        self._nodes: Dict[str, VersionNode] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text())
                for v, d in data.items():
                    self._nodes[v] = VersionNode.from_dict(d)
            except Exception as e:
                logger.warning(f"Lineage load failed: {e}")

    def _save(self) -> None:
        self._path.write_text(
            json.dumps({v: n.to_dict() for v, n in self._nodes.items()}, indent=2)
        )

    def current_version(self) -> SemanticVersion:
        if not self._nodes:
            return SemanticVersion(0, 1, 0)
        versions = [SemanticVersion.parse(v) for v in self._nodes]
        return max(versions)

    def bump(
        self,
        level: str = "minor",   # major | minor | patch
        dataset_version: str = "unknown",
        training_event: str = "train",
        parent_override: Optional[str] = None,
        notes: str = "",
    ) -> str:
        """
        Create next version by bumping level.
        Returns version string (e.g. "0.2.0").
        """
        cur = self.current_version()
        if level == "major":
            nxt = cur.bump_major()
        elif level == "minor":
            nxt = cur.bump_minor()
        else:
            nxt = cur.bump_patch()

        version_str = str(nxt)
        parent = parent_override or str(cur) if self._nodes else None
        node = VersionNode(
            version=version_str, model_type=self.model_type,
            parent_version=parent, dataset_version=dataset_version,
            training_event=training_event, notes=notes,
        )
        self._nodes[version_str] = node
        self._save()
        logger.info(f"Version bump: {cur} → {nxt} ({level}) [{self.model_type}]")
        return version_str

    def add_version(
        self,
        version_str: str,
        parent_version: Optional[str] = None,
        dataset_version: str = "unknown",
        training_event: str = "train",
        notes: str = "",
    ) -> VersionNode:
        """Add an explicit version (for migrations / manual registration)."""
        node = VersionNode(
            version=version_str, model_type=self.model_type,
            parent_version=parent_version, dataset_version=dataset_version,
            training_event=training_event, notes=notes,
        )
        self._nodes[version_str] = node
        self._save()
        return node

    def update_eval(self, version_str: str,
                    clip_score: Optional[float] = None,
                    fid_score: Optional[float] = None) -> None:
        if version_str in self._nodes:
            self._nodes[version_str].eval_clip_score = clip_score
            self._nodes[version_str].eval_fid_score  = fid_score
            self._save()

    def lineage(self, version_str: str) -> List[VersionNode]:
        """Return ancestor chain from version_str to root."""
        chain = []
        v = version_str
        visited = set()
        while v and v not in visited:
            node = self._nodes.get(v)
            if node is None:
                break
            chain.append(node)
            visited.add(v)
            v = node.parent_version
        chain.reverse()
        return chain

    def all_versions(self) -> List[VersionNode]:
        return sorted(self._nodes.values(),
                       key=lambda n: SemanticVersion.parse(n.version).tuple())

    def print_lineage(self) -> None:
        nodes = self.all_versions()
        cur = str(self.current_version())
        print(f"\n{'='*55}")
        print(f"  {self.model_type.upper()} Version Lineage")
        print(f"{'='*55}")
        for n in nodes:
            cur_flag = " ← current" if n.version == cur else ""
            eval_str = ""
            if n.eval_clip_score is not None:
                eval_str = f" CLIP={n.eval_clip_score:.3f}"
            if n.eval_fid_score is not None:
                eval_str += f" FID={n.eval_fid_score:.2f}"
            parent = f"← {n.parent_version}" if n.parent_version else "(root)"
            print(f"  v{n.version:<12} {parent:<16} ds={n.dataset_version:<6}"
                  f"{eval_str}{cur_flag}")
        print(f"{'='*55}\n")


class ModelVersioningHub:
    """
    Central hub managing version trackers for all model types.
    Delegates to per-type VersionTrackers.
    """

    def __init__(self, lineage_dir: str = "models/lineage"):
        self.lineage_dir = lineage_dir
        self._trackers: Dict[str, VersionTracker] = {}

    def tracker(self, model_type: str) -> VersionTracker:
        if model_type not in self._trackers:
            self._trackers[model_type] = VersionTracker(self.lineage_dir, model_type)
        return self._trackers[model_type]

    def current_version(self, model_type: str) -> str:
        return str(self.tracker(model_type).current_version())

    def bump(self, model_type: str, level: str = "minor", **kwargs) -> str:
        return self.tracker(model_type).bump(level, **kwargs)

    def update_eval(self, model_type: str, version: str,
                    clip_score: Optional[float], fid_score: Optional[float]) -> None:
        self.tracker(model_type).update_eval(version, clip_score, fid_score)

    def summary(self) -> dict:
        return {mt: str(tr.current_version())
                for mt, tr in self._trackers.items()}
