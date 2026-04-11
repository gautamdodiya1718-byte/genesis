"""
training/dataset_versioning.py
--------------------------------
Immutable dataset snapshots for reproducible training.

Training MUST always use a fixed, versioned snapshot — never live data.
This prevents dataset drift from corrupting training runs mid-execution.

Snapshot layout:
  <versioned_root>/
    v1/
      images/          — symlinks or copies of images
      index.jsonl      — frozen training manifest
      manifest.json    — snapshot metadata
    v2/
      ...
    latest -> v2/      — symlink to current

Manifest JSON:
  {
    "version": "v2",
    "created_at": 1234567890,
    "n_samples": 12450,
    "source_dataset": "outputs/dataset",
    "filters": {...},
    "hash": "<sha256 of index.jsonl>",
    "parent_version": "v1"
  }
"""
from __future__ import annotations
import hashlib
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _jsonl_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


class DatasetSnapshot:
    """Represents one immutable dataset version."""

    def __init__(self, version_dir: Path):
        self.version_dir = version_dir
        self._manifest: Optional[dict] = None

    @property
    def manifest(self) -> dict:
        if self._manifest is None:
            mp = self.version_dir / "manifest.json"
            if mp.exists():
                self._manifest = json.loads(mp.read_text())
            else:
                self._manifest = {}
        return self._manifest

    @property
    def version(self) -> str:
        return self.manifest.get("version", self.version_dir.name)

    @property
    def n_samples(self) -> int:
        return self.manifest.get("n_samples", 0)

    @property
    def index_path(self) -> Path:
        return self.version_dir / "index.jsonl"

    @property
    def images_dir(self) -> Path:
        return self.version_dir / "images"

    def exists(self) -> bool:
        return self.index_path.exists()

    def load_index(self) -> List[dict]:
        if not self.index_path.exists():
            return []
        records = []
        with open(self.index_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records

    def to_dict(self) -> dict:
        return {**self.manifest,
                "version_dir": str(self.version_dir),
                "exists": self.exists()}


class DatasetVersioning:
    """
    Manages versioned dataset snapshots.

    Snapshots are created from a live DatasetBuilder root.
    Training always reads from a snapshot, never from live data.
    """

    def __init__(self, versioned_root: str):
        self.root = Path(versioned_root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ── Version enumeration ───────────────────────────────────

    def list_versions(self) -> List[str]:
        """Return all version names sorted ascending."""
        versions = [
            d.name for d in self.root.iterdir()
            if d.is_dir() and (d / "manifest.json").exists()
        ]
        return sorted(versions)

    def get_version(self, version: str) -> Optional[DatasetSnapshot]:
        vdir = self.root / version
        if not vdir.exists():
            return None
        return DatasetSnapshot(vdir)

    def get_latest(self) -> Optional[DatasetSnapshot]:
        versions = self.list_versions()
        return self.get_version(versions[-1]) if versions else None

    def _next_version_name(self) -> str:
        existing = self.list_versions()
        n = len(existing) + 1
        return f"v{n}"

    # ── Snapshot creation ─────────────────────────────────────

    def create_snapshot(
        self,
        source_dataset_root: str,
        version_name: Optional[str] = None,
        filter_min_quality: float = 0.0,
        filter_domain: Optional[str] = None,
        max_samples: Optional[int] = None,
        copy_images: bool = False,
        description: str = "",
    ) -> DatasetSnapshot:
        """
        Create a new immutable dataset snapshot from a live dataset.

        Args:
            source_dataset_root:  DatasetBuilder output root
                                  (contains images/ + index.jsonl)
            version_name:         e.g. "v3". Auto-increments if None.
            filter_min_quality:   Only include items with quality >= this
            filter_domain:        Only include items from this domain
            max_samples:          Cap snapshot size
            copy_images:          Copy image files (True) or symlink (False)
            description:          Human label for this snapshot
        """
        version_name = version_name or self._next_version_name()
        version_dir  = self.root / version_name

        if version_dir.exists():
            raise FileExistsError(
                f"Version {version_name} already exists at {version_dir}. "
                f"Use a different name or delete it first."
            )

        version_dir.mkdir(parents=True)
        logger.info(f"Creating snapshot {version_name} from {source_dataset_root}")

        source_root  = Path(source_dataset_root)
        source_index = source_root / "index.jsonl"
        source_imgs  = source_root / "images"

        if not source_index.exists():
            raise FileNotFoundError(f"No index.jsonl found at {source_index}")

        # ── Load + filter source records ───────────────────────
        records = []
        with open(source_index) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Apply filters
                if filter_min_quality > 0:
                    if rec.get("quality_score", 1.0) < filter_min_quality:
                        continue
                if filter_domain and rec.get("domain") != filter_domain:
                    continue

                records.append(rec)

        if max_samples and len(records) > max_samples:
            # Prefer higher quality items
            records.sort(key=lambda r: r.get("quality_score", 0.0), reverse=True)
            records = records[:max_samples]

        logger.info(f"  {len(records)} records selected")

        # ── Write frozen index.jsonl ───────────────────────────
        dest_index = version_dir / "index.jsonl"
        with open(dest_index, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        # ── Images: copy or symlink ────────────────────────────
        dest_imgs = version_dir / "images"
        if copy_images and source_imgs.exists():
            logger.info("  Copying images...")
            shutil.copytree(str(source_imgs), str(dest_imgs))
        elif source_imgs.exists():
            # Relative symlink
            try:
                os.symlink(str(source_imgs.resolve()), str(dest_imgs))
            except OSError:
                dest_imgs.mkdir()  # fallback: empty dir (paths in jsonl point to source)

        # ── Write manifest ─────────────────────────────────────
        parent = self.get_latest()
        manifest = {
            "version": version_name,
            "created_at": time.time(),
            "n_samples": len(records),
            "source_dataset": str(source_dataset_root),
            "filters": {
                "min_quality": filter_min_quality,
                "domain": filter_domain,
                "max_samples": max_samples,
            },
            "hash": _jsonl_hash(dest_index),
            "parent_version": parent.version if parent else None,
            "description": description,
            "copy_images": copy_images,
        }
        (version_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )

        # Update "latest" symlink
        latest_link = self.root / "latest"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        try:
            os.symlink(version_name, str(latest_link))
        except OSError:
            pass

        snap = DatasetSnapshot(version_dir)
        logger.info(
            f"Snapshot {version_name} created: "
            f"{len(records)} samples, hash={manifest['hash']}"
        )
        return snap

    def delete_version(self, version: str, confirm: bool = False) -> None:
        """Delete a snapshot. Requires confirm=True."""
        if not confirm:
            raise ValueError("Pass confirm=True to delete a snapshot.")
        latest = self.get_latest()
        if latest and latest.version == version:
            raise RuntimeError(
                f"Cannot delete the latest version ({version}). "
                f"Create a newer snapshot first."
            )
        vdir = self.root / version
        if vdir.exists():
            shutil.rmtree(str(vdir))
            logger.warning(f"Deleted snapshot {version}")

    # ── Comparison / diff ─────────────────────────────────────

    def diff(self, v1: str, v2: str) -> dict:
        """Compare two snapshots by metadata."""
        s1 = self.get_version(v1)
        s2 = self.get_version(v2)
        if s1 is None or s2 is None:
            return {"error": "version not found"}
        return {
            "from": v1,
            "to": v2,
            "n_samples_delta": s2.n_samples - s1.n_samples,
            "hash_changed": s1.manifest.get("hash") != s2.manifest.get("hash"),
            "from_manifest": s1.manifest,
            "to_manifest":   s2.manifest,
        }

    def stats(self) -> dict:
        versions = self.list_versions()
        total = {v: self.get_version(v).n_samples for v in versions}
        return {
            "n_versions": len(versions),
            "versions": versions,
            "samples_per_version": total,
            "versioned_root": str(self.root),
        }

    def print_versions(self) -> None:
        versions = self.list_versions()
        latest = self.get_latest()
        print(f"\n{'='*55}")
        print(f"  Dataset Versions  ({self.root})")
        print(f"{'='*55}")
        for v in versions:
            snap = self.get_version(v)
            tag  = " ← latest" if (latest and latest.version == v) else ""
            desc = snap.manifest.get("description", "")
            print(f"  {v:<6}  {snap.n_samples:>8,} samples  "
                  f"hash={snap.manifest.get('hash','?')[:8]}  {desc}{tag}")
        print(f"{'='*55}\n")
