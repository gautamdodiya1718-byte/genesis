"""
dataset/lifecycle/pruning.py
------------------------------
Dataset pruning — removes duplicate and low-quality images.

Pruning strategies:
  dedup_phash     Remove near-identical images via perceptual hash
  dedup_embedding Remove semantic duplicates via CLIP embedding similarity
  quality_floor   Remove images with quality_score below threshold
  caption_quality Remove images with empty/garbage captions
  resolution      Remove images below minimum resolution
  combined        Run all strategies in sequence (recommended)

Pruning is non-destructive by default (dry_run=True).
Files are moved to a quarantine directory before deletion
so they can be recovered within a retention window.

Usage:
    pruner = DatasetPruner("outputs/dataset", "outputs/pruned_quarantine")
    report = pruner.prune(strategy="combined", dry_run=False)
    print(report.summary())
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from PIL import Image

logger = logging.getLogger(__name__)


# ── Pruning report ────────────────────────────────────────────

@dataclass
class PruneResult:
    strategy: str
    total_scanned: int = 0
    removed: int = 0
    quarantined: int = 0
    bytes_freed: int = 0
    dry_run: bool = True
    duration_s: float = 0.0
    removed_paths: List[str] = field(default_factory=list)
    reasons: Dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        mode = "DRY RUN" if self.dry_run else "LIVE"
        return (
            f"[{mode}] Prune({self.strategy}): "
            f"scanned={self.total_scanned} removed={self.removed} "
            f"freed={self.bytes_freed/1024**2:.1f}MB in {self.duration_s:.1f}s"
        )

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy, "total_scanned": self.total_scanned,
            "removed": self.removed, "quarantined": self.quarantined,
            "bytes_freed_mb": round(self.bytes_freed/1024**2, 2),
            "dry_run": self.dry_run, "duration_s": round(self.duration_s, 1),
            "reasons": self.reasons,
        }


def _phash(img: Image.Image, hash_size: int = 16) -> str:
    """DCT-based perceptual hash."""
    import struct
    try:
        small = img.convert("L").resize((hash_size * 4, hash_size * 4), Image.LANCZOS)
        import numpy as np
        pixels = np.array(small, dtype=float)
        dct_row = np.zeros_like(pixels)
        for i in range(len(pixels)):
            dct_row[i] = np.fft.rfft(pixels[i])[:hash_size].real
        dct_col = np.zeros((hash_size, hash_size))
        for i in range(hash_size):
            dct_col[:, i] = np.fft.rfft(dct_row[:, i])[:hash_size].real
        med = np.median(dct_col)
        bits = (dct_col > med).flatten()
        return "".join("1" if b else "0" for b in bits)
    except Exception:
        return ""


def _hamming(h1: str, h2: str) -> float:
    if not h1 or not h2 or len(h1) != len(h2):
        return 1.0
    return sum(c1 != c2 for c1, c2 in zip(h1, h2)) / len(h1)


class DatasetPruner:
    """
    Removes duplicate and low-quality images from a Genesis dataset.

    Operates on the dataset index.jsonl and images/ directory.
    Moves removed files to a quarantine directory instead of immediate deletion.
    """

    def __init__(
        self,
        dataset_root: str,
        quarantine_dir: str = "outputs/quarantine",
        phash_threshold: float = 0.05,        # Hamming distance — 0.05 = 95% similar
        quality_threshold: float = 0.3,
        min_resolution: int = 192,
        min_caption_words: int = 2,
    ):
        self.dataset_root     = Path(dataset_root)
        self.quarantine_dir   = Path(quarantine_dir)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        self.phash_threshold  = phash_threshold
        self.quality_threshold= quality_threshold
        self.min_resolution   = min_resolution
        self.min_caption_words= min_caption_words
        self._index_path      = self.dataset_root / "index.jsonl"

    def _load_index(self) -> List[dict]:
        if not self._index_path.exists():
            return []
        records = []
        with open(self._index_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records

    def _write_index(self, records: List[dict]) -> None:
        with open(self._index_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    def _quarantine(self, path: str, reason: str, dry_run: bool) -> int:
        """Move file to quarantine. Returns bytes freed."""
        p = Path(path)
        if not p.exists():
            return 0
        size = p.stat().st_size
        if not dry_run:
            dest = self.quarantine_dir / reason / p.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(dest))
        return size

    # ── Strategies ────────────────────────────────────────────

    def prune_duplicates_phash(
        self, records: List[dict], dry_run: bool = True
    ) -> Tuple[List[dict], PruneResult]:
        """Remove near-duplicate images via perceptual hash."""
        result = PruneResult("dedup_phash", total_scanned=len(records), dry_run=dry_run)
        t0 = time.time()

        hashes: Dict[str, str] = {}  # path → phash
        to_remove: Set[str] = set()

        for rec in records:
            path = rec.get("path", rec.get("file_path", ""))
            if not path or path in to_remove:
                continue
            try:
                img = Image.open(path)
                ph  = _phash(img)
                img.close()
            except Exception:
                continue

            # Compare against all stored hashes
            is_dup = False
            for stored_path, stored_hash in hashes.items():
                if _hamming(ph, stored_hash) < self.phash_threshold:
                    to_remove.add(path)
                    result.reasons["phash_dup"] = result.reasons.get("phash_dup", 0) + 1
                    is_dup = True
                    break
            if not is_dup:
                hashes[path] = ph

        surviving = [r for r in records
                     if r.get("path", r.get("file_path", "")) not in to_remove]

        for path in to_remove:
            bytes_freed = self._quarantine(path, "phash_dup", dry_run)
            result.bytes_freed += bytes_freed
            result.removed     += 1
            result.removed_paths.append(path)

        result.duration_s = time.time() - t0
        logger.info(result.summary())
        return surviving, result

    def prune_quality_floor(
        self, records: List[dict], dry_run: bool = True
    ) -> Tuple[List[dict], PruneResult]:
        """Remove images with quality_score below threshold."""
        result = PruneResult("quality_floor", total_scanned=len(records), dry_run=dry_run)
        t0 = time.time()

        surviving = []
        for rec in records:
            qs = float(rec.get("quality_score", 1.0))
            if qs < self.quality_threshold:
                path = rec.get("path", rec.get("file_path", ""))
                result.bytes_freed += self._quarantine(path, "low_quality", dry_run)
                result.removed     += 1
                result.removed_paths.append(path)
                result.reasons["low_quality"] = result.reasons.get("low_quality", 0) + 1
            else:
                surviving.append(rec)

        result.duration_s = time.time() - t0
        logger.info(result.summary())
        return surviving, result

    def prune_caption_quality(
        self, records: List[dict], dry_run: bool = True
    ) -> Tuple[List[dict], PruneResult]:
        """Remove images with empty, too-short, or garbage captions."""
        _GARBAGE = {"untitled", "image", "photo", "picture", "img", "dsc_", "imgp"}
        result = PruneResult("caption_quality", total_scanned=len(records), dry_run=dry_run)
        t0 = time.time()

        surviving = []
        for rec in records:
            cap = (rec.get("caption") or "").strip()
            words = cap.split()

            reject = False
            if len(words) < self.min_caption_words:
                reject = True
                reason = "caption_too_short"
            elif any(g in cap.lower() for g in _GARBAGE):
                reject = True
                reason = "caption_garbage"
            else:
                reason = ""

            if reject:
                path = rec.get("path", rec.get("file_path", ""))
                result.bytes_freed += self._quarantine(path, reason, dry_run)
                result.removed     += 1
                result.removed_paths.append(path)
                result.reasons[reason] = result.reasons.get(reason, 0) + 1
            else:
                surviving.append(rec)

        result.duration_s = time.time() - t0
        logger.info(result.summary())
        return surviving, result

    def prune_low_resolution(
        self, records: List[dict], dry_run: bool = True
    ) -> Tuple[List[dict], PruneResult]:
        """Remove images below minimum resolution."""
        result = PruneResult("resolution", total_scanned=len(records), dry_run=dry_run)
        t0 = time.time()
        surviving = []

        for rec in records:
            # Check stored dimensions first
            w = rec.get("width", 0)
            h = rec.get("height", 0)
            if w == 0 or h == 0:
                path = rec.get("path", rec.get("file_path", ""))
                try:
                    with Image.open(path) as img:
                        w, h = img.size
                except Exception:
                    surviving.append(rec)
                    continue

            if min(w, h) < self.min_resolution:
                path = rec.get("path", rec.get("file_path", ""))
                result.bytes_freed += self._quarantine(path, "low_resolution", dry_run)
                result.removed     += 1
                result.removed_paths.append(path)
                result.reasons["low_resolution"] = result.reasons.get("low_resolution", 0) + 1
            else:
                surviving.append(rec)

        result.duration_s = time.time() - t0
        logger.info(result.summary())
        return surviving, result

    # ── Combined prune ────────────────────────────────────────

    def prune(
        self,
        strategy: str = "combined",
        dry_run: bool = True,
        save_index: bool = True,
    ) -> PruneResult:
        """
        Run pruning strategy on the dataset.

        Args:
            strategy:   "combined" | "dedup_phash" | "quality_floor" |
                        "caption_quality" | "resolution"
            dry_run:    If True, report what would be removed but don't move files
            save_index: If not dry_run, rewrite index.jsonl with survivors

        Returns:
            Combined PruneResult
        """
        records = self._load_index()
        if not records:
            logger.warning(f"No records found in {self._index_path}")
            return PruneResult(strategy, dry_run=dry_run)

        combined = PruneResult(strategy, total_scanned=len(records), dry_run=dry_run)
        t0 = time.time()

        strategies = {
            "dedup_phash":    [self.prune_duplicates_phash],
            "quality_floor":  [self.prune_quality_floor],
            "caption_quality":[self.prune_caption_quality],
            "resolution":     [self.prune_low_resolution],
            "combined": [
                self.prune_low_resolution,
                self.prune_caption_quality,
                self.prune_quality_floor,
                self.prune_duplicates_phash,
            ],
        }

        funcs = strategies.get(strategy, strategies["combined"])
        for fn in funcs:
            records, sub_result = fn(records, dry_run=dry_run)
            combined.removed       += sub_result.removed
            combined.bytes_freed   += sub_result.bytes_freed
            combined.removed_paths += sub_result.removed_paths
            for k, v in sub_result.reasons.items():
                combined.reasons[k] = combined.reasons.get(k, 0) + v

        if not dry_run and save_index:
            self._write_index(records)
            logger.info(f"Index updated: {len(records)} records remaining")

        combined.duration_s = time.time() - t0
        logger.info(f"Pruning complete: {combined.summary()}")

        # Save report
        rp = self.quarantine_dir / f"prune_report_{int(t0)}.json"
        rp.write_text(json.dumps(combined.to_dict(), indent=2))
        return combined

    def restore_from_quarantine(
        self, quarantine_subdir: str, dataset_images_dir: str
    ) -> int:
        """Restore quarantined files back to dataset images directory."""
        qdir = self.quarantine_dir / quarantine_subdir
        if not qdir.exists():
            return 0
        dest  = Path(dataset_images_dir)
        dest.mkdir(parents=True, exist_ok=True)
        count = 0
        for f in qdir.iterdir():
            if f.is_file():
                shutil.move(str(f), str(dest / f.name))
                count += 1
        logger.info(f"Restored {count} files from quarantine/{quarantine_subdir}")
        return count
