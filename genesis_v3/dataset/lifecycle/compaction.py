"""
dataset/lifecycle/compaction.py
---------------------------------
Dataset compaction — defragments and re-organises the dataset for efficiency.

After many prune/retention cycles the dataset can become:
  - Fragmented: images spread across many subdirectories
  - Redundant: index.jsonl referencing deleted paths
  - Unbalanced: categories with very uneven sample counts
  - Poorly ordered: no locality for batch loading

Compaction operations:
  reindex       Rebuild index.jsonl from actual files on disk
  reorganize    Move images to a flat, content-addressed layout
  balance       Resample categories to target balance ratios
  vacuum        Remove dead entries (paths that no longer exist)
  export_hf     Export compacted dataset in HuggingFace datasets format

Compacted layout:
  outputs/dataset_compact/
    images/
      000/00000.jpg ... (content-addressed, 1000 per dir)
    index.jsonl       (canonical, all paths valid)
    stats.json        (category counts, quality distribution)
    schema.json       (field documentation)

Usage:
    compactor = DatasetCompactor("outputs/dataset", "outputs/dataset_compact")
    report = compactor.compact(operations=["vacuum", "reorganize", "reindex"])
    print(report.summary())
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CompactionReport:
    operations_run: List[str] = field(default_factory=list)
    records_before: int = 0
    records_after: int = 0
    dead_entries_removed: int = 0
    files_moved: int = 0
    bytes_before: int = 0
    bytes_after: int = 0
    duration_s: float = 0.0
    output_dir: str = ""

    def summary(self) -> str:
        return (
            f"Compact({', '.join(self.operations_run)}): "
            f"records {self.records_before}→{self.records_after} "
            f"dead_removed={self.dead_entries_removed} "
            f"moved={self.files_moved} "
            f"{self.duration_s:.1f}s → {self.output_dir}"
        )

    def to_dict(self) -> dict:
        return {
            "operations_run": self.operations_run,
            "records_before": self.records_before,
            "records_after": self.records_after,
            "dead_entries_removed": self.dead_entries_removed,
            "files_moved": self.files_moved,
            "size_before_mb": round(self.bytes_before / 1024**2, 1),
            "size_after_mb": round(self.bytes_after / 1024**2, 1),
            "duration_s": round(self.duration_s, 1),
            "output_dir": self.output_dir,
        }


_SCHEMA = {
    "version": "1.0",
    "fields": {
        "path":          "Absolute or relative path to image file",
        "caption":       "Text caption for the image",
        "source":        "Data source (crawler domain, active_learning, etc.)",
        "category":      "Image category label",
        "quality_score": "Float [0,1] — higher is better quality",
        "timestamp":     "Unix timestamp of when record was added",
        "width":         "Image width in pixels",
        "height":        "Image height in pixels",
        "hash":          "SHA-256 of image file",
        "tags":          "List of string tags",
    },
}


class DatasetCompactor:
    """
    Compacts a Genesis dataset into a clean, efficient layout.

    Can be run standalone or as part of a scheduled maintenance cycle.
    Source dataset is never modified — output written to output_dir.
    """

    def __init__(
        self,
        source_dir: str,
        output_dir: str,
        images_per_subdir: int = 1000,
        target_category_balance: Optional[Dict[str, float]] = None,
    ):
        self.source_dir  = Path(source_dir)
        self.output_dir  = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.imgs_per_subdir = images_per_subdir
        self.target_balance  = target_category_balance  # {cat: fraction}

        self._src_index  = self.source_dir / "index.jsonl"
        self._dst_index  = self.output_dir / "index.jsonl"
        self._dst_images = self.output_dir / "images"

    # ── Load / Save ───────────────────────────────────────────

    def _load_index(self, path: Path) -> List[dict]:
        if not path.exists():
            return []
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records

    def _write_index(self, path: Path, records: List[dict]) -> None:
        with open(path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    def _rec_path(self, rec: dict) -> Optional[Path]:
        p = rec.get("path") or rec.get("file_path") or rec.get("image_path")
        return Path(p) if p else None

    # ── Operations ────────────────────────────────────────────

    def vacuum(self, records: List[dict]) -> Tuple[List[dict], int]:
        """
        Remove index entries pointing to non-existent files.
        Returns (surviving_records, n_removed).
        """
        surviving = []
        removed = 0
        for rec in records:
            p = self._rec_path(rec)
            if p is None or not p.exists():
                removed += 1
            else:
                surviving.append(rec)
        logger.info(f"Vacuum: removed {removed} dead entries from {len(records)} records")
        return surviving, removed

    def reorganize(
        self, records: List[dict], report: CompactionReport
    ) -> List[dict]:
        """
        Move image files to content-addressed layout in output_dir.
        Returns updated records with new paths.

        Layout: output_dir/images/<3-digit-bucket>/<index>.<ext>
        Buckets of 1000 images each for filesystem efficiency.
        """
        self._dst_images.mkdir(parents=True, exist_ok=True)
        updated = []
        counter = 0

        for rec in records:
            src_path = self._rec_path(rec)
            if src_path is None or not src_path.exists():
                continue

            bucket = str(counter // self.imgs_per_subdir).zfill(3)
            (self._dst_images / bucket).mkdir(exist_ok=True)

            ext = src_path.suffix or ".jpg"
            dst_name = f"{counter:08d}{ext}"
            dst_path = self._dst_images / bucket / dst_name

            try:
                shutil.copy2(str(src_path), str(dst_path))
                new_rec = {**rec, "path": str(dst_path)}
                updated.append(new_rec)
                report.files_moved += 1
                report.bytes_after += dst_path.stat().st_size
            except Exception as e:
                logger.warning(f"Failed to copy {src_path}: {e}")
                updated.append(rec)  # keep old path

            counter += 1

        logger.info(f"Reorganized {counter} images into content-addressed layout")
        return updated

    def reindex(self, images_dir: Optional[str] = None) -> List[dict]:
        """
        Rebuild index.jsonl from actual image files on disk.
        Used when index.jsonl is corrupted or missing.

        Reads any existing captions from paired .txt or .json sidecar files.
        """
        scan_dir = Path(images_dir) if images_dir else self._dst_images
        if not scan_dir.exists():
            scan_dir = self.source_dir / "images"

        records = []
        img_extensions = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

        for img_path in sorted(scan_dir.rglob("*")):
            if img_path.suffix.lower() not in img_extensions:
                continue

            rec: dict = {
                "path":      str(img_path),
                "caption":   "",
                "source":    "reindexed",
                "timestamp": img_path.stat().st_mtime,
            }

            # Try sidecar files for caption
            for sidecar_ext in (".txt", ".json", ".caption"):
                sidecar = img_path.with_suffix(sidecar_ext)
                if sidecar.exists():
                    try:
                        content = sidecar.read_text().strip()
                        if sidecar_ext == ".json":
                            data = json.loads(content)
                            rec["caption"] = data.get("caption", "")
                            rec.update({k: v for k, v in data.items()
                                        if k not in ("path",)})
                        else:
                            rec["caption"] = content
                    except Exception:
                        pass
                    break

            # Get dimensions
            try:
                from PIL import Image
                with Image.open(str(img_path)) as img:
                    rec["width"], rec["height"] = img.size
            except Exception:
                pass

            records.append(rec)

        logger.info(f"Reindexed {len(records)} images from {scan_dir}")
        return records

    def balance(
        self,
        records: List[dict],
        target_per_category: Optional[int] = None,
    ) -> List[dict]:
        """
        Resample records to target category balance.

        If target_per_category is None, use target_balance ratios.
        Categories with fewer samples than target are kept fully.
        Over-represented categories are downsampled randomly.
        """
        import random

        by_cat: Dict[str, List[dict]] = {}
        uncategorized = []
        for rec in records:
            cat = rec.get("category")
            if cat:
                by_cat.setdefault(cat, []).append(rec)
            else:
                uncategorized.append(rec)

        if target_per_category is None:
            if self.target_balance:
                total = len(records)
                target_per_category_map = {
                    cat: int(total * frac)
                    for cat, frac in self.target_balance.items()
                }
            else:
                # Equal balance: use median category size
                sizes = sorted(len(v) for v in by_cat.values())
                median = sizes[len(sizes) // 2] if sizes else 0
                target_per_category = median
                target_per_category_map = {cat: target_per_category for cat in by_cat}
        else:
            target_per_category_map = {cat: target_per_category for cat in by_cat}

        balanced = list(uncategorized)
        for cat, recs in by_cat.items():
            target = target_per_category_map.get(cat, target_per_category or len(recs))
            if len(recs) > target:
                # Prefer high quality when downsampling
                recs_sorted = sorted(recs,
                                      key=lambda r: float(r.get("quality_score", 0.5)),
                                      reverse=True)
                selected = recs_sorted[:target]
            else:
                selected = recs
            balanced.extend(selected)

        logger.info(
            f"Balance: {len(records)} → {len(balanced)} records "
            f"({len(by_cat)} categories)"
        )
        return balanced

    def add_hashes(self, records: List[dict]) -> List[dict]:
        """Add SHA-256 hash to records that don't have one."""
        updated = []
        for rec in records:
            if rec.get("hash"):
                updated.append(rec)
                continue
            p = self._rec_path(rec)
            if p and p.exists():
                h = hashlib.sha256()
                with open(p, "rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        h.update(chunk)
                rec = {**rec, "hash": h.hexdigest()}
            updated.append(rec)
        return updated

    # ── Stats ─────────────────────────────────────────────────

    def compute_stats(self, records: List[dict]) -> dict:
        by_source:   Dict[str, int] = {}
        by_category: Dict[str, int] = {}
        quality_sum = 0.0
        quality_n   = 0
        total_size  = 0

        for rec in records:
            by_source[rec.get("source", "unknown")]   = by_source.get(rec.get("source", "unknown"), 0) + 1
            by_category[rec.get("category", "unknown")] = by_category.get(rec.get("category", "unknown"), 0) + 1
            if "quality_score" in rec:
                quality_sum += float(rec["quality_score"])
                quality_n   += 1
            p = self._rec_path(rec)
            if p and p.exists():
                total_size += p.stat().st_size

        return {
            "n_records": len(records),
            "total_size_mb": round(total_size / 1024**2, 1),
            "by_source": by_source,
            "by_category": by_category,
            "mean_quality": round(quality_sum / max(quality_n, 1), 3),
        }

    # ── Main entrypoint ───────────────────────────────────────

    def compact(
        self,
        operations: Optional[List[str]] = None,
        dry_run: bool = False,
        balance_target: Optional[int] = None,
    ) -> CompactionReport:
        """
        Run compaction pipeline.

        Args:
            operations: Ordered list of operations to run.
                        Default: ["vacuum", "reorganize", "reindex"]
                        Available: "vacuum" | "reorganize" | "reindex" |
                                   "balance" | "add_hashes"
            dry_run:    If True, compute stats but don't write output
            balance_target: Target samples per category for "balance" op

        Returns:
            CompactionReport
        """
        if operations is None:
            operations = ["vacuum", "reorganize"]

        report = CompactionReport(
            operations_run=operations,
            output_dir=str(self.output_dir),
        )
        t0 = time.time()

        records = self._load_index(self._src_index)
        report.records_before = len(records)

        for op in operations:
            if op == "vacuum":
                records, dead = self.vacuum(records)
                report.dead_entries_removed += dead

            elif op == "reorganize":
                if not dry_run:
                    records = self.reorganize(records, report)

            elif op == "reindex":
                records = self.reindex()

            elif op == "balance":
                records = self.balance(records, balance_target)

            elif op == "add_hashes":
                if not dry_run:
                    records = self.add_hashes(records)

            else:
                logger.warning(f"Unknown compaction operation: {op}")

        report.records_after = len(records)

        if not dry_run:
            self._write_index(self._dst_index, records)
            (self.output_dir / "stats.json").write_text(
                json.dumps(self.compute_stats(records), indent=2)
            )
            (self.output_dir / "schema.json").write_text(
                json.dumps(_SCHEMA, indent=2)
            )

        report.duration_s = time.time() - t0
        logger.info(f"Compaction complete: {report.summary()}")

        rp = self.output_dir / f"compact_report_{int(t0)}.json"
        if not dry_run:
            rp.write_text(json.dumps(report.to_dict(), indent=2))

        return report

    def export_hf_dataset(
        self,
        records: Optional[List[dict]] = None,
        split: str = "train",
    ) -> str:
        """
        Export compacted dataset in HuggingFace datasets format.
        Returns path to saved dataset directory.
        """
        try:
            from datasets import Dataset, Image as HFImage
        except ImportError:
            raise RuntimeError("datasets not installed: pip install datasets")

        if records is None:
            records = self._load_index(self._dst_index)

        valid = []
        for rec in records:
            p = self._rec_path(rec)
            if p and p.exists() and rec.get("caption"):
                valid.append({"image": str(p), "caption": rec["caption"],
                               "source": rec.get("source", "")})

        ds = Dataset.from_list(valid).cast_column("image", HFImage())
        out = str(self.output_dir / "hf_dataset")
        ds.save_to_disk(out)
        logger.info(f"Exported {len(valid)} samples as HF dataset → {out}")
        return out
