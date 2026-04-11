"""
dataset/builder.py
-------------------
Unified Dataset Builder for the Genesis system.

Merges AutoDiff builder.py with GPU training dataset support.
Handles images from all sources:
  - Generated (SD / LCM)
  - Crawled (Openverse / Wikimedia / Playwright)
  - User-provided (any directory)

Storage layout:
  outputs/dataset/
    images/           normalized PNG files (000001.png ...)
    captions.json     {filename: caption}
    metadata.json     {filename: {source, url, prompt, phash, ...}}
    index.jsonl       one JSON record per image (training-ready)

Export formats:
  - JSONL  (HuggingFace datasets compatible)
  - CSV    (spreadsheet inspection)
  - txt+img pairs  (DreamBooth / Kohya SS compatible)
"""

from __future__ import annotations
import json, logging, os, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from PIL import Image

from core.image_utils import (
    load_image, save_image, resize_center_crop,
    perceptual_hash, are_duplicates, md5_hash,
)

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """
    Manages a structured image+caption dataset with deduplication.
    Thread-safe for single-process use (no multiprocessing sharing).
    """

    def __init__(self, cfg):
        self.cfg = cfg
        dc = cfg.dataset

        self.root     = Path(dc.root)
        self.img_dir  = self.root / dc.images_dir
        self.img_dir.mkdir(parents=True, exist_ok=True)

        self.captions_path = self.root / dc.captions_file
        self.metadata_path = self.root / dc.metadata_file
        self.index_path    = self.root / dc.index_file

        self.target_size   = tuple(dc.image_size)      # (W, H)
        self.img_format    = dc.image_format.upper()    # "PNG"
        self.dedup_thresh  = dc.dedup_threshold         # 0.95
        self.min_cap_len   = dc.get("min_caption_length", 10)
        self.max_cap_len   = dc.get("max_caption_length", 200)

        # Load existing state
        self.captions: Dict[str, str] = self._load_json(self.captions_path)
        self.metadata: Dict[str, dict] = self._load_json(self.metadata_path)
        self._hashes: Dict[str, str] = {
            fn: meta["phash"]
            for fn, meta in self.metadata.items()
            if "phash" in meta
        }

        logger.info(
            f"DatasetBuilder ready | root={self.root} | "
            f"existing={len(self.captions)} images"
        )

    # ── Add images ─────────────────────────────────────────────

    def add_image(
        self,
        image: Union[str, Path, Image.Image],
        caption: str,
        source: str = "unknown",
        url: Optional[str] = None,
        prompt: Optional[str] = None,
        extra_meta: Optional[dict] = None,
    ) -> Optional[str]:
        """
        Add a single image to the dataset.

        Returns:
            filename (e.g. "000042.png") if added, None if rejected (duplicate/bad caption).
        """
        # Validate caption
        caption = caption.strip()
        if len(caption) < self.min_cap_len:
            logger.debug(f"Caption too short ({len(caption)}): '{caption[:40]}'")
            return None
        if len(caption) > self.max_cap_len:
            caption = caption[:self.max_cap_len].rsplit(" ", 1)[0]

        # Load image
        if isinstance(image, (str, Path)):
            pil = load_image(image)
            if pil is None:
                logger.warning(f"Could not load image: {image}")
                return None
        else:
            pil = image

        # Resize to target size
        pil = resize_center_crop(pil, self.target_size)

        # Compute perceptual hash for deduplication
        phash = perceptual_hash(pil)
        if self._is_duplicate(phash):
            logger.debug("Duplicate image rejected")
            return None

        # Assign sequential filename
        idx = len(self.captions) + 1
        filename = f"{idx:06d}.{self.img_format.lower()}"
        out_path = self.img_dir / filename

        # Save normalized image
        save_image(pil, out_path, fmt=self.img_format)

        # Record
        self.captions[filename] = caption
        self._hashes[filename] = phash
        self.metadata[filename] = {
            "source":    source,
            "url":       url,
            "prompt":    prompt,
            "phash":     phash,
            "md5":       md5_hash(out_path),
            "width":     self.target_size[0],
            "height":    self.target_size[1],
            "timestamp": time.time(),
            **(extra_meta or {}),
        }

        # Append to JSONL index immediately
        self._append_index(filename)

        return filename

    def add_batch(
        self,
        items: List[dict],
        save_every: int = 10,
    ) -> Tuple[int, int]:
        """
        Add multiple images. Each item is a dict with keys:
          image, caption, source, url (optional), prompt (optional)

        Returns:
            (added, rejected) counts
        """
        added = rejected = 0
        for i, item in enumerate(items):
            fn = self.add_image(
                image=item["image"],
                caption=item.get("caption", ""),
                source=item.get("source", "unknown"),
                url=item.get("url"),
                prompt=item.get("prompt"),
                extra_meta=item.get("extra_meta"),
            )
            if fn:
                added += 1
            else:
                rejected += 1

            if (i + 1) % save_every == 0:
                self._flush()
                logger.info(f"  Added {added} | rejected {rejected}")

        self._flush()
        logger.info(f"Batch done | added={added} | rejected={rejected}")
        return added, rejected

    def add_from_crawler_results(
        self,
        results: list,          # List[DownloadResult]
        captions: Dict[str, str],
    ) -> Tuple[int, int]:
        """Convenience method: ingest crawler results + caption map."""
        items = []
        for r in results:
            cap = captions.get(r.filename, captions.get(r.local_path, ""))
            items.append({
                "image": r.local_path,
                "caption": cap,
                "source": r.source,
                "url": r.url,
                "prompt": r.query,
            })
        return self.add_batch(items)

    # ── Deduplication ──────────────────────────────────────────

    def _is_duplicate(self, phash: str) -> bool:
        for existing_hash in self._hashes.values():
            if are_duplicates(phash, existing_hash, threshold=1.0 - self.dedup_thresh):
                return True
        return False

    def deduplicate(self, dry_run: bool = False) -> int:
        """
        Full offline deduplication scan. Removes near-duplicate images.
        Much slower than add_image() check — run after bulk ingestion.

        Returns:
            Number of duplicates removed (or would be removed if dry_run=True)
        """
        logger.info(f"Deduplication scan | dry_run={dry_run} | n={len(self.captions)}")
        filenames = list(self._hashes.keys())
        to_remove = set()

        for i, fn_a in enumerate(filenames):
            if fn_a in to_remove:
                continue
            ha = self._hashes[fn_a]
            for fn_b in filenames[i + 1:]:
                if fn_b in to_remove:
                    continue
                hb = self._hashes[fn_b]
                if are_duplicates(ha, hb, threshold=1.0 - self.dedup_thresh):
                    to_remove.add(fn_b)  # Keep the first, remove the later one

        logger.info(f"Found {len(to_remove)} duplicates")
        if not dry_run:
            for fn in to_remove:
                img_path = self.img_dir / fn
                if img_path.exists():
                    img_path.unlink()
                self.captions.pop(fn, None)
                self.metadata.pop(fn, None)
                self._hashes.pop(fn, None)
            self._flush()
            self._rebuild_index()
            logger.info(f"Removed {len(to_remove)} duplicate images")

        return len(to_remove)

    # ── Statistics ─────────────────────────────────────────────

    def stats(self) -> dict:
        sources = {}
        for meta in self.metadata.values():
            src = meta.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1

        cap_lens = [len(c) for c in self.captions.values()]
        return {
            "total_images":    len(self.captions),
            "sources":         sources,
            "avg_caption_len": round(sum(cap_lens) / max(len(cap_lens), 1), 1),
            "min_caption_len": min(cap_lens, default=0),
            "max_caption_len": max(cap_lens, default=0),
            "dataset_root":    str(self.root),
        }

    def print_stats(self) -> None:
        s = self.stats()
        print(f"\n{'='*50}")
        print(f"  Genesis Dataset Statistics")
        print(f"{'='*50}")
        print(f"  Total images : {s['total_images']:,}")
        print(f"  Sources      : {s['sources']}")
        print(f"  Caption len  : avg={s['avg_caption_len']} min={s['min_caption_len']} max={s['max_caption_len']}")
        print(f"  Root         : {s['dataset_root']}")
        print(f"{'='*50}\n")

    # ── Export ─────────────────────────────────────────────────

    def export(self, fmt: str = "jsonl", output_dir: Optional[str] = None) -> str:
        """
        Export the dataset in training-ready format.

        Formats:
          jsonl   — HuggingFace datasets compatible (one JSON per line)
          csv     — Spreadsheet-friendly
          dreambooth  — txt + image pairs for DreamBooth / Kohya SS

        Returns path to exported file/directory.
        """
        out = Path(output_dir or (self.root / "exports"))
        out.mkdir(parents=True, exist_ok=True)

        if fmt == "jsonl":
            return self._export_jsonl(out)
        elif fmt == "csv":
            return self._export_csv(out)
        elif fmt in ("dreambooth", "txt"):
            return self._export_dreambooth(out)
        else:
            raise ValueError(f"Unknown export format: {fmt}")

    def _export_jsonl(self, out: Path) -> str:
        path = out / "dataset.jsonl"
        with open(path, "w") as f:
            for fn, cap in self.captions.items():
                meta = self.metadata.get(fn, {})
                record = {
                    "id":       Path(fn).stem,
                    "image":    f"images/{fn}",
                    "caption":  cap,
                    "source":   meta.get("source", ""),
                    "width":    meta.get("width", self.target_size[0]),
                    "height":   meta.get("height", self.target_size[1]),
                    "phash":    meta.get("phash", ""),
                    "timestamp": meta.get("timestamp", 0),
                }
                f.write(json.dumps(record) + "\n")
        logger.info(f"JSONL export → {path} ({len(self.captions)} records)")
        return str(path)

    def _export_csv(self, out: Path) -> str:
        import csv
        path = out / "dataset.csv"
        fieldnames = ["filename", "image_path", "caption", "source", "width", "height", "phash"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for fn, cap in self.captions.items():
                meta = self.metadata.get(fn, {})
                w.writerow({
                    "filename":   fn,
                    "image_path": f"images/{fn}",
                    "caption":    cap,
                    "source":     meta.get("source",""),
                    "width":      meta.get("width", self.target_size[0]),
                    "height":     meta.get("height", self.target_size[1]),
                    "phash":      meta.get("phash",""),
                })
        logger.info(f"CSV export → {path}")
        return str(path)

    def _export_dreambooth(self, out: Path) -> str:
        """DreamBooth format: image.png + image.txt side by side."""
        db_dir = out / "dreambooth"
        db_dir.mkdir(exist_ok=True)
        for fn, cap in self.captions.items():
            src = self.img_dir / fn
            if not src.exists():
                continue
            import shutil
            shutil.copy(str(src), str(db_dir / fn))
            txt_fn = Path(fn).stem + ".txt"
            (db_dir / txt_fn).write_text(cap)
        logger.info(f"DreamBooth export → {db_dir} ({len(self.captions)} pairs)")
        return str(db_dir)

    # ── HuggingFace Dataset integration ───────────────────────

    def to_hf_dataset(self):
        """
        Convert to a HuggingFace datasets.Dataset object.
        Enables direct use with Trainer and data streaming.
        """
        try:
            import datasets as hf_datasets
        except ImportError:
            raise ImportError("pip install datasets")

        records = []
        for fn, cap in self.captions.items():
            img_path = self.img_dir / fn
            if img_path.exists():
                records.append({
                    "image":   str(img_path),
                    "caption": cap,
                    **self.metadata.get(fn, {}),
                })

        ds = hf_datasets.Dataset.from_list(records)
        return ds.cast_column("image", hf_datasets.Image())

    # ── Internal helpers ───────────────────────────────────────

    def _flush(self) -> None:
        self._save_json(self.captions_path, self.captions)
        self._save_json(self.metadata_path, self.metadata)

    def _append_index(self, filename: str) -> None:
        meta = self.metadata.get(filename, {})
        record = {
            "id":       Path(filename).stem,
            "image":    f"images/{filename}",
            "caption":  self.captions.get(filename, ""),
            "source":   meta.get("source", ""),
            "width":    meta.get("width", self.target_size[0]),
            "height":   meta.get("height", self.target_size[1]),
            "phash":    meta.get("phash", ""),
            "timestamp": meta.get("timestamp", 0),
        }
        with open(self.index_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _rebuild_index(self) -> None:
        with open(self.index_path, "w") as f:
            for fn in self.captions:
                meta = self.metadata.get(fn, {})
                record = {
                    "id":       Path(fn).stem,
                    "image":    f"images/{fn}",
                    "caption":  self.captions[fn],
                    "source":   meta.get("source",""),
                    "width":    meta.get("width", self.target_size[0]),
                    "height":   meta.get("height", self.target_size[1]),
                    "phash":    meta.get("phash",""),
                    "timestamp": meta.get("timestamp", 0),
                }
                f.write(json.dumps(record) + "\n")

    @staticmethod
    def _load_json(path: Path) -> dict:
        if path.exists():
            try:
                with open(path) as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Corrupt JSON at {path}, starting fresh")
        return {}

    @staticmethod
    def _save_json(path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
