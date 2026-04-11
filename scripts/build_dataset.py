#!/usr/bin/env python3
"""
scripts/build_dataset.py — Dataset management CLI

Examples:
    python scripts/build_dataset.py --stats
    python scripts/build_dataset.py --image_dir outputs/crawled_images --captions_json captions.json
    python scripts/build_dataset.py --dedup --dry_run
    python scripts/build_dataset.py --export --format jsonl
    python scripts/build_dataset.py --export --format dreambooth --output_dir exports/
"""

import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import load_config, parse_cli_overrides
from core.logger import setup_logging
from dataset.builder import DatasetBuilder


def main():
    parser = argparse.ArgumentParser(description="Genesis dataset builder")
    parser.add_argument("--config",       default="configs/base.yaml")
    parser.add_argument("--image_dir",    type=str, default=None, help="Add images from directory")
    parser.add_argument("--captions_json",type=str, default=None, help="JSON file: filename→caption")
    parser.add_argument("--dedup",        action="store_true",    help="Run deduplication")
    parser.add_argument("--dry_run",      action="store_true",    help="Dry run (don't delete)")
    parser.add_argument("--stats",        action="store_true",    help="Print dataset statistics")
    parser.add_argument("--export",       action="store_true",    help="Export dataset")
    parser.add_argument("--format",       type=str, default="jsonl",
                        choices=["jsonl","csv","dreambooth"], help="Export format")
    parser.add_argument("--output_dir",   type=str, default=None)
    parser.add_argument("overrides", nargs="*")

    args = parser.parse_args()
    cfg = load_config(args.config, overrides=parse_cli_overrides(args.overrides))
    setup_logging(cfg.system.log_level)

    ds = DatasetBuilder(cfg)

    if args.image_dir:
        captions = {}
        if args.captions_json and os.path.exists(args.captions_json):
            with open(args.captions_json) as f:
                captions = json.load(f)

        from pathlib import Path
        EXTS = {".jpg",".jpeg",".png",".webp"}
        image_paths = sorted(p for p in Path(args.image_dir).iterdir() if p.suffix.lower() in EXTS)
        print(f"Adding {len(image_paths)} images...")

        items = []
        for p in image_paths:
            cap = captions.get(p.name, captions.get(str(p), ""))
            items.append({"image": str(p), "caption": cap or "image", "source": "user"})

        added, rejected = ds.add_batch(items)
        print(f"Added: {added} | Rejected: {rejected}")

    if args.dedup:
        removed = ds.deduplicate(dry_run=args.dry_run)
        action = "Would remove" if args.dry_run else "Removed"
        print(f"{action} {removed} duplicate images")

    if args.stats or not any([args.image_dir, args.dedup, args.export]):
        ds.print_stats()

    if args.export:
        path = ds.export(fmt=args.format, output_dir=args.output_dir)
        print(f"Exported ({args.format}) → {path}")


if __name__ == "__main__":
    main()
