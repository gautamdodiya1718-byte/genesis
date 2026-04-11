#!/usr/bin/env python3
"""
scripts/caption.py — Image captioning CLI

Examples:
    python scripts/caption.py --image photo.jpg
    python scripts/caption.py --dir outputs/crawled_images --output captions.json
"""

import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import load_config, parse_cli_overrides
from core.logger import setup_logging
from core.model_manager import ModelManager
from models.captioning.captioner import ImageCaptioner


def main():
    parser = argparse.ArgumentParser(description="Genesis image captioner")
    parser.add_argument("--config",  default="configs/base.yaml")
    parser.add_argument("--image",   type=str, default=None, help="Single image path")
    parser.add_argument("--dir",     type=str, default=None, help="Directory of images")
    parser.add_argument("--output",  type=str, default=None, help="Output JSON file for captions")
    parser.add_argument("--model",   type=str, default=None, help="Override captioning model ID")
    parser.add_argument("overrides", nargs="*")

    args = parser.parse_args()
    overrides = parse_cli_overrides(args.overrides)
    if args.model: overrides["captioning.model_id"] = args.model

    cfg = load_config(args.config, overrides=overrides)
    setup_logging(cfg.system.log_level)

    mm = ModelManager(cfg.model_cache_dir)
    captioner = ImageCaptioner(cfg, mm)

    results = []
    if args.image:
        cap = captioner.caption(args.image)
        print(f"Caption: {cap}")
        results = [{"path": args.image, "caption": cap}]
    elif args.dir:
        results = captioner.caption_directory(args.dir)
        print(f"\nCaptioned {len(results)} images")
        for r in results[:5]:
            print(f"  {os.path.basename(r['path'])}: {r['caption'][:60]}...")
    else:
        parser.error("Provide --image or --dir")

    if args.output and results:
        out = {os.path.basename(r["path"]): r["caption"] for r in results}
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Captions saved to: {args.output}")


if __name__ == "__main__":
    main()
