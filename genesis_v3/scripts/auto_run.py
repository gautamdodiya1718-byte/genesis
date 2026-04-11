#!/usr/bin/env python3
"""
scripts/auto_run.py
--------------------
Genesis autonomous pipeline entry point.

Runs the full generate → crawl → caption → dataset loop.

Examples:
    # Single cycle with all tasks
    python scripts/auto_run.py

    # 5 cycles, 1 hour apart
    python scripts/auto_run.py --cycles 5 --interval 3600

    # Crawl-only mode, 3 queries
    python scripts/auto_run.py --skip_generate --crawl_queries 5

    # Use LCM for fast CPU generation
    python scripts/auto_run.py --lcm --generate_count 10

    # Also trigger training after 5 cycles of data collection
    python scripts/auto_run.py --cycles 5 --train_after
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import load_config, parse_cli_overrides
from automation.controller import GenesisController


def main():
    parser = argparse.ArgumentParser(
        description="Genesis — Autonomous Generative AI Pipeline"
    )
    parser.add_argument("--config",   default="configs/base.yaml", help="Config file path")
    parser.add_argument("--cycles",   type=int,   default=None,    help="Number of automation cycles")
    parser.add_argument("--interval", type=int,   default=None,    help="Seconds between cycles")

    # Task toggles
    parser.add_argument("--skip_generate", action="store_true", help="Skip image generation")
    parser.add_argument("--skip_crawl",    action="store_true", help="Skip web crawling")
    parser.add_argument("--skip_caption",  action="store_true", help="Skip image captioning")
    parser.add_argument("--train_after",   action="store_true", help="Run training after automation")

    # Generation options
    parser.add_argument("--generate_count", type=int,   default=None, help="Images to generate per cycle")
    parser.add_argument("--crawl_queries",  type=int,   default=None, help="Crawl queries per cycle")
    parser.add_argument("--categories",     nargs="+",  default=None, help="Prompt categories to use")
    parser.add_argument("--lcm",    action="store_true", help="Force LCM (fast CPU) generation")

    # Config overrides (key=value pairs)
    parser.add_argument("overrides", nargs="*", help="Config overrides: key=value ...")

    args = parser.parse_args()

    # Build overrides dict
    overrides = parse_cli_overrides(args.overrides)

    if args.cycles:          overrides["automation.max_cycles"] = args.cycles
    if args.interval:        overrides["automation.cycle_interval_seconds"] = args.interval
    if args.generate_count:  overrides["automation.generate_count"] = args.generate_count
    if args.crawl_queries:   overrides["automation.crawl_queries"] = args.crawl_queries
    if args.skip_generate:   overrides["automation.tasks.generate_images"] = False
    if args.skip_crawl:      overrides["automation.tasks.crawl_images"] = False
    if args.skip_caption:    overrides["automation.tasks.caption_images"] = False
    if args.lcm:             overrides["lcm.enabled"] = True
    if args.train_after:
        overrides["automation.tasks.train_vae"] = True
        overrides["automation.tasks.train_diffusion"] = True

    cfg = load_config(args.config, overrides=overrides)
    controller = GenesisController(cfg)

    if args.categories:
        controller.run_cycle_with_categories(args.categories)
    else:
        controller.run()


if __name__ == "__main__":
    main()
