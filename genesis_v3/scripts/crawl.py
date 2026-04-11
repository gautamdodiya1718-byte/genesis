#!/usr/bin/env python3
"""
scripts/crawl.py — Web image crawler CLI

Examples:
    python scripts/crawl.py --query "tropical beach"
    python scripts/crawl.py --queries_file queries.txt --max 100
    python scripts/crawl.py --query "nature" --sources openverse wikimedia
"""

import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import load_config, parse_cli_overrides
from core.logger import setup_logging
from crawler.web_crawler import ImageCrawler
import logging


def main():
    parser = argparse.ArgumentParser(description="Genesis web crawler")
    parser.add_argument("--config",       default="configs/base.yaml")
    parser.add_argument("--query",        type=str, default=None)
    parser.add_argument("--queries_file", type=str, default=None, help="Text file with one query per line")
    parser.add_argument("--max",          type=int, default=50, help="Max images per query")
    parser.add_argument("--sources",      nargs="+", default=["openverse", "wikimedia"])
    parser.add_argument("--output_dir",   type=str, default=None)
    parser.add_argument("overrides", nargs="*")

    args = parser.parse_args()
    overrides = parse_cli_overrides(args.overrides)
    if args.max:       overrides["crawler.max_images_per_query"] = args.max
    if args.output_dir: overrides["crawler.download_dir"] = args.output_dir

    cfg = load_config(args.config, overrides=overrides)
    setup_logging(cfg.system.log_level)
    logger = logging.getLogger(__name__)

    queries = []
    if args.query:
        queries = [args.query]
    elif args.queries_file:
        with open(args.queries_file) as f:
            queries = [l.strip() for l in f if l.strip()]

    if not queries:
        parser.error("Provide --query or --queries_file")

    crawler = ImageCrawler(cfg)
    results = crawler.crawl_multiple(queries=queries, sources=args.sources, max_per_query=args.max)

    print(f"\nDownloaded {len(results)} images to {cfg.crawler.download_dir}")
    for r in results[:10]:
        print(f"  [{r.source}] {r.filename} — {r.query}")
    if len(results) > 10:
        print(f"  ... and {len(results)-10} more")


if __name__ == "__main__":
    main()
