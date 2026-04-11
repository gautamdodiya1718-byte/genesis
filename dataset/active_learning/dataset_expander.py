"""
dataset/active_learning/dataset_expander.py
---------------------------------------------
Orchestrates targeted dataset expansion to fix model weaknesses.

Full feedback loop:
  WeaknessReport → QueryGenerator → Crawler → QualityFilter
                → DatasetBuilder → MetadataManager → (retrain signal)

The DatasetExpander is the bridge between the Evaluation Engine
and the Data Engine. It:
  1. Reads a WeaknessReport
  2. Generates targeted queries via QueryGenerator
  3. Runs the web crawler for those queries
  4. Filters crawled images through QualityFilter
  5. Adds accepted images to the dataset with weakness-aware metadata
  6. Reports an ExpansionResult back to the orchestrator

This closes the self-improvement loop:
  evaluate → detect weaknesses → expand dataset → retrain → evaluate
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from dataset.active_learning.weakness_detector import WeaknessReport
from dataset.active_learning.query_generator import QueryGenerator, CrawlQuery

logger = logging.getLogger(__name__)


@dataclass
class ExpansionResult:
    weakness_categories: List[str]
    queries_generated: int = 0
    images_crawled: int = 0
    images_accepted: int = 0
    images_rejected: int = 0
    added_to_dataset: int = 0
    by_category: Dict[str, int] = field(default_factory=dict)
    duration_s: float = 0.0
    error: Optional[str] = None

    def summary(self) -> str:
        return (
            f"DatasetExpander | cats={self.weakness_categories} "
            f"queries={self.queries_generated} crawled={self.images_crawled} "
            f"accepted={self.images_accepted} added={self.added_to_dataset} "
            f"{self.duration_s:.1f}s"
        )

    def to_dict(self) -> dict:
        return {
            "weakness_categories": self.weakness_categories,
            "queries_generated":   self.queries_generated,
            "images_crawled":      self.images_crawled,
            "images_accepted":     self.images_accepted,
            "images_rejected":     self.images_rejected,
            "added_to_dataset":    self.added_to_dataset,
            "by_category":         self.by_category,
            "duration_s":          round(self.duration_s, 1),
            "error":               self.error,
        }


class DatasetExpander:
    """
    Targeted dataset expansion driven by model weakness analysis.

    Integrates:
      - QueryGenerator (maps weaknesses → search queries)
      - ImageCrawler (fetches images from Openverse / Wikimedia)
      - QualityFilter (rejects low-quality images)
      - ImageCaptioner (captions accepted images)
      - DatasetBuilder (adds to dataset with metadata)

    Usage:
        expander = DatasetExpander.from_config(cfg)
        result = expander.expand(weakness_report)
    """

    def __init__(
        self,
        dataset_builder,       # DatasetBuilder instance
        crawler,               # ImageCrawler instance
        captioner,             # ImageCaptioner instance
        quality_filter,        # QualityFilter instance
        metadata_manager=None, # MetadataManager (optional)
        query_generator: Optional[QueryGenerator] = None,
        max_queries_per_cycle: int = 20,
        max_images_per_category: int = 100,
        output_dir: str = "outputs/active_learning",
    ):
        self.dataset_builder  = dataset_builder
        self.crawler          = crawler
        self.captioner        = captioner
        self.quality_filter   = quality_filter
        self.metadata_manager = metadata_manager
        self.query_gen        = query_generator or QueryGenerator()
        self.max_queries      = max_queries_per_cycle
        self.max_per_cat      = max_images_per_category
        self.output_dir       = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, cfg) -> "DatasetExpander":
        """Build DatasetExpander from unified Genesis config."""
        from core.model_manager import ModelManager
        from crawler.web_crawler import ImageCrawler
        from models.captioning.captioner import ImageCaptioner
        from dataset.builder import DatasetBuilder
        from dataset.intelligence.quality_filter import QualityFilter
        from dataset.intelligence.metadata_manager import MetadataManager

        mm = ModelManager(cfg)
        return cls(
            dataset_builder  = DatasetBuilder(cfg),
            crawler          = ImageCrawler(cfg),
            captioner        = ImageCaptioner(cfg, mm),
            quality_filter   = QualityFilter.from_config(cfg),
            metadata_manager = MetadataManager(
                db_path=cfg.get_nested("dataset.metadata_db",
                                       "outputs/metadata/dataset.db")
            ),
            max_queries_per_cycle=cfg.get_nested(
                "active_learning.max_queries_per_cycle", 20),
            max_images_per_category=cfg.get_nested(
                "active_learning.max_images_per_category", 100),
            output_dir=cfg.get_nested(
                "active_learning.output_dir", "outputs/active_learning"),
        )

    def expand(
        self,
        report: WeaknessReport,
        dry_run: bool = False,
    ) -> ExpansionResult:
        """
        Run one targeted expansion cycle from a WeaknessReport.

        Args:
            report:  WeaknessReport from WeaknessDetector
            dry_run: If True, generate queries but don't crawl/modify dataset

        Returns:
            ExpansionResult with full stats
        """
        t0 = time.time()
        result = ExpansionResult(weakness_categories=report.weak_categories)

        if not report.weaknesses:
            logger.info("No weaknesses found — dataset expansion skipped")
            result.duration_s = time.time() - t0
            return result

        # 1. Generate targeted queries
        queries = self.query_gen.generate(report, max_queries=self.max_queries)
        result.queries_generated = len(queries)
        logger.info(
            f"DatasetExpander: {len(queries)} queries for "
            f"categories: {report.weak_categories}"
        )

        if dry_run:
            self._save_queries(queries, report.run_id)
            result.duration_s = time.time() - t0
            return result

        # 2. Execute crawling + processing per query
        cat_counts: Dict[str, int] = {}
        for q in queries:
            cat = q.category
            if cat_counts.get(cat, 0) >= self.max_per_cat:
                logger.debug(f"  Category {cat} at limit, skipping: {q.query}")
                continue

            added = self._process_query(q, cat_counts)
            cat_counts[cat] = cat_counts.get(cat, 0) + added
            result.added_to_dataset += added

        result.by_category = cat_counts
        result.duration_s = time.time() - t0

        # 3. Persist expansion log
        log_path = self.output_dir / f"expansion_{report.run_id}_{int(t0)}.json"
        with open(log_path, "w") as f:
            json.dump({**result.to_dict(), "queries": [q.to_dict() for q in queries]},
                      f, indent=2)

        logger.info(result.summary())
        return result

    def _process_query(self, q: CrawlQuery, cat_counts: Dict[str, int]) -> int:
        """
        Crawl one query, filter, caption, and add to dataset.
        Returns number of images added.
        """
        added = 0
        try:
            # Crawl
            crawl_results = self.crawler.search(
                query=q.query,
                max_results=q.n_images,
                source=None if q.source_hint == "any" else q.source_hint,
            )
            if not crawl_results:
                return 0

            # Download images to temp location
            downloaded = self.crawler.download_batch(crawl_results)
            logger.debug(f"  [{q.query}] crawled={len(crawl_results)} downloaded={len(downloaded)}")

            # Quality filter
            items = [{"path": r.local_path, "caption": ""} for r in downloaded
                     if r.local_path]
            accepted, _ = self.quality_filter.filter_batch(items, show_progress=False)

            # Caption accepted images
            paths = [item["path"] for item in accepted]
            if not paths:
                return 0

            captions = self.captioner.caption_paths(paths)

            # Add to dataset
            for path, caption_data in zip(paths, captions):
                cap = caption_data.get("caption", "") if isinstance(caption_data, dict) else str(caption_data)
                if not cap:
                    continue
                # Add with weakness metadata for downstream use
                extra = {
                    "category": q.category,
                    "weakness_type": q.weakness_type,
                    "query": q.query,
                    "source": "active_learning",
                    "tags": q.tags,
                }
                success = self.dataset_builder.add_image(
                    image_path=path,
                    caption=cap,
                    source="active_learning",
                    extra_meta=extra,
                )
                if success:
                    added += 1

        except Exception as e:
            logger.warning(f"Query failed [{q.query}]: {e}")

        return added

    def _save_queries(self, queries: List[CrawlQuery], run_id: str) -> None:
        p = self.output_dir / f"queries_{run_id}.json"
        with open(p, "w") as f:
            json.dump([q.to_dict() for q in queries], f, indent=2)
        logger.info(f"[DRY RUN] Queries saved → {p}")

    def expand_from_feedback(
        self,
        feedback_categories: List[str],
        n_images: int = 50,
    ) -> ExpansionResult:
        """
        Direct expansion from user feedback categories (from API feedback_store).
        No WeaknessReport needed — uses raw category list.
        """
        queries = self.query_gen.generate_from_feedback(
            feedback_categories, n_per_category=3
        )
        # Build a minimal WeaknessReport wrapper
        from dataset.active_learning.weakness_detector import (
            WeaknessReport, CategoryWeakness
        )
        mock_report = WeaknessReport(
            run_id="user_feedback",
            model_version="current",
            baseline_run_id=None,
            weaknesses=[
                CategoryWeakness(
                    category=cat,
                    weakness_type="user_feedback",
                    severity="medium",
                    score=0.0,
                    threshold=0.0,
                )
                for cat in feedback_categories
            ],
        )
        return self.expand(mock_report)

    def status(self) -> dict:
        logs = sorted(self.output_dir.glob("expansion_*.json"))
        if not logs:
            return {"expansions": 0}
        latest = json.loads(logs[-1].read_text())
        return {
            "expansions": len(logs),
            "latest_added": latest.get("added_to_dataset", 0),
            "latest_categories": latest.get("weakness_categories", []),
        }
