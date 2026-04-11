"""
dataset/active_learning/query_generator.py
--------------------------------------------
Generates targeted crawl queries from a WeaknessReport.

Maps each weakness category + type to specific search terms that
will yield training-relevant images when fed to the web crawler.

Query strategy per weakness type:
  semantic  — broad concept queries + related subjects
  aesthetic — style-focused queries ("award winning", "professional photography")
  variance  — anchor queries with strong style descriptors
  regression — both concept + style to rebuild signal

Output: List[CrawlQuery] consumed by dataset_expander.py
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from dataset.active_learning.weakness_detector import WeaknessReport, CategoryWeakness

logger = logging.getLogger(__name__)


@dataclass
class CrawlQuery:
    """A single search query with metadata for targeted crawling."""
    query: str
    category: str
    weakness_type: str
    priority: float          # higher = crawl first
    n_images: int = 50       # target number of images to collect
    source_hint: str = "any" # "openverse" | "wikimedia" | "any"
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "query": self.query, "category": self.category,
            "weakness_type": self.weakness_type,
            "priority": round(self.priority, 3),
            "n_images": self.n_images,
            "source_hint": self.source_hint,
            "tags": self.tags,
        }


# ── Category → query expansion templates ─────────────────────

_CATEGORY_SUBJECTS: Dict[str, List[str]] = {
    "portrait": [
        "portrait photography person", "headshot professional",
        "face closeup natural light", "environmental portrait",
        "candid portrait street", "studio portrait soft light",
        "portrait elderly person", "portrait child natural",
    ],
    "landscape": [
        "landscape photography nature", "mountain valley sunrise",
        "forest path autumn", "ocean cliff sunset",
        "desert dunes golden hour", "lake reflection mountain",
        "rolling hills countryside", "waterfall long exposure",
    ],
    "cityscape": [
        "cityscape night lights", "urban street photography",
        "architecture building modern", "aerial city view",
        "neon lights rain city", "bridge skyline city",
        "downtown skyscraper", "city alley moody",
    ],
    "animal": [
        "wildlife photography animal", "dog portrait studio",
        "cat natural light", "bird in flight",
        "horse meadow", "wildlife nature predator",
        "animal closeup macro", "pet photography",
    ],
    "art": [
        "digital art illustration", "oil painting fine art",
        "watercolor artwork nature", "concept art fantasy",
        "surreal painting dreamlike", "abstract expressionism",
        "impressionist landscape painting", "art nouveau illustration",
    ],
    "food": [
        "food photography plating", "restaurant dish professional",
        "ingredients flat lay", "dessert macro photography",
        "street food market", "coffee latte art",
    ],
    "architecture": [
        "architecture interior design", "building exterior modern",
        "historical architecture facade", "minimalist interior",
        "gothic cathedral architecture", "japanese architecture",
    ],
    "texture": [
        "texture material surface", "abstract pattern detail",
        "fabric close up", "stone wall texture",
        "wood grain texture", "metal surface macro",
    ],
}

_AESTHETIC_MODIFIERS = [
    "award winning photography",
    "professional photography",
    "highly detailed sharp focus",
    "golden hour lighting",
    "studio lighting professional",
    "4k ultra detailed",
    "masterpiece fine art",
]

_STYLE_ANCHORS = [
    "DSLR photograph",
    "35mm film photography",
    "studio professional",
    "National Geographic style",
    "fine art photography",
]


class QueryGenerator:
    """
    Generates targeted crawl queries from a WeaknessReport.

    Usage:
        qgen = QueryGenerator()
        queries = qgen.generate(weakness_report, max_queries=30)
        # Feed queries to DatasetExpander / web crawler
    """

    def __init__(
        self,
        queries_per_weakness: int = 5,
        n_images_per_query: int = 50,
        boost_critical: float = 2.0,
        custom_subjects: Optional[Dict[str, List[str]]] = None,
    ):
        self.queries_per_weakness = queries_per_weakness
        self.n_images_per_query   = n_images_per_query
        self.boost_critical       = boost_critical
        self._subjects = {**_CATEGORY_SUBJECTS, **(custom_subjects or {})}

    def generate(
        self,
        report: WeaknessReport,
        max_queries: int = 50,
        seed: Optional[int] = None,
    ) -> List[CrawlQuery]:
        """
        Generate CrawlQuery list from a WeaknessReport.

        Critical weaknesses get 2× more queries and higher priority.
        Returns queries sorted by priority descending.
        """
        if seed is not None:
            random.seed(seed)

        queries: List[CrawlQuery] = []

        for weakness in report.weaknesses:
            cat_queries = self._generate_for_weakness(weakness)
            queries.extend(cat_queries)

        # Deduplicate by query string
        seen: set = set()
        unique = []
        for q in queries:
            if q.query not in seen:
                seen.add(q.query)
                unique.append(q)

        # Sort by priority
        unique.sort(key=lambda q: q.priority, reverse=True)

        result = unique[:max_queries]
        logger.info(
            f"QueryGenerator: {len(result)} queries from "
            f"{len(report.weaknesses)} weaknesses "
            f"(weak cats: {', '.join(report.weak_categories)})"
        )
        return result

    def _generate_for_weakness(
        self, weakness: CategoryWeakness,
    ) -> List[CrawlQuery]:
        cat   = weakness.category
        wtype = weakness.weakness_type
        base_priority = weakness.priority
        n_imgs = self.n_images_per_query

        # Boost critical/high
        if weakness.severity in ("critical", "high"):
            base_priority *= self.boost_critical
            n_imgs = int(n_imgs * 1.5)

        subjects = self._subjects.get(cat, [cat + " photography"])
        selected = random.sample(subjects, min(self.queries_per_weakness, len(subjects)))

        queries: List[CrawlQuery] = []

        if wtype == "semantic":
            # Broad concept queries to teach the model the subject
            for subj in selected:
                queries.append(CrawlQuery(
                    query=subj,
                    category=cat,
                    weakness_type=wtype,
                    priority=base_priority,
                    n_images=n_imgs,
                    source_hint="any",
                    tags=[cat, "semantic", weakness.severity],
                ))
            # Add a few style-anchored versions
            for i, subj in enumerate(selected[:2]):
                anchor = _STYLE_ANCHORS[i % len(_STYLE_ANCHORS)]
                queries.append(CrawlQuery(
                    query=f"{subj} {anchor}",
                    category=cat,
                    weakness_type=wtype,
                    priority=base_priority * 0.8,
                    n_images=max(20, n_imgs // 2),
                    tags=[cat, "semantic", "styled"],
                ))

        elif wtype == "aesthetic":
            # Style-heavy queries to improve visual quality signal
            for subj in selected[:3]:
                for mod in random.sample(_AESTHETIC_MODIFIERS, min(2, len(_AESTHETIC_MODIFIERS))):
                    queries.append(CrawlQuery(
                        query=f"{subj} {mod}",
                        category=cat,
                        weakness_type=wtype,
                        priority=base_priority,
                        n_images=n_imgs,
                        source_hint="openverse",
                        tags=[cat, "aesthetic", "high_quality"],
                    ))

        elif wtype == "regression":
            # Combine concept + style to restore lost signal
            for subj in selected:
                queries.append(CrawlQuery(
                    query=subj,
                    category=cat,
                    weakness_type=wtype,
                    priority=base_priority * 1.2,  # extra boost for regressions
                    n_images=int(n_imgs * 1.2),
                    tags=[cat, "regression", "recovery"],
                ))

        elif wtype == "variance":
            # Anchor queries with strong descriptors to reduce variance
            for subj in selected[:3]:
                anchor = random.choice(_STYLE_ANCHORS)
                queries.append(CrawlQuery(
                    query=f"{subj} {anchor}",
                    category=cat,
                    weakness_type=wtype,
                    priority=base_priority,
                    n_images=n_imgs,
                    tags=[cat, "variance", "anchor"],
                ))

        return queries

    def generate_from_feedback(
        self,
        feedback_categories: List[str],
        n_per_category: int = 3,
    ) -> List[CrawlQuery]:
        """
        Generate queries from user feedback categories.
        Used when the API's feedback_store reports repeated failures.
        """
        queries = []
        for cat in feedback_categories:
            subjects = self._subjects.get(cat, [cat + " photography"])
            for subj in subjects[:n_per_category]:
                queries.append(CrawlQuery(
                    query=subj,
                    category=cat,
                    weakness_type="user_feedback",
                    priority=3.0,  # user feedback is high priority
                    n_images=30,
                    tags=[cat, "user_feedback"],
                ))
        return queries

    def to_crawler_queries(self, crawl_queries: List[CrawlQuery]) -> List[Dict]:
        """Convert to format expected by web_crawler.ImageCrawler."""
        return [
            {
                "query": q.query,
                "max_results": q.n_images,
                "source": q.source_hint if q.source_hint != "any" else None,
                "metadata": {
                    "category": q.category,
                    "weakness_type": q.weakness_type,
                    "priority": q.priority,
                    "tags": q.tags,
                },
            }
            for q in crawl_queries
        ]
