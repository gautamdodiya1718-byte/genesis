"""Active Data Learning — weakness detection, query generation, targeted expansion."""
from .weakness_detector import WeaknessDetector, WeaknessReport, CategoryWeakness
from .query_generator import QueryGenerator, CrawlQuery
from .dataset_expander import DatasetExpander, ExpansionResult

__all__ = [
    "WeaknessDetector", "WeaknessReport", "CategoryWeakness",
    "QueryGenerator", "CrawlQuery",
    "DatasetExpander", "ExpansionResult",
]
