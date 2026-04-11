"""Dataset Intelligence Engine — deduplication, quality filtering, embeddings, metadata."""
from .deduplicator import Deduplicator, DeduplicationReport
from .quality_filter import QualityFilter, FilterDecision
from .embedding_index import EmbeddingIndex
from .metadata_manager import MetadataManager, MetadataRecord

__all__ = [
    "Deduplicator", "DeduplicationReport",
    "QualityFilter", "FilterDecision",
    "EmbeddingIndex",
    "MetadataManager", "MetadataRecord",
]
