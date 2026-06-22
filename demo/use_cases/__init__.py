"""pgvector use case demos."""

from . import basic_search
from . import deduplication
from . import distance_metrics
from . import filtering
from . import hybrid_search
from . import index_comparison
from . import rag
from . import recommendations
from . import semantic_search

USE_CASES = {
    "1": ("Basic vector search (exact)", basic_search.run),
    "2": ("Semantic document search", semantic_search.run),
    "3": ("RAG retrieval simulation", rag.run),
    "4": ("Hybrid search (FTS + vector)", hybrid_search.run),
    "5": ("Recommendation engine", recommendations.run),
    "6": ("Duplicate / near-duplicate detection", deduplication.run),
    "7": ("Filtered search (metadata + vector)", filtering.run),
    "8": ("Distance metrics comparison", distance_metrics.run),
    "9": ("HNSW vs exact index comparison", index_comparison.run),
}
