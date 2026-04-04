"""Public API exports."""

from .benchmark import (
    BenchmarkCase,
    RetrievalItem,
    evaluate_retrieval_benchmark,
    load_benchmark_cases,
)
from .chunking import FixedChunker, RecursiveChunker
from .config import (
    ChunkingConfig,
    EmbeddingConfig,
    RetrievalConfig,
    StorageConfig,
    YFanRAGConfig,
)
from .embedders import EmbedderFactory, HashingEmbedder, HttpEmbedder
from .fts import FtsMatch, SqliteFtsIndex
from .interfaces import Chunker, DocumentLoader, Embedder, Retriever, VectorStore
from .loaders.text import TextFileLoader
from .migrations import (
    migrate_duckdb_vss_to_sqlite_vec1,
    migrate_sqlite_vec0_to_vec1,
    migrate_sqlite_vec1_to_duckdb_vss,
)
from .models import Chunk, Document
from .observability import configure_logging, get_logger, log_slow_query
from .pipeline import SimplePipeline
from .retrievers import HybridHit, HybridRetriever
from .security import ensure_path_in_whitelist, parse_whitelist, whitelist_from_env
from .vectorstores.duckdb_vss import DuckDbVssStore
from .vectorstores.memory import InMemoryVectorStore
from .vectorstores.sqlite_vec import SqliteVecStore
from .vectorstores.sqlite_vec1 import SqliteVec1Store

__all__ = [
    "Chunk",
    "Document",
    "BenchmarkCase",
    "RetrievalItem",
    "Chunker",
    "DocumentLoader",
    "Embedder",
    "Retriever",
    "VectorStore",
    "ChunkingConfig",
    "EmbeddingConfig",
    "RetrievalConfig",
    "StorageConfig",
    "YFanRAGConfig",
    "TextFileLoader",
    "FixedChunker",
    "RecursiveChunker",
    "HashingEmbedder",
    "HttpEmbedder",
    "EmbedderFactory",
    "InMemoryVectorStore",
    "SqliteVecStore",
    "SqliteVec1Store",
    "DuckDbVssStore",
    "FtsMatch",
    "SqliteFtsIndex",
    "SimplePipeline",
    "HybridHit",
    "HybridRetriever",
    "load_benchmark_cases",
    "evaluate_retrieval_benchmark",
    "migrate_sqlite_vec0_to_vec1",
    "migrate_sqlite_vec1_to_duckdb_vss",
    "migrate_duckdb_vss_to_sqlite_vec1",
    "configure_logging",
    "get_logger",
    "log_slow_query",
    "ensure_path_in_whitelist",
    "parse_whitelist",
    "whitelist_from_env",
]
