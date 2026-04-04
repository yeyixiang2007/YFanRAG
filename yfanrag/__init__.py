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
from .models import Chunk, Document
from .pipeline import SimplePipeline
from .retrievers import HybridHit, HybridRetriever
from .vectorstores.memory import InMemoryVectorStore
from .vectorstores.sqlite_vec import SqliteVecStore

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
    "FtsMatch",
    "SqliteFtsIndex",
    "SimplePipeline",
    "HybridHit",
    "HybridRetriever",
    "load_benchmark_cases",
    "evaluate_retrieval_benchmark",
]
