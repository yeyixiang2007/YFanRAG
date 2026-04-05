"""Configuration schema and helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ChunkingConfig:
    strategy: str = "fixed"
    chunk_size: int = 800
    chunk_overlap: int = 120
    separator: str = "\n\n"


@dataclass
class EmbeddingConfig:
    provider: str = "auto"
    model: Optional[str] = "BAAI/bge-small-en-v1.5"
    dims: Optional[int] = None
    batch_size: int = 64
    cache_embeddings: bool = True
    endpoint: Optional[str] = None
    api_key_env: Optional[str] = None


@dataclass
class StorageConfig:
    backend: str = "sqlite"
    path: str = "yfanrag.db"
    vector_extension: Optional[str] = "sqlite-vec"
    enable_fts: bool = True
    fts_backend: Optional[str] = "fts5"
    path_whitelist: list[str] = field(default_factory=list)
    extension_whitelist: list[str] = field(default_factory=list)
    slow_query_ms: float = 200.0


@dataclass
class RetrievalConfig:
    top_k: int = 5
    use_fts: bool = False
    hybrid_alpha: float = 0.5
    score_norm: str = "sigmoid"
    filters: Dict[str, Any] = field(default_factory=dict)
    range_filters: Dict[str, tuple[float | int | None, float | int | None]] = field(
        default_factory=dict
    )


@dataclass
class YFanRAGConfig:
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "YFanRAGConfig":
        chunking = ChunkingConfig(**data.get("chunking", {}))
        embedding = EmbeddingConfig(**data.get("embedding", {}))
        storage = StorageConfig(**data.get("storage", {}))
        retrieval = RetrievalConfig(**data.get("retrieval", {}))
        return cls(chunking=chunking, embedding=embedding, storage=storage, retrieval=retrieval)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunking": self.chunking.__dict__.copy(),
            "embedding": self.embedding.__dict__.copy(),
            "storage": self.storage.__dict__.copy(),
            "retrieval": self.retrieval.__dict__.copy(),
        }
