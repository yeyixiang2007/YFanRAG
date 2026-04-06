"""In-memory vector store for tests and local demos."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence
import math
from time import perf_counter

from ..interfaces import FieldFilters, RangeFilters
from ..models import Chunk
from ..observability import log_slow_query


@dataclass
class InMemoryVectorStore:
    embeddings: List[List[float]] = field(default_factory=list)
    chunks: List[Chunk] = field(default_factory=list)

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        for chunk, embedding in zip(chunks, embeddings):
            self.chunks.append(chunk)
            self.embeddings.append([float(x) for x in embedding])

    def query(
        self,
        embedding: Sequence[float],
        top_k: int,
        filters: FieldFilters | None = None,
        range_filters: RangeFilters | None = None,
    ) -> List[Chunk]:
        start_ts = perf_counter()
        if not self.embeddings:
            return []
        query_norm = self._norm(embedding)
        if query_norm == 0:
            return []

        scored = []
        for chunk, stored in zip(self.chunks, self.embeddings):
            if not self._matches_filters(chunk, filters=filters, range_filters=range_filters):
                continue
            score = self._dot(embedding, stored) / (query_norm * self._norm(stored))
            scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        result = [chunk for _, chunk in scored[:top_k]]
        elapsed_ms = (perf_counter() - start_ts) * 1000.0
        log_slow_query("InMemoryVectorStore.query", elapsed_ms, f"rows={len(self.chunks)}")
        return result

    def delete_by_doc_ids(self, doc_ids: Sequence[str]) -> int:
        targets = set(doc_ids)
        if not targets:
            return 0

        kept_chunks: List[Chunk] = []
        kept_embeddings: List[List[float]] = []
        deleted = 0
        for chunk, embedding in zip(self.chunks, self.embeddings):
            if chunk.doc_id in targets:
                deleted += 1
                continue
            kept_chunks.append(chunk)
            kept_embeddings.append(embedding)

        self.chunks = kept_chunks
        self.embeddings = kept_embeddings
        return deleted

    def replace_by_doc_ids(
        self,
        doc_ids: Sequence[str],
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        targets = set(doc_ids)
        kept_chunks: List[Chunk] = []
        kept_embeddings: List[List[float]] = []
        deleted = 0
        for chunk, embedding in zip(self.chunks, self.embeddings):
            if chunk.doc_id in targets:
                deleted += 1
                continue
            kept_chunks.append(chunk)
            kept_embeddings.append(embedding)
        for chunk, embedding in zip(chunks, embeddings):
            kept_chunks.append(chunk)
            kept_embeddings.append([float(x) for x in embedding])
        self.chunks = kept_chunks
        self.embeddings = kept_embeddings
        return deleted

    @staticmethod
    def _dot(a: Sequence[float], b: Sequence[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    @staticmethod
    def _norm(a: Sequence[float]) -> float:
        return math.sqrt(sum(x * x for x in a))

    @staticmethod
    def _field_value(chunk: Chunk, key: str) -> object | None:
        if hasattr(chunk, key):
            return getattr(chunk, key)
        return chunk.metadata.get(key)

    @classmethod
    def _matches_filters(
        cls,
        chunk: Chunk,
        filters: FieldFilters | None,
        range_filters: RangeFilters | None,
    ) -> bool:
        if filters:
            for key, expected in filters.items():
                if cls._field_value(chunk, key) != expected:
                    return False
        if range_filters:
            for key, (lower, upper) in range_filters.items():
                value = cls._field_value(chunk, key)
                if not isinstance(value, (int, float)):
                    return False
                if lower is not None and value < lower:
                    return False
                if upper is not None and value > upper:
                    return False
        return True
