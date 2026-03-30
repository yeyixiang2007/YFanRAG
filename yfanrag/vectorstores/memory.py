"""In-memory vector store for tests and local demos."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence
import math

from ..models import Chunk


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

    def query(self, embedding: Sequence[float], top_k: int) -> List[Chunk]:
        if not self.embeddings:
            return []
        query_norm = self._norm(embedding)
        if query_norm == 0:
            return []

        scored = []
        for chunk, stored in zip(self.chunks, self.embeddings):
            score = self._dot(embedding, stored) / (query_norm * self._norm(stored))
            scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]]

    @staticmethod
    def _dot(a: Sequence[float], b: Sequence[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    @staticmethod
    def _norm(a: Sequence[float]) -> float:
        return math.sqrt(sum(x * x for x in a))
