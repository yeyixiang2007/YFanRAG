"""Retriever implementations."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, sqrt
from typing import Dict, List, Sequence

from .embedders import embed_queries
from .fts import FtsMatch, SqliteFtsIndex
from .interfaces import Embedder, FieldFilters, RangeFilters, Retriever, VectorStore
from .models import Chunk


@dataclass(frozen=True)
class HybridHit:
    chunk: Chunk
    fused_score: float
    vector_score: float
    fts_score: float


@dataclass
class HybridRetriever(Retriever):
    embedder: Embedder
    vector_store: VectorStore
    fts_index: SqliteFtsIndex
    alpha: float = 0.5
    score_norm: str = "sigmoid"

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")

    def retrieve(
        self,
        query: str,
        top_k: int,
        filters: FieldFilters | None = None,
        range_filters: RangeFilters | None = None,
    ) -> List[Chunk]:
        return [
            hit.chunk
            for hit in self.retrieve_with_scores(
                query=query,
                top_k=top_k,
                filters=filters,
                range_filters=range_filters,
            )
        ]

    def retrieve_with_scores(
        self,
        query: str,
        top_k: int = 5,
        vector_top_k: int | None = None,
        fts_top_k: int | None = None,
        filters: FieldFilters | None = None,
        range_filters: RangeFilters | None = None,
    ) -> List[HybridHit]:
        if top_k <= 0:
            return []
        vector_k = vector_top_k or top_k
        fts_k = fts_top_k or top_k

        embedding = embed_queries(self.embedder, [query])[0]
        vector_chunks = self.vector_store.query(
            embedding,
            vector_k,
            filters=filters,
            range_filters=range_filters,
        )
        fts_matches = self.fts_index.query(query, fts_k)

        vector_raw = self._vector_raw_scores(vector_chunks)
        fts_raw = self._fts_raw_scores(fts_matches)
        vector_norm = self._normalize(vector_raw)
        fts_norm = self._normalize(fts_raw)

        chunk_map: Dict[str, Chunk] = {}
        for chunk in vector_chunks:
            chunk_map[chunk.chunk_id] = chunk
        for match in fts_matches:
            chunk_map.setdefault(
                match.chunk_id,
                Chunk(
                    chunk_id=match.chunk_id,
                    doc_id=match.doc_id or "",
                    text=match.text,
                    start=0,
                    end=len(match.text),
                ),
            )

        hits: List[HybridHit] = []
        for chunk_id, chunk in chunk_map.items():
            if not self._matches_filters(chunk, filters=filters, range_filters=range_filters):
                continue
            vector_score = vector_norm.get(chunk_id, 0.0)
            fts_score = fts_norm.get(chunk_id, 0.0)
            fused_score = self.alpha * vector_score + (1.0 - self.alpha) * fts_score
            metadata = dict(chunk.metadata)
            metadata.update(
                {
                    "hybrid_score": fused_score,
                    "vector_score": vector_score,
                    "fts_score": fts_score,
                }
            )
            scored_chunk = Chunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                start=chunk.start,
                end=chunk.end,
                metadata=metadata,
            )
            hits.append(
                HybridHit(
                    chunk=scored_chunk,
                    fused_score=fused_score,
                    vector_score=vector_score,
                    fts_score=fts_score,
                )
            )

        hits.sort(
            key=lambda hit: (
                hit.fused_score,
                hit.vector_score,
                hit.fts_score,
                hit.chunk.chunk_id,
            ),
            reverse=True,
        )
        return hits[:top_k]

    @staticmethod
    def _matches_filters(
        chunk: Chunk,
        filters: FieldFilters | None,
        range_filters: RangeFilters | None,
    ) -> bool:
        if filters:
            for key, expected in filters.items():
                value = getattr(chunk, key, chunk.metadata.get(key))
                if value != expected:
                    return False
        if range_filters:
            for key, (lower, upper) in range_filters.items():
                value = getattr(chunk, key, chunk.metadata.get(key))
                if not isinstance(value, (int, float)):
                    return False
                if lower is not None and value < lower:
                    return False
                if upper is not None and value > upper:
                    return False
        return True

    def _normalize(self, raw_scores: Dict[str, float]) -> Dict[str, float]:
        if not raw_scores:
            return {}
        if self.score_norm == "none":
            return dict(raw_scores)
        if self.score_norm == "sigmoid":
            values = list(raw_scores.values())
            mean = sum(values) / float(len(values))
            variance = sum((value - mean) ** 2 for value in values) / float(len(values))
            stddev = sqrt(variance) if variance > 0.0 else 0.0
            scale = stddev if stddev > 1e-9 else 1.0
            normalized: Dict[str, float] = {}
            for chunk_id, score in raw_scores.items():
                z_score = (score - mean) / scale
                bounded = max(-60.0, min(60.0, z_score))
                normalized[chunk_id] = 1.0 / (1.0 + exp(-bounded))
            return normalized
        if self.score_norm != "minmax":
            raise ValueError(f"unsupported score normalization: {self.score_norm}")

        values = list(raw_scores.values())
        min_v = min(values)
        max_v = max(values)
        if max_v == min_v:
            return {chunk_id: 1.0 for chunk_id in raw_scores}
        denom = max_v - min_v
        return {
            chunk_id: (score - min_v) / denom for chunk_id, score in raw_scores.items()
        }

    @staticmethod
    def _vector_raw_scores(chunks: Sequence[Chunk]) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for idx, chunk in enumerate(chunks):
            distance = chunk.metadata.get("distance")
            if isinstance(distance, (int, float)) and isfinite(float(distance)):
                numeric = max(0.0, float(distance))
                score = 1.0 / (1.0 + numeric)
            else:
                score = 1.0 / (idx + 1)
            scores[chunk.chunk_id] = score
        return scores

    @staticmethod
    def _fts_raw_scores(matches: Sequence[FtsMatch]) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for idx, match in enumerate(matches):
            if isfinite(float(match.score)):
                # FTS5 bm25 score is lower-is-better, convert to higher-is-better.
                score = -float(match.score)
            else:
                score = 1.0 / (idx + 1)
            scores[match.chunk_id] = score
        return scores
