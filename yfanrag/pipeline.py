"""Minimal end-to-end pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence
from time import perf_counter

from .embedders import embed_documents, embed_queries
from .interfaces import Chunker, Embedder, FieldFilters, RangeFilters, VectorStore
from .models import Chunk, Document
from .observability import log_slow_query


@dataclass(frozen=True)
class PreparedUpsert:
    doc_ids: list[str]
    chunks: list[Chunk]
    embeddings: list[list[float]]


@dataclass
class SimplePipeline:
    chunker: Chunker
    embedder: Embedder
    store: VectorStore
    embed_batch_size: int = 64
    use_embedding_cache: bool = True

    _embedding_cache: dict[str, list[float]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def ingest(self, documents: Iterable[Document]) -> List[Chunk]:
        start_ts = perf_counter()
        all_chunks, embeddings = self._prepare_chunks_and_embeddings(documents)
        if not all_chunks:
            return []

        self.store.add(all_chunks, embeddings)
        elapsed_ms = (perf_counter() - start_ts) * 1000.0
        log_slow_query("SimplePipeline.ingest", elapsed_ms, f"chunks={len(all_chunks)}")
        return all_chunks

    def upsert(self, documents: Iterable[Document], fts_index: object | None = None) -> List[Chunk]:
        prepared = self.prepare_upsert(documents)
        if not prepared.doc_ids:
            return []
        self.replace_vectors(prepared)
        self.replace_fts(prepared, fts_index=fts_index)
        return prepared.chunks

    def delete(
        self,
        doc_ids: Sequence[str],
        fts_index: object | None = None,
    ) -> dict[str, int]:
        ids = [doc_id for doc_id in doc_ids if doc_id]
        if not ids:
            return {"vector_deleted": 0, "fts_deleted": 0}

        fts_deleted = 0
        if fts_index is not None and hasattr(fts_index, "delete_by_doc_ids"):
            fts_deleted = int(fts_index.delete_by_doc_ids(ids))

        vector_deleted = int(self.store.delete_by_doc_ids(ids))
        return {"vector_deleted": vector_deleted, "fts_deleted": fts_deleted}

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        filters: FieldFilters | None = None,
        range_filters: RangeFilters | None = None,
    ) -> List[Chunk]:
        start_ts = perf_counter()
        embedding = embed_queries(self.embedder, [query_text])[0]
        result = self.store.query(
            embedding,
            top_k,
            filters=filters,
            range_filters=range_filters,
        )
        elapsed_ms = (perf_counter() - start_ts) * 1000.0
        log_slow_query("SimplePipeline.query", elapsed_ms, f"rows={len(result)}")
        return result

    def clear_embedding_cache(self) -> None:
        self._embedding_cache.clear()

    def prepare_upsert(self, documents: Iterable[Document]) -> PreparedUpsert:
        docs = list(documents)
        if not docs:
            return PreparedUpsert(doc_ids=[], chunks=[], embeddings=[])
        doc_ids = [doc.doc_id for doc in docs]
        chunks, embeddings = self._prepare_chunks_and_embeddings(docs)
        return PreparedUpsert(doc_ids=doc_ids, chunks=chunks, embeddings=embeddings)

    def replace_vectors(self, prepared: PreparedUpsert) -> None:
        if not prepared.doc_ids:
            return
        if hasattr(self.store, "replace_by_doc_ids"):
            self.store.replace_by_doc_ids(
                prepared.doc_ids,
                prepared.chunks,
                prepared.embeddings,
            )
            return
        self.store.delete_by_doc_ids(prepared.doc_ids)
        if prepared.chunks:
            self.store.add(prepared.chunks, prepared.embeddings)

    @staticmethod
    def replace_fts(
        prepared: PreparedUpsert,
        *,
        fts_index: object | None = None,
    ) -> None:
        if fts_index is None or not prepared.doc_ids:
            return
        if hasattr(fts_index, "replace_by_doc_ids"):
            fts_index.replace_by_doc_ids(prepared.doc_ids, prepared.chunks)
            return
        if hasattr(fts_index, "delete_by_doc_ids"):
            fts_index.delete_by_doc_ids(prepared.doc_ids)
        if hasattr(fts_index, "add"):
            fts_index.add(prepared.chunks)

    def _embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if self.embed_batch_size <= 0:
            raise ValueError("embed_batch_size must be positive")
        if not texts:
            return []

        vectors: list[list[float] | None] = [None] * len(texts)
        pending: dict[str, list[int]] = {}

        for idx, text in enumerate(texts):
            if self.use_embedding_cache and text in self._embedding_cache:
                vectors[idx] = list(self._embedding_cache[text])
            else:
                pending.setdefault(text, []).append(idx)

        unique_pending = list(pending.keys())
        for start in range(0, len(unique_pending), self.embed_batch_size):
            batch_texts = unique_pending[start : start + self.embed_batch_size]
            batch_vectors = embed_documents(self.embedder, batch_texts)
            if len(batch_vectors) != len(batch_texts):
                raise ValueError("embedder returned unexpected vector count")
            for text, raw_vec in zip(batch_texts, batch_vectors):
                vec = [float(x) for x in raw_vec]
                if self.use_embedding_cache:
                    self._embedding_cache[text] = vec
                for idx in pending[text]:
                    vectors[idx] = list(vec)

        if any(vec is None for vec in vectors):  # pragma: no cover - defensive
            raise RuntimeError("failed to generate embeddings for all chunks")
        return [vec for vec in vectors if vec is not None]

    def _prepare_chunks_and_embeddings(
        self,
        documents: Iterable[Document],
    ) -> tuple[List[Chunk], List[List[float]]]:
        all_chunks: List[Chunk] = []
        for document in documents:
            all_chunks.extend(self.chunker.chunk(document))
        if not all_chunks:
            return [], []
        embeddings = self._embed_texts([chunk.text for chunk in all_chunks])
        return all_chunks, embeddings
