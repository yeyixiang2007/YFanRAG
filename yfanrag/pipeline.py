"""Minimal end-to-end pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .interfaces import Chunker, Embedder, VectorStore
from .models import Chunk, Document


@dataclass
class SimplePipeline:
    chunker: Chunker
    embedder: Embedder
    store: VectorStore

    def ingest(self, documents: Iterable[Document]) -> List[Chunk]:
        all_chunks: List[Chunk] = []
        for document in documents:
            chunks = self.chunker.chunk(document)
            embeddings = self.embedder.embed([chunk.text for chunk in chunks])
            self.store.add(chunks, embeddings)
            all_chunks.extend(chunks)
        return all_chunks

    def upsert(self, documents: Iterable[Document], fts_index: object | None = None) -> List[Chunk]:
        docs = list(documents)
        if not docs:
            return []
        doc_ids = [doc.doc_id for doc in docs]
        self.delete(doc_ids=doc_ids, fts_index=fts_index)

        chunks = self.ingest(docs)
        if fts_index is not None and hasattr(fts_index, "add"):
            fts_index.add(chunks)
        return chunks

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

    def query(self, query_text: str, top_k: int = 5) -> List[Chunk]:
        embedding = self.embedder.embed([query_text])[0]
        return self.store.query(embedding, top_k)
