"""Minimal end-to-end pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

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

    def query(self, query_text: str, top_k: int = 5) -> List[Chunk]:
        embedding = self.embedder.embed([query_text])[0]
        return self.store.query(embedding, top_k)
