"""Core interfaces and protocols."""

from __future__ import annotations

from typing import List, Protocol, Sequence, runtime_checkable

from .models import Chunk, Document

FieldFilters = dict[str, object]
RangeFilter = tuple[float | int | None, float | int | None]
RangeFilters = dict[str, RangeFilter]


@runtime_checkable
class DocumentLoader(Protocol):
    def load(self) -> List[Document]:
        """Load and normalize documents."""


@runtime_checkable
class Chunker(Protocol):
    def chunk(self, document: Document) -> List[Chunk]:
        """Split a document into chunks."""


@runtime_checkable
class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""


@runtime_checkable
class VectorStore(Protocol):
    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        """Persist chunks and embeddings into storage."""

    def query(
        self,
        embedding: Sequence[float],
        top_k: int,
        filters: FieldFilters | None = None,
        range_filters: RangeFilters | None = None,
    ) -> List[Chunk]:
        """Query similar chunks by vector embedding."""

    def delete_by_doc_ids(self, doc_ids: Sequence[str]) -> int:
        """Delete all chunks by document ids and return affected row count."""


@runtime_checkable
class Retriever(Protocol):
    def retrieve(
        self,
        query: str,
        top_k: int,
        filters: FieldFilters | None = None,
        range_filters: RangeFilters | None = None,
    ) -> List[Chunk]:
        """Retrieve chunks for a query."""
