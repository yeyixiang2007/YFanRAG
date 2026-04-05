from __future__ import annotations

import pytest

from yfanrag.fts import FtsMatch
from yfanrag.models import Chunk
from yfanrag.retrievers import HybridRetriever
from yfanrag.vectorstores.memory import InMemoryVectorStore


class DummyEmbedder:
    def __init__(self, vector):
        self._vector = list(vector)

    def embed(self, texts):
        return [list(self._vector) for _ in texts]


class DummyFtsIndex:
    def __init__(self, matches):
        self._matches = list(matches)

    def query(self, query: str, top_k: int = 5):
        return self._matches[:top_k]


def test_hybrid_retriever_fuses_vector_and_fts_scores():
    store = InMemoryVectorStore()
    chunks = [
        Chunk(chunk_id="c1", doc_id="d1", text="vector only", start=0, end=11),
        Chunk(chunk_id="c2", doc_id="d1", text="shared", start=12, end=18),
        Chunk(chunk_id="c3", doc_id="d1", text="fts only boost", start=19, end=33),
    ]
    embeddings = [
        [1.0, 0.0],
        [0.6, 0.8],
        [0.0, 1.0],
    ]
    store.add(chunks, embeddings)

    fts = DummyFtsIndex(
        [
            FtsMatch(chunk_id="c2", doc_id="d1", text="shared", score=-2.0),
            FtsMatch(chunk_id="c3", doc_id="d1", text="fts only boost", score=-1.0),
        ]
    )

    retriever = HybridRetriever(
        embedder=DummyEmbedder([1.0, 0.0]),
        vector_store=store,
        fts_index=fts,
        alpha=0.5,
    )
    hits = retriever.retrieve_with_scores("hello", top_k=3)

    assert [hit.chunk.chunk_id for hit in hits] == ["c2", "c1", "c3"]
    assert hits[0].fused_score > hits[1].fused_score
    assert hits[0].chunk.metadata["hybrid_score"] == hits[0].fused_score


def test_hybrid_retriever_vector_only_when_fts_empty():
    store = InMemoryVectorStore()
    chunks = [
        Chunk(chunk_id="c1", doc_id="d1", text="hello", start=0, end=5),
        Chunk(chunk_id="c2", doc_id="d1", text="world", start=6, end=11),
    ]
    store.add(chunks, [[1.0, 0.0], [0.0, 1.0]])

    retriever = HybridRetriever(
        embedder=DummyEmbedder([1.0, 0.0]),
        vector_store=store,
        fts_index=DummyFtsIndex([]),
        alpha=0.5,
    )
    results = retriever.retrieve("query", top_k=2)
    assert [chunk.chunk_id for chunk in results] == ["c1", "c2"]


def test_hybrid_retriever_with_filters():
    store = InMemoryVectorStore()
    chunks = [
        Chunk(chunk_id="c1", doc_id="d1", text="hello", start=0, end=5),
        Chunk(chunk_id="c2", doc_id="d2", text="hello", start=6, end=11),
    ]
    store.add(chunks, [[1.0, 0.0], [1.0, 0.0]])

    retriever = HybridRetriever(
        embedder=DummyEmbedder([1.0, 0.0]),
        vector_store=store,
        fts_index=DummyFtsIndex([]),
        alpha=0.5,
    )
    results = retriever.retrieve("query", top_k=5, filters={"doc_id": "d2"})
    assert [chunk.chunk_id for chunk in results] == ["c2"]


def test_hybrid_retriever_requires_valid_alpha():
    with pytest.raises(ValueError):
        HybridRetriever(
            embedder=DummyEmbedder([1.0, 0.0]),
            vector_store=InMemoryVectorStore(),
            fts_index=DummyFtsIndex([]),
            alpha=1.2,
        )


def test_hybrid_retriever_supports_sigmoid_normalization():
    store = InMemoryVectorStore()
    chunks = [
        Chunk(chunk_id="c1", doc_id="d1", text="vector only", start=0, end=11),
        Chunk(chunk_id="c2", doc_id="d1", text="shared", start=12, end=18),
        Chunk(chunk_id="c3", doc_id="d1", text="fts only boost", start=19, end=33),
    ]
    store.add(
        chunks,
        [
            [1.0, 0.0],
            [0.6, 0.8],
            [0.0, 1.0],
        ],
    )
    retriever = HybridRetriever(
        embedder=DummyEmbedder([1.0, 0.0]),
        vector_store=store,
        fts_index=DummyFtsIndex(
            [
                FtsMatch(chunk_id="c2", doc_id="d1", text="shared", score=-2.0),
                FtsMatch(chunk_id="c3", doc_id="d1", text="fts only boost", score=-1.0),
            ]
        ),
        alpha=0.5,
        score_norm="sigmoid",
    )

    hits = retriever.retrieve_with_scores("hello", top_k=3)

    assert [hit.chunk.chunk_id for hit in hits] == ["c2", "c1", "c3"]
    assert all(0.0 <= hit.vector_score <= 1.0 for hit in hits)
    assert all(0.0 <= hit.fts_score <= 1.0 for hit in hits)
