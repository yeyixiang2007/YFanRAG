from yfanrag.chunking import FixedChunker
from yfanrag.embedders import HashingEmbedder
from yfanrag.models import Document
from yfanrag.pipeline import SimplePipeline
from yfanrag.vectorstores.memory import InMemoryVectorStore


class DummyFtsIndex:
    def __init__(self):
        self.added_chunk_ids = []
        self.deleted_doc_ids = []

    def add(self, chunks):
        self.added_chunk_ids.extend([chunk.chunk_id for chunk in chunks])

    def delete_by_doc_ids(self, doc_ids):
        self.deleted_doc_ids.extend(doc_ids)
        return len(doc_ids)


def test_pipeline_minimal_loop():
    doc = Document(doc_id="doc-1", text="hello yfanrag")
    pipeline = SimplePipeline(
        chunker=FixedChunker(chunk_size=50, chunk_overlap=0),
        embedder=HashingEmbedder(dims=8),
        store=InMemoryVectorStore(),
    )

    chunks = pipeline.ingest([doc])
    assert len(chunks) == 1

    results = pipeline.query("hello", top_k=1)
    assert results
    assert results[0].doc_id == "doc-1"


def test_pipeline_upsert_replaces_existing_doc_chunks():
    store = InMemoryVectorStore()
    pipeline = SimplePipeline(
        chunker=FixedChunker(chunk_size=50, chunk_overlap=0),
        embedder=HashingEmbedder(dims=8),
        store=store,
    )
    fts = DummyFtsIndex()

    first = Document(doc_id="doc-1", text="hello v1")
    second = Document(doc_id="doc-1", text="hello v2 updated")

    first_chunks = pipeline.upsert([first], fts_index=fts)
    second_chunks = pipeline.upsert([second], fts_index=fts)

    assert len(first_chunks) == 1
    assert len(second_chunks) == 1
    assert len(store.chunks) == 1
    assert store.chunks[0].text == "hello v2 updated"
    assert fts.deleted_doc_ids == ["doc-1", "doc-1"]


def test_pipeline_delete_removes_doc_chunks():
    store = InMemoryVectorStore()
    pipeline = SimplePipeline(
        chunker=FixedChunker(chunk_size=50, chunk_overlap=0),
        embedder=HashingEmbedder(dims=8),
        store=store,
    )
    doc = Document(doc_id="doc-1", text="hello")
    pipeline.ingest([doc])

    stats = pipeline.delete(doc_ids=["doc-1"])
    assert stats == {"vector_deleted": 1, "fts_deleted": 0}
    assert store.chunks == []
