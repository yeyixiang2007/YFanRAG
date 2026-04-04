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


class CountingEmbedder:
    def __init__(self):
        self.calls = []

    def embed(self, texts):
        batch = list(texts)
        self.calls.append(batch)
        return [[float(len(text))] for text in batch]


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


def test_pipeline_query_with_filters():
    pipeline = SimplePipeline(
        chunker=FixedChunker(chunk_size=50, chunk_overlap=0),
        embedder=HashingEmbedder(dims=8),
        store=InMemoryVectorStore(),
    )
    docs = [
        Document(doc_id="doc-1", text="hello alpha"),
        Document(doc_id="doc-2", text="hello beta"),
    ]
    pipeline.ingest(docs)

    results = pipeline.query("hello", top_k=5, filters={"doc_id": "doc-2"})
    assert [chunk.doc_id for chunk in results] == ["doc-2"]


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


def test_pipeline_ingest_batches_embeddings():
    embedder = CountingEmbedder()
    store = InMemoryVectorStore()
    pipeline = SimplePipeline(
        chunker=FixedChunker(chunk_size=2, chunk_overlap=0),
        embedder=embedder,
        store=store,
        embed_batch_size=2,
        use_embedding_cache=False,
    )
    docs = [
        Document(doc_id="doc-1", text="abcd"),
        Document(doc_id="doc-2", text="efgh"),
    ]

    chunks = pipeline.ingest(docs)
    assert len(chunks) == 4
    assert len(embedder.calls) == 2
    assert all(len(batch) <= 2 for batch in embedder.calls)


def test_pipeline_ingest_uses_embedding_cache():
    embedder = CountingEmbedder()
    pipeline = SimplePipeline(
        chunker=FixedChunker(chunk_size=50, chunk_overlap=0),
        embedder=embedder,
        store=InMemoryVectorStore(),
        embed_batch_size=16,
        use_embedding_cache=True,
    )

    docs = [
        Document(doc_id="doc-1", text="hello"),
        Document(doc_id="doc-2", text="hello"),
    ]
    pipeline.ingest(docs)
    assert sum(len(batch) for batch in embedder.calls) == 1

    calls_before = len(embedder.calls)
    pipeline.ingest([Document(doc_id="doc-3", text="hello")])
    assert len(embedder.calls) == calls_before
