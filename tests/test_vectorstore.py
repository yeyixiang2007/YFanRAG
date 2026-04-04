from yfanrag.embedders import HashingEmbedder
from yfanrag.models import Chunk
from yfanrag.vectorstores.memory import InMemoryVectorStore


def test_embedder_vector_shapes():
    embedder = HashingEmbedder(dims=4)
    vectors = embedder.embed(["hello", "world"])
    assert len(vectors) == 2
    assert all(len(vec) == 4 for vec in vectors)


def test_inmemory_vectorstore_query_top1():
    embedder = HashingEmbedder(dims=4)
    store = InMemoryVectorStore()

    chunks = [
        Chunk(chunk_id="c1", doc_id="d", text="hello", start=0, end=5),
        Chunk(chunk_id="c2", doc_id="d", text="world", start=6, end=11),
    ]
    embeddings = embedder.embed(["hello", "world"])
    store.add(chunks, embeddings)

    query_embedding = embedder.embed(["hello"])[0]
    results = store.query(query_embedding, top_k=1)
    assert results[0].chunk_id == "c1"


def test_inmemory_vectorstore_delete_by_doc_ids():
    embedder = HashingEmbedder(dims=4)
    store = InMemoryVectorStore()

    chunks = [
        Chunk(chunk_id="c1", doc_id="d1", text="hello", start=0, end=5),
        Chunk(chunk_id="c2", doc_id="d2", text="world", start=0, end=5),
    ]
    store.add(chunks, embedder.embed([chunk.text for chunk in chunks]))

    deleted = store.delete_by_doc_ids(["d1"])
    assert deleted == 1
    assert [chunk.chunk_id for chunk in store.chunks] == ["c2"]
