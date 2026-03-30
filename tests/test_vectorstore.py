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
