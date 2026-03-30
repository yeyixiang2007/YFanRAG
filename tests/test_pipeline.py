from yfanrag.chunking import FixedChunker
from yfanrag.embedders import HashingEmbedder
from yfanrag.models import Document
from yfanrag.pipeline import SimplePipeline
from yfanrag.vectorstores.memory import InMemoryVectorStore


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
