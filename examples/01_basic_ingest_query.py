"""Example 1: ingest documents and run vector query with in-memory backend."""

from __future__ import annotations

from yfanrag.chunking import FixedChunker
from yfanrag.embedders import HashingEmbedder
from yfanrag.loaders.text import TextFileLoader
from yfanrag.pipeline import SimplePipeline
from yfanrag.vectorstores.memory import InMemoryVectorStore


def main() -> None:
    loader = TextFileLoader(paths=["docs"])
    docs = loader.load()

    pipeline = SimplePipeline(
        chunker=FixedChunker(chunk_size=200, chunk_overlap=40),
        embedder=HashingEmbedder(dims=8),
        store=InMemoryVectorStore(),
    )
    pipeline.ingest(docs)
    results = pipeline.query("SQLite 向量检索", top_k=3)

    print("Top 3 results:")
    for idx, chunk in enumerate(results, start=1):
        print(f"{idx}. {chunk.chunk_id} | doc={chunk.doc_id} | text={chunk.text[:60]!r}")


if __name__ == "__main__":
    main()
