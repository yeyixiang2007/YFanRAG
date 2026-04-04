"""Example 2: hybrid retrieval (vector + FTS) on sqlite backend."""

from __future__ import annotations

from yfanrag.chunking import FixedChunker
from yfanrag.embedders import HashingEmbedder
from yfanrag.fts import SqliteFtsIndex
from yfanrag.loaders.text import TextFileLoader
from yfanrag.pipeline import SimplePipeline
from yfanrag.retrievers import HybridRetriever
from yfanrag.vectorstores.sqlite_vec1 import SqliteVec1Store


def main() -> None:
    db_path = "examples_hybrid.db"
    loader = TextFileLoader(paths=["docs"])
    docs = loader.load()

    store = SqliteVec1Store(path=db_path, embedding_dim=8, load_extension=False)
    embedder = HashingEmbedder(dims=8)
    pipeline = SimplePipeline(
        chunker=FixedChunker(chunk_size=200, chunk_overlap=40),
        embedder=embedder,
        store=store,
    )

    fts = SqliteFtsIndex(path=db_path)
    try:
        pipeline.upsert(docs, fts_index=fts)
        retriever = HybridRetriever(
            embedder=embedder,
            vector_store=store,
            fts_index=fts,
            alpha=0.5,
        )
        hits = retriever.retrieve_with_scores("向量索引", top_k=3)
    finally:
        fts.close()
        store.close()

    print("Top 3 hybrid hits:")
    for idx, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        print(
            f"{idx}. {chunk.chunk_id} score={hit.fused_score:.4f} "
            f"(v={hit.vector_score:.4f}, fts={hit.fts_score:.4f})"
        )


if __name__ == "__main__":
    main()
