"""Command line interface for YFanRAG."""

from __future__ import annotations

import argparse
import json
import sys

from .chunking import FixedChunker, RecursiveChunker
from .embedders import HashingEmbedder, HttpEmbedder
from .fts import SqliteFtsIndex
from .loaders.text import TextFileLoader
from .pipeline import SimplePipeline
from .vectorstores.memory import InMemoryVectorStore
from .vectorstores.sqlite_vec import SqliteVecStore


def _build_chunker(args: argparse.Namespace):
    if args.chunker == "recursive":
        return RecursiveChunker(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    return FixedChunker(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )


def _build_embedder(args: argparse.Namespace):
    if args.embedder == "http":
        return HttpEmbedder(
            endpoint=args.endpoint,
            model=args.model,
            api_key_env=args.api_key_env,
        )
    return HashingEmbedder(dims=args.dims)


def _build_store(args: argparse.Namespace):
    if args.store == "memory":
        return InMemoryVectorStore()
    return SqliteVecStore(
        path=args.db,
        embedding_dim=args.dims,
        distance_metric=args.distance_metric,
    )


def cmd_ingest(args: argparse.Namespace) -> int:
    loader = TextFileLoader(paths=args.paths)
    docs = loader.load()
    if not docs:
        print("No documents found.")
        return 1

    chunker = _build_chunker(args)
    embedder = _build_embedder(args)
    store = _build_store(args)

    pipeline = SimplePipeline(chunker=chunker, embedder=embedder, store=store)
    chunks = pipeline.ingest(docs)

    if args.enable_fts:
        fts = SqliteFtsIndex(path=args.db)
        fts.add(chunks)
        fts.close()

    print(f"Ingested {len(chunks)} chunks from {len(docs)} documents.")
    if args.store == "memory":
        print("Warning: memory store is not persisted across runs.")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    embedder = _build_embedder(args)
    store = _build_store(args)

    embedding = embedder.embed([args.query])[0]
    results = store.query(embedding, args.top_k)
    for idx, chunk in enumerate(results, start=1):
        payload = {
            "rank": idx,
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "start": chunk.start,
            "end": chunk.end,
            "distance": chunk.metadata.get("distance"),
            "text": chunk.text,
        }
        print(json.dumps(payload, ensure_ascii=False))
    return 0


def cmd_fts_query(args: argparse.Namespace) -> int:
    fts = SqliteFtsIndex(path=args.db)
    results = fts.query(args.query, args.top_k)
    for idx, match in enumerate(results, start=1):
        payload = {
            "rank": idx,
            "chunk_id": match.chunk_id,
            "doc_id": match.doc_id,
            "score": match.score,
            "text": match.text,
        }
        print(json.dumps(payload, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yfanrag")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Ingest documents into the store")
    ingest.add_argument("paths", nargs="+", help="Paths to files or directories")
    ingest.add_argument("--db", default="yfanrag.db")
    ingest.add_argument("--store", choices=["sqlite-vec", "memory"], default="sqlite-vec")
    ingest.add_argument("--chunker", choices=["fixed", "recursive"], default="fixed")
    ingest.add_argument("--chunk-size", type=int, default=800)
    ingest.add_argument("--chunk-overlap", type=int, default=120)
    ingest.add_argument("--embedder", choices=["hashing", "http"], default="hashing")
    ingest.add_argument("--dims", type=int, default=8)
    ingest.add_argument("--endpoint")
    ingest.add_argument("--model")
    ingest.add_argument("--api-key-env")
    ingest.add_argument("--enable-fts", action="store_true")
    ingest.add_argument("--distance-metric", choices=["l2", "cosine"], default=None)
    ingest.set_defaults(func=cmd_ingest)

    query = sub.add_parser("query", help="Vector search by query text")
    query.add_argument("query")
    query.add_argument("--db", default="yfanrag.db")
    query.add_argument("--store", choices=["sqlite-vec", "memory"], default="sqlite-vec")
    query.add_argument("--embedder", choices=["hashing", "http"], default="hashing")
    query.add_argument("--dims", type=int, default=8)
    query.add_argument("--endpoint")
    query.add_argument("--model")
    query.add_argument("--api-key-env")
    query.add_argument("--top-k", type=int, default=5)
    query.add_argument("--distance-metric", choices=["l2", "cosine"], default=None)
    query.set_defaults(func=cmd_query)

    fts_query = sub.add_parser("fts-query", help="FTS5 query by text")
    fts_query.add_argument("query")
    fts_query.add_argument("--db", default="yfanrag.db")
    fts_query.add_argument("--top-k", type=int, default=5)
    fts_query.set_defaults(func=cmd_fts_query)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
