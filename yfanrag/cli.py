"""Command line interface for YFanRAG."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path

from .benchmark import RetrievalItem, evaluate_retrieval_benchmark, load_benchmark_cases
from .chunking import FixedChunker, RecursiveChunker, StructureAwareChunker
from .embedders import FastEmbedder, HashingEmbedder, HttpEmbedder, embed_queries, embedder_dims
from .fts import SqliteFtsIndex
from .loaders.text import TextFileLoader
from .migrations import (
    migrate_duckdb_vss_to_sqlite_vec1,
    migrate_sqlite_vec0_to_vec1,
    migrate_sqlite_vec1_to_duckdb_vss,
)
from .observability import configure_logging
from .pipeline import SimplePipeline
from .retrievers import HybridRetriever
from .vectorstores.duckdb_vss import DuckDbVssStore
from .vectorstores.memory import InMemoryVectorStore
from .vectorstores.sqlite_vec import SqliteVecStore
from .vectorstores.sqlite_vec1 import SqliteVec1Store

_NUMERIC_FILTER_FIELDS = {"start", "end", "index"}
_STORE_CHOICES = ["sqlite-vec", "sqlite-vec1", "duckdb-vss", "memory"]


def _build_chunker(args: argparse.Namespace):
    if args.chunker == "structured":
        return StructureAwareChunker(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
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
    if args.embedder in {"auto", "fastembed"}:
        try:
            importlib.import_module("fastembed")
            return FastEmbedder(
                model_name=args.model or "BAAI/bge-small-en-v1.5",
            )
        except ImportError:
            if args.embedder == "fastembed":
                raise RuntimeError("fastembed is not installed. Install with `pip install fastembed`.")
    return HashingEmbedder(dims=args.dims)


def _build_store(args: argparse.Namespace, embedding_dim: int | None = None):
    if args.store == "memory":
        return InMemoryVectorStore()
    dims = embedding_dim if embedding_dim is not None else getattr(args, "dims", None)
    distance_metric = getattr(args, "distance_metric", None)
    if args.store == "sqlite-vec":
        return SqliteVecStore(
            path=args.db,
            embedding_dim=dims,
            distance_metric=distance_metric,
        )
    if args.store == "sqlite-vec1":
        return SqliteVec1Store(
            path=args.db,
            embedding_dim=dims,
            distance_metric=distance_metric or "l2",
            load_extension=not getattr(args, "disable_sqlite_extension", False),
            extension_path=getattr(args, "sqlite_extension_path", None),
            extension_whitelist=getattr(args, "extension_whitelist", None),
        )
    if args.store == "duckdb-vss":
        return DuckDbVssStore(
            path=args.db,
            embedding_dim=dims,
            distance_metric=distance_metric or "l2",
            enable_vss=not getattr(args, "disable_vss_extension", False),
            persistent_index=getattr(args, "vss_persistent_index", False),
            fail_if_no_vss=False,
        )
    raise ValueError(f"unsupported store backend: {args.store}")


def _coerce_filter_value(key: str, raw: str) -> object:
    if key in _NUMERIC_FILTER_FIELDS:
        return int(raw)
    return raw


def _parse_query_filters(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, tuple[float | int | None, float | int | None]]]:
    filters: dict[str, object] = {}
    range_filters: dict[str, tuple[float | int | None, float | int | None]] = {}

    for item in getattr(args, "filters", []) or []:
        if "=" not in item:
            raise ValueError(f"invalid --filter format: {item}; expected key=value")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("filter key cannot be empty")
        filters[key] = _coerce_filter_value(key, raw_value.strip())

    for item in getattr(args, "ranges", []) or []:
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"invalid --range format: {item}; expected key:min:max"
            )
        key, lower_raw, upper_raw = (part.strip() for part in parts)
        if not key:
            raise ValueError("range key cannot be empty")
        if key not in _NUMERIC_FILTER_FIELDS:
            raise ValueError(
                f"range key must be one of {sorted(_NUMERIC_FILTER_FIELDS)}"
            )
        lower = int(lower_raw) if lower_raw else None
        upper = int(upper_raw) if upper_raw else None
        range_filters[key] = (lower, upper)

    return filters, range_filters


def cmd_ingest(args: argparse.Namespace) -> int:
    loader = TextFileLoader(
        paths=args.paths,
        path_whitelist=args.path_whitelist or None,
    )
    docs = loader.load()
    if not docs:
        print("No documents found.")
        return 1

    chunker = _build_chunker(args)
    embedder = _build_embedder(args)
    store = _build_store(args, embedding_dim=embedder_dims(embedder))
    try:
        pipeline = SimplePipeline(
            chunker=chunker,
            embedder=embedder,
            store=store,
            embed_batch_size=args.embed_batch_size,
            use_embedding_cache=not args.disable_embed_cache,
        )
        if args.enable_fts:
            fts = SqliteFtsIndex(path=args.db)
            try:
                chunks = pipeline.upsert(docs, fts_index=fts)
            finally:
                fts.close()
        else:
            chunks = pipeline.upsert(docs)
    finally:
        if hasattr(store, "close"):
            store.close()

    print(f"Ingested {len(chunks)} chunks from {len(docs)} documents.")
    if args.store == "memory":
        print("Warning: memory store is not persisted across runs.")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    embedder = _build_embedder(args)
    store = _build_store(args, embedding_dim=embedder_dims(embedder))
    filters, range_filters = _parse_query_filters(args)
    try:
        embedding = embed_queries(embedder, [args.query])[0]
        results = store.query(
            embedding,
            args.top_k,
            filters=filters or None,
            range_filters=range_filters or None,
        )
    finally:
        if hasattr(store, "close"):
            store.close()
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
    fts.close()
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


def cmd_hybrid_query(args: argparse.Namespace) -> int:
    embedder = _build_embedder(args)
    store = _build_store(args, embedding_dim=embedder_dims(embedder))
    fts = SqliteFtsIndex(path=args.db)
    filters, range_filters = _parse_query_filters(args)
    try:
        retriever = HybridRetriever(
            embedder=embedder,
            vector_store=store,
            fts_index=fts,
            alpha=args.alpha,
            score_norm=args.score_norm,
        )
        results = retriever.retrieve_with_scores(
            query=args.query,
            top_k=args.top_k,
            vector_top_k=args.vector_top_k,
            fts_top_k=args.fts_top_k,
            filters=filters or None,
            range_filters=range_filters or None,
        )
    finally:
        fts.close()
        if hasattr(store, "close"):
            store.close()

    for idx, hit in enumerate(results, start=1):
        chunk = hit.chunk
        payload = {
            "rank": idx,
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "start": chunk.start,
            "end": chunk.end,
            "fused_score": hit.fused_score,
            "vector_score": hit.vector_score,
            "fts_score": hit.fts_score,
            "text": chunk.text,
        }
        print(json.dumps(payload, ensure_ascii=False))
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    store = _build_store(args)
    try:
        if args.enable_fts:
            fts = SqliteFtsIndex(path=args.db)
            try:
                fts_deleted = fts.delete_by_doc_ids(args.doc_id)
            finally:
                fts.close()
        else:
            fts_deleted = 0
        vector_deleted = store.delete_by_doc_ids(args.doc_id)
    finally:
        if hasattr(store, "close"):
            store.close()

    payload = {
        "doc_ids": args.doc_id,
        "vector_deleted": vector_deleted,
        "fts_deleted": fts_deleted,
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


def cmd_migrate_vec0_to_vec1(args: argparse.Namespace) -> int:
    migrated = migrate_sqlite_vec0_to_vec1(
        path=args.db,
        source_table=args.source_table,
        target_table=args.target_table,
        target_index_table=args.target_index_table,
        load_extension=not args.disable_sqlite_extension,
        extension_path=args.sqlite_extension_path,
        extension_whitelist=args.extension_whitelist or None,
    )
    payload = {
        "db": args.db,
        "source_table": args.source_table,
        "target_table": args.target_table,
        "target_index_table": args.target_index_table,
        "migrated_rows": migrated,
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


def cmd_migrate_sqlite_duckdb(args: argparse.Namespace) -> int:
    direction = args.direction
    if direction == "sqlite-to-duckdb":
        migrated = migrate_sqlite_vec1_to_duckdb_vss(
            sqlite_path=args.sqlite_db,
            duckdb_path=args.duckdb_db,
            source_table=args.sqlite_table,
            target_table=args.duckdb_table,
            enable_vss=not args.disable_vss_extension,
            persistent_index=args.vss_persistent_index,
        )
    else:
        migrated = migrate_duckdb_vss_to_sqlite_vec1(
            duckdb_path=args.duckdb_db,
            sqlite_path=args.sqlite_db,
            source_table=args.duckdb_table,
            target_table=args.sqlite_table,
            target_index_table=args.sqlite_index_table,
            load_extension=not args.disable_sqlite_extension,
            extension_path=args.sqlite_extension_path,
            extension_whitelist=args.extension_whitelist or None,
        )

    payload = {
        "direction": direction,
        "sqlite_db": args.sqlite_db,
        "duckdb_db": args.duckdb_db,
        "migrated_rows": migrated,
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    cases = load_benchmark_cases(args.dataset)
    if args.case_limit is not None:
        if args.case_limit <= 0:
            raise ValueError("case-limit must be positive")
        cases = cases[: args.case_limit]

    mode = args.mode
    filters: dict[str, object] | None = None
    range_filters: dict[str, tuple[float | int | None, float | int | None]] | None = None
    if mode in {"vector", "hybrid"}:
        parsed_filters, parsed_ranges = _parse_query_filters(args)
        filters = parsed_filters or None
        range_filters = parsed_ranges or None

    report: dict[str, object]
    if mode == "fts":
        fts = SqliteFtsIndex(path=args.db)
        try:
            report = evaluate_retrieval_benchmark(
                cases=cases,
                default_top_k=args.top_k,
                retrieve=lambda query, top_k: [
                    RetrievalItem(chunk_id=item.chunk_id, doc_id=item.doc_id or "")
                    for item in fts.query(query, top_k)
                ],
            )
        finally:
            fts.close()
    elif mode == "hybrid":
        embedder = _build_embedder(args)
        store = _build_store(args, embedding_dim=embedder_dims(embedder))
        fts = SqliteFtsIndex(path=args.db)
        try:
            retriever = HybridRetriever(
                embedder=embedder,
                vector_store=store,
                fts_index=fts,
                alpha=args.alpha,
                score_norm=args.score_norm,
            )
            report = evaluate_retrieval_benchmark(
                cases=cases,
                default_top_k=args.top_k,
                retrieve=lambda query, top_k: [
                    RetrievalItem(
                        chunk_id=hit.chunk.chunk_id,
                        doc_id=hit.chunk.doc_id,
                    )
                    for hit in retriever.retrieve_with_scores(
                        query=query,
                        top_k=top_k,
                        vector_top_k=args.vector_top_k,
                        fts_top_k=args.fts_top_k,
                        filters=filters,
                        range_filters=range_filters,
                    )
                ],
            )
        finally:
            fts.close()
            if hasattr(store, "close"):
                store.close()
    else:
        embedder = _build_embedder(args)
        store = _build_store(args, embedding_dim=embedder_dims(embedder))
        try:
            report = evaluate_retrieval_benchmark(
                cases=cases,
                default_top_k=args.top_k,
                retrieve=lambda query, top_k: [
                    RetrievalItem(chunk_id=chunk.chunk_id, doc_id=chunk.doc_id)
                    for chunk in store.query(
                        embed_queries(embedder, [query])[0],
                        top_k,
                        filters=filters,
                        range_filters=range_filters,
                    )
                ],
            )
        finally:
            if hasattr(store, "close"):
                store.close()

    payload = {
        "mode": mode,
        "dataset": args.dataset,
        "top_k": args.top_k,
        "case_count": len(cases),
        "report": report,
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    print(serialized)
    if args.output:
        Path(args.output).write_text(serialized + "\n", encoding="utf-8")
    return 0


def cmd_chat_ui(_args: argparse.Namespace) -> int:
    from .tk_chat_app import launch_tk_chat_app

    launch_tk_chat_app()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yfanrag")
    parser.add_argument("--log-level", default=None, help="Logging level, e.g. INFO/DEBUG")
    parser.add_argument(
        "--slow-query-ms",
        type=float,
        default=None,
        help="Slow-query warning threshold in milliseconds",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Ingest documents into the store")
    ingest.add_argument("paths", nargs="+", help="Paths to files or directories")
    ingest.add_argument(
        "--path-whitelist",
        action="append",
        default=[],
        help="Allowed root path for ingestion, repeatable",
    )
    ingest.add_argument("--db", default="yfanrag.db")
    ingest.add_argument("--store", choices=_STORE_CHOICES, default="sqlite-vec")
    ingest.add_argument("--chunker", choices=["fixed", "recursive", "structured"], default="structured")
    ingest.add_argument("--chunk-size", type=int, default=800)
    ingest.add_argument("--chunk-overlap", type=int, default=120)
    ingest.add_argument("--embedder", choices=["auto", "hashing", "fastembed", "http"], default="auto")
    ingest.add_argument("--dims", type=int, default=8)
    ingest.add_argument("--endpoint")
    ingest.add_argument("--model")
    ingest.add_argument("--api-key-env")
    ingest.add_argument("--embed-batch-size", type=int, default=64)
    ingest.add_argument("--disable-embed-cache", action="store_true")
    ingest.add_argument("--sqlite-extension-path", default=None)
    ingest.add_argument(
        "--extension-whitelist",
        action="append",
        default=[],
        help="Allowed sqlite extension root path, repeatable",
    )
    ingest.add_argument("--disable-sqlite-extension", action="store_true")
    ingest.add_argument("--disable-vss-extension", action="store_true")
    ingest.add_argument("--vss-persistent-index", action="store_true")
    ingest.add_argument("--enable-fts", action="store_true")
    ingest.add_argument("--distance-metric", choices=["l2", "cosine"], default=None)
    ingest.set_defaults(func=cmd_ingest)

    query = sub.add_parser("query", help="Vector search by query text")
    query.add_argument("query")
    query.add_argument("--db", default="yfanrag.db")
    query.add_argument("--store", choices=_STORE_CHOICES, default="sqlite-vec")
    query.add_argument("--embedder", choices=["auto", "hashing", "fastembed", "http"], default="auto")
    query.add_argument("--dims", type=int, default=8)
    query.add_argument("--endpoint")
    query.add_argument("--model")
    query.add_argument("--api-key-env")
    query.add_argument("--sqlite-extension-path", default=None)
    query.add_argument(
        "--extension-whitelist",
        action="append",
        default=[],
        help="Allowed sqlite extension root path, repeatable",
    )
    query.add_argument("--disable-sqlite-extension", action="store_true")
    query.add_argument("--disable-vss-extension", action="store_true")
    query.add_argument("--vss-persistent-index", action="store_true")
    query.add_argument("--top-k", type=int, default=5)
    query.add_argument("--distance-metric", choices=["l2", "cosine"], default=None)
    query.add_argument(
        "--filter",
        dest="filters",
        action="append",
        default=[],
        help="Field equality filter, format key=value; repeatable",
    )
    query.add_argument(
        "--range",
        dest="ranges",
        action="append",
        default=[],
        help="Numeric range filter, format key:min:max; min/max can be empty; repeatable",
    )
    query.set_defaults(func=cmd_query)

    fts_query = sub.add_parser("fts-query", help="FTS5 query by text")
    fts_query.add_argument("query")
    fts_query.add_argument("--db", default="yfanrag.db")
    fts_query.add_argument("--top-k", type=int, default=5)
    fts_query.set_defaults(func=cmd_fts_query)

    hybrid_query = sub.add_parser(
        "hybrid-query",
        help="Hybrid retrieval by vector + FTS score fusion",
    )
    hybrid_query.add_argument("query")
    hybrid_query.add_argument("--db", default="yfanrag.db")
    hybrid_query.add_argument(
        "--store",
        choices=_STORE_CHOICES,
        default="sqlite-vec",
    )
    hybrid_query.add_argument("--embedder", choices=["auto", "hashing", "fastembed", "http"], default="auto")
    hybrid_query.add_argument("--dims", type=int, default=8)
    hybrid_query.add_argument("--endpoint")
    hybrid_query.add_argument("--model")
    hybrid_query.add_argument("--api-key-env")
    hybrid_query.add_argument("--sqlite-extension-path", default=None)
    hybrid_query.add_argument(
        "--extension-whitelist",
        action="append",
        default=[],
        help="Allowed sqlite extension root path, repeatable",
    )
    hybrid_query.add_argument("--disable-sqlite-extension", action="store_true")
    hybrid_query.add_argument("--disable-vss-extension", action="store_true")
    hybrid_query.add_argument("--vss-persistent-index", action="store_true")
    hybrid_query.add_argument("--top-k", type=int, default=5)
    hybrid_query.add_argument("--vector-top-k", type=int, default=None)
    hybrid_query.add_argument("--fts-top-k", type=int, default=None)
    hybrid_query.add_argument("--alpha", type=float, default=0.5)
    hybrid_query.add_argument("--score-norm", choices=["sigmoid", "minmax", "none"], default="sigmoid")
    hybrid_query.add_argument("--distance-metric", choices=["l2", "cosine"], default=None)
    hybrid_query.add_argument(
        "--filter",
        dest="filters",
        action="append",
        default=[],
        help="Field equality filter, format key=value; repeatable",
    )
    hybrid_query.add_argument(
        "--range",
        dest="ranges",
        action="append",
        default=[],
        help="Numeric range filter, format key:min:max; min/max can be empty; repeatable",
    )
    hybrid_query.set_defaults(func=cmd_hybrid_query)

    delete = sub.add_parser("delete", help="Delete by document id")
    delete.add_argument("--db", default="yfanrag.db")
    delete.add_argument("--store", choices=_STORE_CHOICES, default="sqlite-vec")
    delete.add_argument("--sqlite-extension-path", default=None)
    delete.add_argument(
        "--extension-whitelist",
        action="append",
        default=[],
        help="Allowed sqlite extension root path, repeatable",
    )
    delete.add_argument("--disable-sqlite-extension", action="store_true")
    delete.add_argument("--disable-vss-extension", action="store_true")
    delete.add_argument("--vss-persistent-index", action="store_true")
    delete.add_argument("--doc-id", action="append", required=True)
    delete.add_argument("--enable-fts", action="store_true")
    delete.set_defaults(func=cmd_delete)

    migrate_vec = sub.add_parser(
        "migrate-vec0-to-vec1",
        help="Migrate sqlite-vec vec0 table rows into sqlite vec1 store",
    )
    migrate_vec.add_argument("--db", default="yfanrag.db")
    migrate_vec.add_argument("--source-table", default="vec_chunks")
    migrate_vec.add_argument("--target-table", default="vec1_chunks_data")
    migrate_vec.add_argument("--target-index-table", default="vec1_chunks_index")
    migrate_vec.add_argument("--sqlite-extension-path", default=None)
    migrate_vec.add_argument(
        "--extension-whitelist",
        action="append",
        default=[],
        help="Allowed sqlite extension root path, repeatable",
    )
    migrate_vec.add_argument("--disable-sqlite-extension", action="store_true")
    migrate_vec.set_defaults(func=cmd_migrate_vec0_to_vec1)

    migrate_cross = sub.add_parser(
        "migrate-sqlite-duckdb",
        help="Migrate vector data between sqlite(vec1) and duckdb(vss)",
    )
    migrate_cross.add_argument(
        "--direction",
        choices=["sqlite-to-duckdb", "duckdb-to-sqlite"],
        required=True,
    )
    migrate_cross.add_argument("--sqlite-db", default="yfanrag.db")
    migrate_cross.add_argument("--duckdb-db", default="yfanrag.duckdb")
    migrate_cross.add_argument("--sqlite-table", default="vec1_chunks_data")
    migrate_cross.add_argument("--sqlite-index-table", default="vec1_chunks_index")
    migrate_cross.add_argument("--duckdb-table", default="vss_chunks")
    migrate_cross.add_argument("--sqlite-extension-path", default=None)
    migrate_cross.add_argument(
        "--extension-whitelist",
        action="append",
        default=[],
        help="Allowed sqlite extension root path, repeatable",
    )
    migrate_cross.add_argument("--disable-sqlite-extension", action="store_true")
    migrate_cross.add_argument("--disable-vss-extension", action="store_true")
    migrate_cross.add_argument("--vss-persistent-index", action="store_true")
    migrate_cross.set_defaults(func=cmd_migrate_sqlite_duckdb)

    benchmark = sub.add_parser(
        "benchmark",
        help="Run retrieval quality and latency benchmark",
    )
    benchmark.add_argument("dataset", help="Path to benchmark json/jsonl file")
    benchmark.add_argument("--mode", choices=["vector", "fts", "hybrid"], default="vector")
    benchmark.add_argument("--db", default="yfanrag.db")
    benchmark.add_argument("--store", choices=_STORE_CHOICES, default="sqlite-vec")
    benchmark.add_argument("--embedder", choices=["auto", "hashing", "fastembed", "http"], default="auto")
    benchmark.add_argument("--dims", type=int, default=8)
    benchmark.add_argument("--endpoint")
    benchmark.add_argument("--model")
    benchmark.add_argument("--api-key-env")
    benchmark.add_argument("--sqlite-extension-path", default=None)
    benchmark.add_argument(
        "--extension-whitelist",
        action="append",
        default=[],
        help="Allowed sqlite extension root path, repeatable",
    )
    benchmark.add_argument("--disable-sqlite-extension", action="store_true")
    benchmark.add_argument("--disable-vss-extension", action="store_true")
    benchmark.add_argument("--vss-persistent-index", action="store_true")
    benchmark.add_argument("--top-k", type=int, default=5)
    benchmark.add_argument("--case-limit", type=int, default=None)
    benchmark.add_argument("--output", default=None)
    benchmark.add_argument("--alpha", type=float, default=0.5)
    benchmark.add_argument("--score-norm", choices=["sigmoid", "minmax", "none"], default="sigmoid")
    benchmark.add_argument("--vector-top-k", type=int, default=None)
    benchmark.add_argument("--fts-top-k", type=int, default=None)
    benchmark.add_argument("--distance-metric", choices=["l2", "cosine"], default=None)
    benchmark.add_argument(
        "--filter",
        dest="filters",
        action="append",
        default=[],
        help="Field equality filter, format key=value; repeatable",
    )
    benchmark.add_argument(
        "--range",
        dest="ranges",
        action="append",
        default=[],
        help="Numeric range filter, format key:min:max; min/max can be empty; repeatable",
    )
    benchmark.set_defaults(func=cmd_benchmark)

    chat_ui = sub.add_parser(
        "chat-ui",
        help="Launch Tkinter chat UI for real-model API conversation",
    )
    chat_ui.set_defaults(func=cmd_chat_ui)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.slow_query_ms is not None:
        os.environ["YFANRAG_SLOW_QUERY_MS"] = str(args.slow_query_ms)
    configure_logging(args.log_level)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
