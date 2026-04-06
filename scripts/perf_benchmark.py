#!/usr/bin/env python
"""Repository-local performance benchmark for YFanRAG."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import sys
from tempfile import TemporaryDirectory
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yfanrag.knowledge_base import KnowledgeBaseConfig, KnowledgeBaseManager
from yfanrag.loaders.text import TextFileLoader


DEFAULT_CORPUS_PATHS = ("README.md", "yfanrag", "tests", "examples")
DEFAULT_QUERY_SETS = {
    "vector": [
        "sqlite vec1 extension",
        "hybrid retriever score fusion",
        "slow query log threshold",
        "chunk overlap structured chunker",
        "secure extension whitelist",
        "duckdb vss migration",
        "benchmark latency p95",
        "tkinter chat ui markdown",
    ],
    "fts": [
        "sqlite-vec1",
        "duckdb-vss",
        "slow-query-ms",
        "extension whitelist",
        "hybrid-query",
        "benchmark",
        "chat-ui",
        "structured",
    ],
    "hybrid": [
        "sqlite vec1 benchmark latency",
        "duckdb migration vector store",
        "whitelist extension security",
        "structured chunker overlap",
        "hybrid query score fusion",
        "tkinter chat markdown rendering",
        "full text search vector",
        "benchmark report recall mrr",
    ],
}
PROFILE_DESCRIPTIONS = {
    "core": "Disable multi-query and reranker to measure raw retrieval latency.",
    "default": "Use project defaults, including multi-query expansion and reranking.",
}


def main() -> int:
    args = parse_args()
    report = run_benchmark(args)

    print(render_summary(report))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\njson report written to {output_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a reproducible local performance benchmark for YFanRAG",
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        default=list(DEFAULT_CORPUS_PATHS),
        help="corpus paths to ingest",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=["core", "default"],
        default=["core", "default"],
        help="query profiles to benchmark",
    )
    parser.add_argument("--repeat", type=int, default=5, help="measured repeat count")
    parser.add_argument("--warmup", type=int, default=1, help="warmup repeat count")
    parser.add_argument("--top-k", type=int, default=5, help="retrieval top-k")
    parser.add_argument("--store", choices=["sqlite-vec1", "sqlite-vec", "duckdb-vss"], default="sqlite-vec1")
    parser.add_argument("--dims", type=int, default=384, help="embedding dimensions")
    parser.add_argument(
        "--embedding-provider",
        choices=["auto", "hashing", "fastembed"],
        default="hashing",
        help="embedding backend",
    )
    parser.add_argument(
        "--chunker",
        choices=["structured", "recursive", "fixed"],
        default="structured",
        help="chunker implementation",
    )
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--disable-fts", action="store_true", help="disable FTS indexing and FTS/hybrid benchmarks")
    parser.add_argument(
        "--try-sqlite-extension",
        action="store_true",
        help="attempt to load the sqlite vec1 extension when store=sqlite-vec1",
    )
    parser.add_argument("--sqlite-extension-path", default=None, help="optional explicit vec1 extension path")
    parser.add_argument(
        "--enable-vss-extension",
        action="store_true",
        help="enable DuckDB VSS extension when store=duckdb-vss",
    )
    parser.add_argument(
        "--vss-persistent-index",
        action="store_true",
        help="request DuckDB persistent HNSW index when supported",
    )
    parser.add_argument("--output", default=None, help="optional JSON output path")
    return parser.parse_args()


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    if args.repeat <= 0:
        raise ValueError("--repeat must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    manager = KnowledgeBaseManager()
    loader = TextFileLoader(paths=args.paths)
    documents = loader.load()
    if not documents:
        raise ValueError("no supported documents found for benchmark corpus")

    available_modes = ["vector"]
    if not args.disable_fts and args.store != "duckdb-vss":
        available_modes.extend(["fts", "hybrid"])

    base_config = KnowledgeBaseConfig(
        db_path="benchmark.db",
        store=args.store,
        enable_fts=not args.disable_fts,
        dims=args.dims,
        embedding_provider=args.embedding_provider,
        chunker=args.chunker,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        disable_sqlite_extension=not args.try_sqlite_extension,
        sqlite_extension_path=args.sqlite_extension_path,
        disable_vss_extension=not args.enable_vss_extension,
        vss_persistent_index=args.vss_persistent_index,
    )

    with TemporaryDirectory() as tmpdir:
        ingest_samples: list[float] = []
        chunk_count = 0
        for index in range(args.repeat):
            config = replace(base_config, db_path=str(Path(tmpdir) / f"ingest_{index}.db"))
            start_ts = perf_counter()
            result = manager.ingest_paths(args.paths, config)
            ingest_samples.append((perf_counter() - start_ts) * 1000.0)
            chunk_count = result.chunk_count

        query_db_path = str(Path(tmpdir) / "query.db")
        query_seed_config = replace(base_config, db_path=query_db_path)
        manager.ingest_paths(args.paths, query_seed_config)

        profiles: dict[str, object] = {}
        for profile_name in args.profiles:
            profile_config = _profile_config(base_config, profile_name, db_path=query_db_path)
            profile_modes: dict[str, object] = {}
            for mode in available_modes:
                queries = list(DEFAULT_QUERY_SETS[mode])
                latencies = _measure_queries(
                    manager=manager,
                    config=profile_config,
                    mode=mode,
                    queries=queries,
                    top_k=args.top_k,
                    warmup=args.warmup,
                    repeat=args.repeat,
                )
                profile_modes[mode] = {
                    "description": PROFILE_DESCRIPTIONS[profile_name],
                    "query_count": len(queries),
                    "sample_count": len(latencies),
                    "latency_ms": _summarize(latencies),
                }
            profiles[profile_name] = profile_modes

    ingest_summary = _summarize(ingest_samples)
    doc_count = len(documents)
    total_bytes = sum(int(doc.metadata.get("size", 0)) for doc in documents)
    ingest_summary["docs_per_sec"] = round(doc_count / (ingest_summary["avg"] / 1000.0), 3)
    ingest_summary["chunks_per_sec"] = round(chunk_count / (ingest_summary["avg"] / 1000.0), 3)

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "corpus": {
            "paths": list(args.paths),
            "document_count": doc_count,
            "chunk_count": chunk_count,
            "total_bytes": total_bytes,
        },
        "config": {
            "store": args.store,
            "enable_fts": not args.disable_fts,
            "embedding_provider": args.embedding_provider,
            "dims": args.dims,
            "chunker": args.chunker,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "top_k": args.top_k,
            "repeat": args.repeat,
            "warmup": args.warmup,
            "try_sqlite_extension": args.try_sqlite_extension,
            "sqlite_extension_path": args.sqlite_extension_path,
            "enable_vss_extension": args.enable_vss_extension,
            "vss_persistent_index": args.vss_persistent_index,
        },
        "ingest": ingest_summary,
        "profiles": profiles,
    }


def _profile_config(
    base_config: KnowledgeBaseConfig,
    profile_name: str,
    *,
    db_path: str,
) -> KnowledgeBaseConfig:
    config = replace(base_config, db_path=db_path)
    if profile_name == "core":
        return replace(
            config,
            multi_query_enabled=False,
            reranker_enabled=False,
            context_compress_enabled=False,
        )
    if profile_name == "default":
        return config
    raise ValueError(f"unsupported profile: {profile_name}")


def _measure_queries(
    *,
    manager: KnowledgeBaseManager,
    config: KnowledgeBaseConfig,
    mode: str,
    queries: list[str],
    top_k: int,
    warmup: int,
    repeat: int,
) -> list[float]:
    latencies: list[float] = []
    for _ in range(warmup):
        for query in queries:
            manager.query(query, top_k=top_k, mode=mode, config=config)

    for _ in range(repeat):
        for query in queries:
            start_ts = perf_counter()
            manager.query(query, top_k=top_k, mode=mode, config=config)
            latencies.append((perf_counter() - start_ts) * 1000.0)
    return latencies


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "avg": round(sum(values) / len(values), 3),
        "p50": round(_percentile(values, 50), 3),
        "p95": round(_percentile(values, 95), 3),
        "max": round(max(values), 3),
    }


def _percentile(values: list[float], percentile: int) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (percentile / 100.0) * (len(ordered) - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def render_summary(report: dict[str, object]) -> str:
    corpus = report["corpus"]
    ingest = report["ingest"]
    profiles = report["profiles"]

    lines = [
        "YFanRAG local performance benchmark",
        f"timestamp (UTC): {report['timestamp_utc']}",
        f"platform: {report['environment']['platform']}",
        f"python: {report['environment']['python']}",
        (
            "corpus: "
            f"{corpus['document_count']} docs, "
            f"{corpus['chunk_count']} chunks, "
            f"{corpus['total_bytes']} bytes"
        ),
        (
            "ingest: "
            f"avg={ingest['avg']} ms, "
            f"p50={ingest['p50']} ms, "
            f"p95={ingest['p95']} ms, "
            f"max={ingest['max']} ms, "
            f"docs/s={ingest['docs_per_sec']}, "
            f"chunks/s={ingest['chunks_per_sec']}"
        ),
    ]

    for profile_name, modes in profiles.items():
        lines.append(f"\n[{profile_name}]")
        for mode_name, mode_report in modes.items():
            latency = mode_report["latency_ms"]
            lines.append(
                (
                    f"{mode_name}: avg={latency['avg']} ms, "
                    f"p50={latency['p50']} ms, "
                    f"p95={latency['p95']} ms, "
                    f"max={latency['max']} ms, "
                    f"samples={mode_report['sample_count']}"
                )
            )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
