"""Benchmark and evaluation helpers for retrieval quality and latency."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Callable, List, Sequence
import json
import math


@dataclass(frozen=True)
class BenchmarkCase:
    query: str
    expected_doc_ids: List[str] = field(default_factory=list)
    expected_chunk_ids: List[str] = field(default_factory=list)
    top_k: int | None = None

    def expected_target(self) -> tuple[str, list[str]]:
        if self.expected_chunk_ids:
            return ("chunk_id", list(dict.fromkeys(self.expected_chunk_ids)))
        return ("doc_id", list(dict.fromkeys(self.expected_doc_ids)))


@dataclass(frozen=True)
class RetrievalItem:
    chunk_id: str
    doc_id: str


def load_benchmark_cases(path: str) -> List[BenchmarkCase]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"benchmark file not found: {path}")

    if input_path.suffix.lower() == ".jsonl":
        rows = _load_jsonl(input_path)
    else:
        rows = _load_json(input_path)
    return [_parse_case(row) for row in rows]


def evaluate_retrieval_benchmark(
    cases: Sequence[BenchmarkCase],
    retrieve: Callable[[str, int], Sequence[RetrievalItem]],
    default_top_k: int = 5,
) -> dict[str, object]:
    if default_top_k <= 0:
        raise ValueError("default_top_k must be positive")
    if not cases:
        return {
            "total_cases": 0,
            "hit_rate": 0.0,
            "mrr": 0.0,
            "recall": 0.0,
            "latency_ms": {"avg": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0},
            "cases": [],
        }

    hit_count = 0
    mrr_sum = 0.0
    recall_sum = 0.0
    latencies: List[float] = []
    per_case: List[dict[str, object]] = []

    for case in cases:
        top_k = case.top_k or default_top_k
        if top_k <= 0:
            raise ValueError("case top_k must be positive")

        start = perf_counter()
        items = list(retrieve(case.query, top_k))
        latency_ms = (perf_counter() - start) * 1000.0
        latencies.append(latency_ms)

        target_field, expected_ids = case.expected_target()
        retrieved_ids = [
            item.chunk_id if target_field == "chunk_id" else item.doc_id for item in items
        ]
        expected_set = set(expected_ids)
        found_positions = [
            idx for idx, got in enumerate(retrieved_ids) if got in expected_set
        ]

        hit = bool(found_positions)
        if hit:
            hit_count += 1
            mrr_sum += 1.0 / (found_positions[0] + 1)

        matched = len({retrieved_ids[idx] for idx in found_positions})
        recall = matched / len(expected_set) if expected_set else 0.0
        recall_sum += recall

        per_case.append(
            {
                "query": case.query,
                "top_k": top_k,
                "target": target_field,
                "expected_ids": expected_ids,
                "retrieved_ids": retrieved_ids,
                "hit": hit,
                "reciprocal_rank": (1.0 / (found_positions[0] + 1)) if hit else 0.0,
                "recall": recall,
                "latency_ms": round(latency_ms, 3),
            }
        )

    total = len(cases)
    return {
        "total_cases": total,
        "hit_rate": hit_count / total,
        "mrr": mrr_sum / total,
        "recall": recall_sum / total,
        "latency_ms": {
            "avg": _avg(latencies),
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
            "max": max(latencies),
        },
        "cases": per_case,
    }


def _parse_case(data: dict[str, object]) -> BenchmarkCase:
    query = str(data.get("query", "")).strip()
    if not query:
        raise ValueError("benchmark case requires non-empty 'query'")

    raw_doc_ids = data.get("expected_doc_ids") or []
    raw_chunk_ids = data.get("expected_chunk_ids") or []
    expected_doc_ids = [str(item) for item in raw_doc_ids]
    expected_chunk_ids = [str(item) for item in raw_chunk_ids]
    if not expected_doc_ids and not expected_chunk_ids:
        raise ValueError(
            "benchmark case requires 'expected_doc_ids' or 'expected_chunk_ids'"
        )

    top_k = data.get("top_k")
    if top_k is not None:
        top_k = int(top_k)

    return BenchmarkCase(
        query=query,
        expected_doc_ids=expected_doc_ids,
        expected_chunk_ids=expected_chunk_ids,
        top_k=top_k,
    )


def _load_json(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("cases"), list):
        return payload["cases"]
    raise ValueError("benchmark json must be a list or object containing 'cases'")


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid jsonl at line {lineno}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"invalid jsonl row at line {lineno}: expected object")
        rows.append(row)
    return rows


def _avg(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _percentile(values: Sequence[float], p: int) -> float:
    if not values:
        return 0.0
    if p < 0 or p > 100:
        raise ValueError("percentile must be between 0 and 100")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (p / 100.0) * (len(ordered) - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight
