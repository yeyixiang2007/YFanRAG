import json

import pytest

from yfanrag.benchmark import (
    BenchmarkCase,
    RetrievalItem,
    evaluate_retrieval_benchmark,
    load_benchmark_cases,
)


def test_load_benchmark_cases_from_jsonl(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"query": "q1", "expected_doc_ids": ["d1"]}),
                json.dumps({"query": "q2", "expected_chunk_ids": ["c2"], "top_k": 3}),
            ]
        ),
        encoding="utf-8",
    )

    cases = load_benchmark_cases(str(path))
    assert len(cases) == 2
    assert cases[0].query == "q1"
    assert cases[1].expected_chunk_ids == ["c2"]
    assert cases[1].top_k == 3


def test_load_benchmark_cases_requires_expected_ids(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([{"query": "q1"}]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_benchmark_cases(str(path))


def test_evaluate_retrieval_benchmark_doc_id_targets():
    cases = [
        BenchmarkCase(query="q1", expected_doc_ids=["d1"]),
        BenchmarkCase(query="q2", expected_doc_ids=["d2"]),
    ]
    mapping = {
        "q1": [RetrievalItem(chunk_id="c1", doc_id="d1")],
        "q2": [RetrievalItem(chunk_id="c2", doc_id="d3")],
    }

    report = evaluate_retrieval_benchmark(
        cases=cases,
        default_top_k=5,
        retrieve=lambda query, top_k: mapping[query][:top_k],
    )

    assert report["total_cases"] == 2
    assert report["hit_rate"] == 0.5
    assert report["mrr"] == 0.5
    assert report["recall"] == 0.5
    assert report["latency_ms"]["avg"] >= 0.0


def test_evaluate_retrieval_benchmark_chunk_id_targets():
    cases = [BenchmarkCase(query="q", expected_chunk_ids=["c2", "c3"])]
    report = evaluate_retrieval_benchmark(
        cases=cases,
        default_top_k=5,
        retrieve=lambda query, top_k: [
            RetrievalItem(chunk_id="c1", doc_id="d1"),
            RetrievalItem(chunk_id="c2", doc_id="d2"),
            RetrievalItem(chunk_id="c3", doc_id="d3"),
        ][:top_k],
    )

    assert report["hit_rate"] == 1.0
    assert report["mrr"] == 0.5
    assert report["recall"] == 1.0
