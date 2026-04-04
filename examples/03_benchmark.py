"""Example 3: run retrieval benchmark and print quality/latency report."""

from __future__ import annotations

from yfanrag.benchmark import BenchmarkCase, RetrievalItem, evaluate_retrieval_benchmark


def main() -> None:
    cases = [
        BenchmarkCase(query="hello", expected_doc_ids=["doc-1"]),
        BenchmarkCase(query="world", expected_doc_ids=["doc-2"]),
    ]
    retrieval_map = {
        "hello": [RetrievalItem(chunk_id="c1", doc_id="doc-1")],
        "world": [RetrievalItem(chunk_id="c2", doc_id="doc-x")],
    }

    report = evaluate_retrieval_benchmark(
        cases=cases,
        default_top_k=3,
        retrieve=lambda query, top_k: retrieval_map[query][:top_k],
    )
    print("Benchmark report:")
    print(report)


if __name__ == "__main__":
    main()
