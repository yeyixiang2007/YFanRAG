# Performance

English | [简体中文](performance.zh-CN.md)

YFanRAG currently exposes two different kinds of evaluation:

| Type | Entry point | Focus |
| --- | --- | --- |
| Retrieval quality benchmark | `yfanrag benchmark` | `hit_rate`, `mrr`, `recall`, query latency |
| Local performance benchmark | `scripts/perf_benchmark.py` | Ingest cost and end-to-end latency across retrieval modes |

## Local Performance Benchmark Flow

```mermaid
flowchart LR
  A["Choose corpus"] --> B["Repeat ingest with the selected config"]
  B --> C["Build query database"]
  C --> D["Warm up per profile"]
  D --> E["Repeat vector / fts / hybrid queries"]
  E --> F["Aggregate avg / p50 / p95 / max"]
  F --> G["Print summary + write JSON report"]
```

## How to Run

```powershell
.\.venv\Scripts\python scripts\perf_benchmark.py --repeat 5 --warmup 1 --output perf-report.json
```

Default configuration:

- Corpus: `README.md`, `yfanrag/`, `tests/`, `examples/`
- Store: `sqlite-vec1`
- Embedding: `hashing`, `384 dims`
- Chunker: `structured`
- `top_k=5`

Profile summary:

- `core`: disable multi-query, reranker, and context compression to measure the raw retrieval path
- `default`: keep the project defaults to measure the real cost of the enhanced retrieval chain

## Baseline on 2026-04-06

### Test Environment

| Item | Value |
| --- | --- |
| OS | Windows 11 |
| Python | 3.13.5 |
| CPU | AMD Ryzen 9 8945HX |
| Memory | 32 GB |

### Corpus Size

| Item | Value |
| --- | --- |
| Documents | `67` |
| Chunks | `829` |
| Raw text size | `436,183 bytes` |

### Ingest Performance

| Metric | avg | p50 | p95 | max | Throughput |
| --- | --- | --- | --- | --- | --- |
| ingest | `277.465 ms` | `278.918 ms` | `279.832 ms` | `279.843 ms` | `241.472 docs/s` / `2987.764 chunks/s` |

### Query Latency

Each mode uses `8` queries with `warmup=1` and `repeat=5`, for `40` total measured samples.

| Profile | Mode | avg | p50 | p95 | max |
| --- | --- | --- | --- | --- | --- |
| `core` | `vector` | `38.412 ms` | `33.467 ms` | `54.006 ms` | `54.359 ms` |
| `core` | `fts` | `2.826 ms` | `2.854 ms` | `3.171 ms` | `3.299 ms` |
| `core` | `hybrid` | `51.085 ms` | `54.916 ms` | `57.631 ms` | `58.601 ms` |
| `default` | `vector` | `201.299 ms` | `206.654 ms` | `218.185 ms` | `224.171 ms` |
| `default` | `fts` | `12.461 ms` | `12.902 ms` | `15.960 ms` | `16.200 ms` |
| `default` | `hybrid` | `213.487 ms` | `215.258 ms` | `231.182 ms` | `233.802 ms` |

## Interpretation

- Rebuilding the index takes about `0.28s` at the current repository scale, which is still suitable for local development loops
- `FTS` is the fastest path in this environment
- `vector / hybrid` are noticeably slower because this run did not enable the `vec1` extension, so `sqlite-vec1` fell back to SQLite + Python exact scanning
- With multi-query and reranking enabled, latency moves into roughly the `200-215 ms avg` range with about `230 ms p95`, which is the real cost of the enhanced retrieval path

## Tuning Suggestions

Highest-priority directions:

1. Enable the `vec1` extension in the target environment and rerun the same benchmark
2. Compare `hashing` and `fastembed` when you need better semantic recall
3. Save a dedicated `perf-report.json` for real production-like corpora rather than relying only on repository source files
4. Separate offline ingest cost from online query cost instead of looking only at a single average number

## Quality Benchmark

If you care more about retrieval quality than raw throughput, use:

```powershell
yfanrag benchmark benchmarks/cases.jsonl --db yfanrag.db --mode hybrid --output report.json
```

The output includes:

- `hit_rate`
- `mrr`
- `recall`
- `latency_ms`
- `cases`

## Related Docs

- [CLI Guide](cli.md)
- [Architecture](architecture.md)
- [TECHNICAL.md](TECHNICAL.md)
