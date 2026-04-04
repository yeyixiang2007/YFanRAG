# Examples

All examples are runnable directly from repo root:

```powershell
python examples/01_basic_ingest_query.py
python examples/02_hybrid_query.py
python examples/03_benchmark.py
```

Notes:

- `01_basic_ingest_query.py` uses `InMemoryVectorStore` and requires no optional deps.
- `02_hybrid_query.py` uses SQLite + FTS + `SqliteVec1Store` in fallback mode (`load_extension=False`), so it can run without vec1 extension.
- `03_benchmark.py` demonstrates benchmark quality/latency report generation.
