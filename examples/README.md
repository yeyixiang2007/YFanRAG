# Examples

All examples are runnable directly from repo root:

```powershell
python examples/01_basic_ingest_query.py
python examples/02_hybrid_query.py
python examples/03_benchmark.py
python examples/04_tk_chat_app.py
```

Notes:

- `01_basic_ingest_query.py` uses `InMemoryVectorStore` and requires no optional deps.
- `02_hybrid_query.py` uses SQLite + FTS + `SqliteVec1Store` in fallback mode (`load_extension=False`), so it can run without vec1 extension.
- `03_benchmark.py` demonstrates benchmark quality/latency report generation.
- `04_tk_chat_app.py` launches a Codex/RooCode-style Tkinter chat UI that can call real model APIs (OpenAI-compatible, DeepSeek, Anthropic), and includes a built-in knowledge base manager window for ingest/query/delete.
