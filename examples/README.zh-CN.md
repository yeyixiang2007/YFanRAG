# Examples

[English](README.md) | 简体中文

所有示例都可以直接从仓库根目录运行：

```powershell
python examples/01_basic_ingest_query.py
python examples/02_hybrid_query.py
python examples/03_benchmark.py
python examples/04_tk_chat_app.py
```

说明：

- `01_basic_ingest_query.py` 使用 `InMemoryVectorStore`，不依赖任何可选包。
- `02_hybrid_query.py` 使用 SQLite + FTS + `SqliteVec1Store` 的 fallback 模式（`load_extension=False`），因此在没有 vec1 扩展时也能运行。
- `03_benchmark.py` 演示检索质量与延迟报告生成。
- `04_tk_chat_app.py` 启动一个 Codex/RooCode 风格的 Tkinter 聊天界面，可连接真实模型 API（OpenAI-compatible、DeepSeek、Anthropic），并内置知识库管理窗口用于 ingest/query/delete。
