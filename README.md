# YFanRAG

一个面向个人开发者与小项目的极简本地 RAG 库：文档向量化、分块与检索，底层封装 SQLite/DuckDB 扩展，不需要部署大型向量数据库。

**核心特点**

- 本地优先：SQLite 或 DuckDB 即可完成存储与检索。
- 极简上手：少量配置即可 ingest 与 query。
- 可扩展：分块、Embedding、检索器与存储层可插拔。

**适用场景**

- 小型知识库、个人笔记检索
- 离线或隐私敏感场景
- 需要轻量部署的内部工具

**当前状态**

已完成最小闭环与基础测试，API 与实现以 `docs/TECHNICAL.md` 为准。

**目录结构（规划）**

- `docs/` 技术文档与设计说明
- `yfanrag/` 核心库
- `examples/` 示例
- `tests/` 测试

**路线图**

- SQLite `sqlite-vec` 端到端可用
- FTS 与混合检索
- `vec1` 与 DuckDB 适配层

**开发与测试**

创建虚拟环境并安装开发依赖：

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -e .[dev]
```

运行测试：

```powershell
pytest
```

**CLI 使用**

安装可选的 SQLite 向量扩展：

```powershell
pip install -e .[sqlite]
```

安装可选 DuckDB 后端：

```powershell
pip install -e .[duckdb]
```

文档入库（使用 sqlite-vec）：  

```powershell
yfanrag ingest docs/ --db yfanrag.db --store sqlite-vec --enable-fts
```

文档入库（使用 sqlite-vec1 / duckdb-vss）：  

```powershell
yfanrag ingest docs/ --db yfanrag.db --store sqlite-vec1
yfanrag ingest docs/ --db yfanrag.duckdb --store duckdb-vss --vss-persistent-index
```

重复执行 `ingest` 时会按 `doc_id` 先删旧索引再写新索引（增量更新，不产生孤儿索引）。

批处理与缓存可通过参数调整：  

```powershell
yfanrag ingest docs/ --db yfanrag.db --store sqlite-vec --embed-batch-size 128
```

向量检索：  

```powershell
yfanrag query \"hello\" --db yfanrag.db --store sqlite-vec --top-k 3
```

字段与范围过滤（示例：按文档 + 起始偏移范围）：

```powershell
yfanrag query \"hello\" --db yfanrag.db --top-k 5 --filter "doc_id=file:docs/TECHNICAL.md" --range "start:0:2000"
```

全文检索：  

```powershell
yfanrag fts-query \"hello\" --db yfanrag.db --top-k 3
```

混合检索（向量 + FTS 融合）：  

```powershell
yfanrag hybrid-query \"hello\" --db yfanrag.db --top-k 3 --alpha 0.5 --filter "doc_id=file:docs/TECHNICAL.md"
```

按 `doc_id` 删除（含可选 FTS）：  

```powershell
yfanrag delete --db yfanrag.db --store sqlite-vec --doc-id "file:docs/TECHNICAL.md" --enable-fts
```

vec0 -> vec1 迁移：  

```powershell
yfanrag migrate-vec0-to-vec1 --db yfanrag.db --source-table vec_chunks
```

Benchmark（质量 + 性能报告）：  

```powershell
yfanrag benchmark benchmarks/cases.jsonl --db yfanrag.db --mode vector --top-k 5 --output benchmark_report.json
```

`cases.jsonl` 每行示例：

```json
{"query":"hello","expected_doc_ids":["file:docs/TECHNICAL.md"]}
```

**贡献**

欢迎提交 Issue 与 PR。建议先从 `docs/TECHNICAL.md` 的任务表中挑选任务。

**License**

待定
