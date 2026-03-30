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

文档入库（使用 sqlite-vec）：  

```powershell
yfanrag ingest docs/ --db yfanrag.db --store sqlite-vec --enable-fts
```

向量检索：  

```powershell
yfanrag query \"hello\" --db yfanrag.db --store sqlite-vec --top-k 3
```

全文检索：  

```powershell
yfanrag fts-query \"hello\" --db yfanrag.db --top-k 3
```

**贡献**

欢迎提交 Issue 与 PR。建议先从 `docs/TECHNICAL.md` 的任务表中挑选任务。

**License**

待定
