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

项目处于规划与文档阶段，API 与实现以 `docs/TECHNICAL.md` 为准。

**目录结构（规划）**

- `docs/` 技术文档与设计说明
- `yfanrag/` 核心库
- `examples/` 示例
- `tests/` 测试

**路线图**

- SQLite `sqlite-vec` 端到端可用
- FTS 与混合检索
- `vec1` 与 DuckDB 适配层

**贡献**

欢迎提交 Issue 与 PR。建议先从 `docs/TECHNICAL.md` 的任务表中挑选任务。

**License**

待定
