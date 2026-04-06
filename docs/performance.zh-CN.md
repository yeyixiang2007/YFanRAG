# 性能测试

[English](performance.md) | 简体中文

YFanRAG 里有两类不同目标的评测能力：

| 类型 | 入口 | 关注点 |
| --- | --- | --- |
| 检索质量 Benchmark | `yfanrag benchmark` | `hit_rate`、`mrr`、`recall`、查询延迟 |
| 本地性能 Benchmark | `scripts/perf_benchmark.py` | 入库耗时、不同检索模式的端到端延迟 |

## 本地性能 Benchmark 流程

```mermaid
flowchart LR
  A["选择语料"] --> B["按配置重复 ingest"]
  B --> C["构建 query 数据库"]
  C --> D["按 profile 预热"]
  D --> E["重复执行 vector / fts / hybrid 查询"]
  E --> F["汇总 avg / p50 / p95 / max"]
  F --> G["输出控制台摘要 + JSON 报告"]
```

## 如何运行

```powershell
.\.venv\Scripts\python scripts\perf_benchmark.py --repeat 5 --warmup 1 --output perf-report.json
```

默认配置：

- 语料：`README.md`、`yfanrag/`、`tests/`、`examples/`
- Store：`sqlite-vec1`
- Embedding：`hashing`，`384 dims`
- Chunker：`structured`
- `top_k=5`

Profile 说明：

- `core`：关闭 multi-query、reranker、上下文压缩，测原始检索链路
- `default`：保留项目默认增强链路，测真实体验成本

## 2026-04-06 本机基线

### 测试环境

| 项目 | 值 |
| --- | --- |
| 系统 | Windows 11 |
| Python | 3.13.5 |
| CPU | AMD Ryzen 9 8945HX |
| 内存 | 32 GB |

### 语料规模

| 项目 | 值 |
| --- | --- |
| 文档数 | `67` |
| Chunk 数 | `829` |
| 原始文本体积 | `436,183 bytes` |

### 入库性能

| 指标 | avg | p50 | p95 | max | 吞吐 |
| --- | --- | --- | --- | --- | --- |
| ingest | `277.465 ms` | `278.918 ms` | `279.832 ms` | `279.843 ms` | `241.472 docs/s` / `2987.764 chunks/s` |

### 查询延迟

每种模式 `8` 条查询，`warmup=1`，`repeat=5`，共 `40` 个样本。

| Profile | Mode | avg | p50 | p95 | max |
| --- | --- | --- | --- | --- | --- |
| `core` | `vector` | `38.412 ms` | `33.467 ms` | `54.006 ms` | `54.359 ms` |
| `core` | `fts` | `2.826 ms` | `2.854 ms` | `3.171 ms` | `3.299 ms` |
| `core` | `hybrid` | `51.085 ms` | `54.916 ms` | `57.631 ms` | `58.601 ms` |
| `default` | `vector` | `201.299 ms` | `206.654 ms` | `218.185 ms` | `224.171 ms` |
| `default` | `fts` | `12.461 ms` | `12.902 ms` | `15.960 ms` | `16.200 ms` |
| `default` | `hybrid` | `213.487 ms` | `215.258 ms` | `231.182 ms` | `233.802 ms` |

## 结果解读

- 当前仓库规模下，重建索引约 `0.28s`，仍适合本地开发中的频繁重建
- `FTS` 是当前环境中最快的路径
- `vector / hybrid` 明显慢于 `FTS`，主要原因是这次测试没有启用 `vec1` 扩展，`sqlite-vec1` 回退到了 SQLite + Python 精确扫描
- 开启 multi-query 与 reranker 后，查询延迟进入大约 `200-215 ms avg`、`230 ms p95` 的区间，这是召回增强换来的真实成本

## 调优建议

优先级最高的方向：

1. 在目标环境中启用 `vec1` 扩展，再重新跑同一套基准
2. 需要更强语义召回时，比较 `hashing` 与 `fastembed` 的效果与成本
3. 为真实语料单独保存一份 `perf-report.json`，避免只用仓库源码做基准
4. 区分离线 ingest 成本与在线 query 成本，不要只看单个平均值

## 质量 Benchmark

如果你更关心召回质量而不是纯性能，使用：

```powershell
yfanrag benchmark benchmarks/cases.jsonl --db yfanrag.db --mode hybrid --output report.json
```

输出包含：

- `hit_rate`
- `mrr`
- `recall`
- `latency_ms`
- `cases`

## 相关文档

- [CLI 指南](cli.zh-CN.md)
- [架构设计](architecture.zh-CN.md)
- [TECHNICAL.zh-CN.md](TECHNICAL.zh-CN.md)
