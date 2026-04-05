from dataclasses import replace
from pathlib import Path

import pytest

from yfanrag.knowledge_base import KnowledgeBaseConfig, KnowledgeBaseManager


def _write_demo_docs(root: Path) -> None:
    (root / "alpha.md").write_text(
        "# Alpha\nSQLite and vector search are useful.\n",
        encoding="utf-8",
    )
    (root / "beta.txt").write_text(
        "DuckDB analytics and full text search can coexist.\n",
        encoding="utf-8",
    )


def _build_config(tmp_path: Path) -> KnowledgeBaseConfig:
    return KnowledgeBaseConfig(
        db_path=str(tmp_path / "kb.db"),
        store="sqlite-vec1",
        enable_fts=True,
        dims=16,
        chunker="fixed",
        chunk_size=64,
        chunk_overlap=8,
        disable_sqlite_extension=True,
        reranker_backend="heuristic",
        reranker_candidate_top_k=50,
    )


def test_knowledge_base_ingest_query_delete(tmp_path: Path) -> None:
    _write_demo_docs(tmp_path)
    manager = KnowledgeBaseManager()
    config = _build_config(tmp_path)

    ingest = manager.ingest_paths([str(tmp_path)], config)
    assert ingest.document_count == 2
    assert ingest.chunk_count > 0

    stats = manager.stats(config)
    assert stats.chunk_count >= ingest.chunk_count
    assert stats.doc_count == 2

    doc_ids = manager.list_doc_ids(config)
    assert len(doc_ids) == 2
    assert all(doc_id.startswith("file:") for doc_id in doc_ids)

    vector_hits = manager.query("vector search", top_k=3, mode="vector", config=config)
    assert vector_hits
    assert vector_hits[0].source == "vector"

    hybrid_hits = manager.query("sqlite", top_k=3, mode="hybrid", config=config)
    assert hybrid_hits
    assert hybrid_hits[0].source == "hybrid"

    fts_hits = manager.query("analytics", top_k=3, mode="fts", config=config)
    assert fts_hits
    assert fts_hits[0].source == "fts"

    deleted = manager.delete_doc_ids([doc_ids[0]], config)
    assert deleted.vector_deleted > 0
    assert deleted.fts_deleted > 0

    after = manager.stats(config)
    assert after.doc_count == 1
    assert len(manager.list_doc_ids(config)) == 1


def test_knowledge_base_invalid_mode(tmp_path: Path) -> None:
    manager = KnowledgeBaseManager()
    config = _build_config(tmp_path)
    with pytest.raises(ValueError):
        manager.query("hello", top_k=3, mode="unknown", config=config)


def test_knowledge_base_empty_paths(tmp_path: Path) -> None:
    manager = KnowledgeBaseManager()
    config = _build_config(tmp_path)
    with pytest.raises(ValueError):
        manager.ingest_paths([], config)


def test_knowledge_base_auto_routing_and_dynamic_params(tmp_path: Path) -> None:
    _write_demo_docs(tmp_path)
    manager = KnowledgeBaseManager()
    config = _build_config(tmp_path)
    manager.ingest_paths([str(tmp_path)], config)

    manager.query("`alpha.md` line 12 error", top_k=3, mode="auto", config=config)
    keyword_plan = manager.last_query_plan
    assert keyword_plan is not None
    assert keyword_plan.requested_mode == "auto"
    assert keyword_plan.resolved_mode == "fts"
    assert keyword_plan.query_type == "keyword"
    assert keyword_plan.fusion == "rrf"
    assert 3 <= len(keyword_plan.query_variants) <= 5
    assert keyword_plan.rrf_k is not None
    assert keyword_plan.candidate_top_k is not None
    assert keyword_plan.reranker_backend == "heuristic"
    assert keyword_plan.reranker_candidate_top_k == 50
    assert keyword_plan.reranker_top_k == 3

    manager.query(
        "请解释 vector search 与 full text search 在语义召回质量、精确匹配能力和延迟成本上的区别，"
        "并结合这个项目给出推荐使用策略与取舍建议？",
        top_k=3,
        mode="auto",
        config=config,
    )
    semantic_plan = manager.last_query_plan
    assert semantic_plan is not None
    assert semantic_plan.resolved_mode == "vector"
    assert semantic_plan.query_type == "semantic"
    assert semantic_plan.fusion == "rrf"
    assert 3 <= len(semantic_plan.query_variants) <= 5
    assert semantic_plan.reranker_backend == "heuristic"

    manager.query("sqlite vector search 区别", top_k=4, mode="auto", config=config)
    hybrid_plan = manager.last_query_plan
    assert hybrid_plan is not None
    assert hybrid_plan.resolved_mode == "hybrid"
    assert hybrid_plan.alpha is not None
    assert hybrid_plan.vector_top_k is not None
    assert hybrid_plan.fts_top_k is not None
    assert hybrid_plan.vector_top_k >= 4
    assert hybrid_plan.fts_top_k >= 4
    assert hybrid_plan.fusion == "rrf"
    assert 3 <= len(hybrid_plan.query_variants) <= 5
    assert hybrid_plan.reranker_backend == "heuristic"

    no_fts_config = replace(config, enable_fts=False)
    manager.query("sqlite keyword", top_k=3, mode="auto", config=no_fts_config)
    fallback_plan = manager.last_query_plan
    assert fallback_plan is not None
    assert fallback_plan.resolved_mode == "vector"
    assert fallback_plan.query_type == "fts-unavailable"
    assert fallback_plan.fusion == "rrf"
    assert fallback_plan.reranker_backend == "heuristic"


def test_knowledge_base_multi_query_rrf_hits(tmp_path: Path) -> None:
    _write_demo_docs(tmp_path)
    manager = KnowledgeBaseManager()
    config = _build_config(tmp_path)
    manager.ingest_paths([str(tmp_path)], config)

    hits = manager.query(
        "sqlite vector search 的区别是什么",
        top_k=3,
        mode="hybrid",
        config=config,
    )
    assert hits
    assert all(hit.rrf_score is not None for hit in hits)
    assert all(hit.rerank_score is not None for hit in hits)
    assert hits[0].source == "hybrid"

    plan = manager.last_query_plan
    assert plan is not None
    assert plan.requested_mode == "hybrid"
    assert plan.fusion == "rrf"
    assert 3 <= len(plan.query_variants) <= 5
    assert plan.candidate_top_k is not None
    assert plan.reranker_backend == "heuristic"
    assert plan.reranker_candidate_top_k == 50
    assert plan.reranker_top_k == 3


def test_knowledge_base_disable_multi_query(tmp_path: Path) -> None:
    _write_demo_docs(tmp_path)
    manager = KnowledgeBaseManager()
    config = replace(_build_config(tmp_path), multi_query_enabled=False)
    manager.ingest_paths([str(tmp_path)], config)

    hits = manager.query("vector search", top_k=3, mode="vector", config=config)
    assert hits
    plan = manager.last_query_plan
    assert plan is not None
    assert plan.fusion is None
    assert len(plan.query_variants) == 1
    assert plan.reranker_backend == "heuristic"


def test_knowledge_base_disable_reranker(tmp_path: Path) -> None:
    _write_demo_docs(tmp_path)
    manager = KnowledgeBaseManager()
    config = replace(
        _build_config(tmp_path),
        reranker_enabled=False,
    )
    manager.ingest_paths([str(tmp_path)], config)
    hits = manager.query("vector search", top_k=3, mode="vector", config=config)

    assert hits
    assert all(hit.rerank_score is None for hit in hits)
    plan = manager.last_query_plan
    assert plan is not None
    assert plan.reranker_backend is None
