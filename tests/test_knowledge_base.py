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

