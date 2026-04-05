import pytest

from yfanrag.fts import SqliteFtsIndex
from yfanrag.models import Chunk


def test_sqlite_fts_index(tmp_path):
    db_path = tmp_path / "fts.db"
    try:
        fts = SqliteFtsIndex(path=str(db_path))
    except RuntimeError as exc:
        pytest.skip(str(exc))

    chunks = [
        Chunk(chunk_id="c1", doc_id="d1", text="hello world", start=0, end=11),
        Chunk(chunk_id="c2", doc_id="d1", text="another doc", start=12, end=23),
    ]
    fts.add(chunks)
    results = fts.query("hello", top_k=1)
    fts.close()

    assert results
    assert results[0].chunk_id == "c1"


def test_sqlite_fts_delete_by_doc_ids(tmp_path):
    db_path = tmp_path / "fts_delete.db"
    try:
        fts = SqliteFtsIndex(path=str(db_path))
    except RuntimeError as exc:
        pytest.skip(str(exc))

    chunks = [
        Chunk(chunk_id="c1", doc_id="d1", text="hello world", start=0, end=11),
        Chunk(chunk_id="c2", doc_id="d2", text="hello vector", start=0, end=12),
    ]
    fts.add(chunks)
    deleted = fts.delete_by_doc_ids(["d1"])
    remaining = fts.query("hello", top_k=10)
    fts.close()

    assert deleted == 1
    assert [match.chunk_id for match in remaining] == ["c2"]


def test_sqlite_fts_query_with_special_symbols(tmp_path):
    db_path = tmp_path / "fts_symbols.db"
    try:
        fts = SqliteFtsIndex(path=str(db_path))
    except RuntimeError as exc:
        pytest.skip(str(exc))

    chunks = [
        Chunk(
            chunk_id="c1",
            doc_id="d1",
            text="alpha md line 12 error context",
            start=0,
            end=29,
        ),
    ]
    fts.add(chunks)
    results = fts.query("`alpha.md` line 12 error", top_k=3)
    fts.close()

    assert results
    assert results[0].chunk_id == "c1"


def test_sqlite_fts_query_with_cjk_text(tmp_path):
    db_path = tmp_path / "fts_cjk.db"
    try:
        fts = SqliteFtsIndex(path=str(db_path))
    except RuntimeError as exc:
        pytest.skip(str(exc))

    fts.add(
        [
            Chunk(
                chunk_id="c1",
                doc_id="d1",
                text="这是一个中文检索测试案例",
                start=0,
                end=12,
            )
        ]
    )
    results = fts.query("中文检索", top_k=3)
    fts.close()

    assert results
    assert results[0].chunk_id == "c1"


def test_sqlite_fts_query_with_code_symbols(tmp_path):
    db_path = tmp_path / "fts_code_symbols.db"
    try:
        fts = SqliteFtsIndex(path=str(db_path))
    except RuntimeError as exc:
        pytest.skip(str(exc))

    fts.add(
        [
            Chunk(
                chunk_id="c1",
                doc_id="d1",
                text="Use C++ with React.js in this sample project.",
                start=0,
                end=45,
            )
        ]
    )
    results = fts.query("C++ React.js", top_k=3)
    fts.close()

    assert results
    assert results[0].chunk_id == "c1"


def test_sqlite_fts_delete_by_doc_ids_batches_large_input(tmp_path):
    db_path = tmp_path / "fts_batch_delete.db"
    try:
        fts = SqliteFtsIndex(path=str(db_path))
    except RuntimeError as exc:
        pytest.skip(str(exc))

    chunks = [
        Chunk(
            chunk_id=f"c{i}",
            doc_id=f"d{i}",
            text=f"document {i}",
            start=0,
            end=10,
        )
        for i in range(1100)
    ]
    fts.add(chunks)
    deleted = fts.delete_by_doc_ids([chunk.doc_id for chunk in chunks])
    remaining = fts.query("document", top_k=5)
    fts.close()

    assert deleted == 1100
    assert remaining == []


def test_sqlite_fts_rejects_unsafe_table_names(tmp_path):
    with pytest.raises(ValueError):
        SqliteFtsIndex(path=str(tmp_path / "bad.db"), table="fts; DROP TABLE x;")
