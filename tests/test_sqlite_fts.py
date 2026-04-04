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
