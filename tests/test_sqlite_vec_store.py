import sqlite3

import pytest

from yfanrag.models import Chunk
from yfanrag.vectorstores.sqlite_vec import SqliteVecStore

sqlite_vec = pytest.importorskip("sqlite_vec")


@pytest.mark.skipif(
    not hasattr(sqlite3.connect(":memory:"), "enable_load_extension"),
    reason="SQLite extension loading is not supported",
)
def test_sqlite_vec_store_roundtrip(tmp_path):
    db_path = tmp_path / "vec.db"
    store = SqliteVecStore(path=str(db_path), embedding_dim=4)

    chunks = [
        Chunk(chunk_id="c1", doc_id="d1", text="hello", start=0, end=5),
        Chunk(chunk_id="c2", doc_id="d1", text="world", start=6, end=11),
    ]
    embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
    store.add(chunks, embeddings)

    results = store.query([1.0, 0.0, 0.0, 0.0], top_k=1)
    store.close()

    assert results
    assert results[0].chunk_id == "c1"


@pytest.mark.skipif(
    not hasattr(sqlite3.connect(":memory:"), "enable_load_extension"),
    reason="SQLite extension loading is not supported",
)
def test_sqlite_vec_store_query_with_filters(tmp_path):
    db_path = tmp_path / "vec_filter.db"
    store = SqliteVecStore(path=str(db_path), embedding_dim=4)

    chunks = [
        Chunk(chunk_id="c1", doc_id="d1", text="hello", start=0, end=5, metadata={"index": 0}),
        Chunk(chunk_id="c2", doc_id="d2", text="hello", start=20, end=25, metadata={"index": 1}),
    ]
    embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ]
    store.add(chunks, embeddings)

    by_doc = store.query([1.0, 0.0, 0.0, 0.0], top_k=5, filters={"doc_id": "d2"})
    by_range = store.query(
        [1.0, 0.0, 0.0, 0.0],
        top_k=5,
        range_filters={"start": (10, None)},
    )
    by_index = store.query([1.0, 0.0, 0.0, 0.0], top_k=5, filters={"index": 1})
    store.close()

    assert [chunk.chunk_id for chunk in by_doc] == ["c2"]
    assert [chunk.chunk_id for chunk in by_range] == ["c2"]
    assert [chunk.chunk_id for chunk in by_index] == ["c2"]


@pytest.mark.skipif(
    not hasattr(sqlite3.connect(":memory:"), "enable_load_extension"),
    reason="SQLite extension loading is not supported",
)
def test_sqlite_vec_store_delete_by_doc_ids(tmp_path):
    db_path = tmp_path / "vec_delete.db"
    store = SqliteVecStore(path=str(db_path), embedding_dim=4)
    chunks = [
        Chunk(chunk_id="c1", doc_id="d1", text="hello", start=0, end=5),
        Chunk(chunk_id="c2", doc_id="d2", text="world", start=6, end=11),
    ]
    embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
    store.add(chunks, embeddings)

    deleted = store.delete_by_doc_ids(["d1"])
    results = store.query([1.0, 0.0, 0.0, 0.0], top_k=5)
    store.close()

    assert deleted == 1
    assert all(chunk.doc_id != "d1" for chunk in results)
