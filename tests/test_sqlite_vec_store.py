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
