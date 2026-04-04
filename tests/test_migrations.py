import sqlite3
import struct

from yfanrag.migrations import migrate_sqlite_vec0_to_vec1
from yfanrag.vectorstores.sqlite_vec1 import SqliteVec1Store


def test_migrate_sqlite_vec0_to_vec1(tmp_path):
    db_path = tmp_path / "migrate.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE vec_chunks (chunk_id TEXT, doc_id TEXT, start INTEGER, end INTEGER, text TEXT, embedding BLOB)"
    )
    emb = struct.pack("<ffff", 1.0, 0.0, 0.0, 0.0)
    conn.execute(
        "INSERT INTO vec_chunks(chunk_id, doc_id, start, end, text, embedding) VALUES (?, ?, ?, ?, ?, ?)",
        ("c1", "d1", 0, 5, "hello", emb),
    )
    conn.commit()
    conn.close()

    count = migrate_sqlite_vec0_to_vec1(
        path=str(db_path),
        load_extension=False,
    )
    store = SqliteVec1Store(path=str(db_path), load_extension=False, embedding_dim=4)
    results = store.query([1.0, 0.0, 0.0, 0.0], top_k=1)
    store.close()

    assert count == 1
    assert results
    assert results[0].chunk_id == "c1"
