import pytest

from yfanrag.migrations import (
    migrate_duckdb_vss_to_sqlite_vec1,
    migrate_sqlite_vec1_to_duckdb_vss,
)
from yfanrag.models import Chunk
from yfanrag.vectorstores.duckdb_vss import DuckDbVssStore
from yfanrag.vectorstores.sqlite_vec1 import SqliteVec1Store

duckdb = pytest.importorskip("duckdb")


def test_migrate_sqlite_vec1_and_duckdb_vss_roundtrip(tmp_path):
    sqlite_path = tmp_path / "a.db"
    duckdb_path = tmp_path / "b.duckdb"

    src_store = SqliteVec1Store(
        path=str(sqlite_path),
        embedding_dim=4,
        load_extension=False,
    )
    src_store.add(
        [Chunk(chunk_id="c1", doc_id="d1", text="hello", start=0, end=5)],
        [[1.0, 0.0, 0.0, 0.0]],
    )
    src_store.close()

    migrated1 = migrate_sqlite_vec1_to_duckdb_vss(
        sqlite_path=str(sqlite_path),
        duckdb_path=str(duckdb_path),
        enable_vss=False,
    )
    assert migrated1 == 1

    mid_store = DuckDbVssStore(
        path=str(duckdb_path),
        embedding_dim=4,
        enable_vss=False,
    )
    mid_results = mid_store.query([1.0, 0.0, 0.0, 0.0], top_k=1)
    mid_store.close()
    assert mid_results
    assert mid_results[0].chunk_id == "c1"

    migrated2 = migrate_duckdb_vss_to_sqlite_vec1(
        duckdb_path=str(duckdb_path),
        sqlite_path=str(sqlite_path),
        target_table="vec1_chunks_data_copy",
        target_index_table="vec1_chunks_index_copy",
        load_extension=False,
    )
    assert migrated2 == 1

    dst_store = SqliteVec1Store(
        path=str(sqlite_path),
        table="vec1_chunks_data_copy",
        index_table="vec1_chunks_index_copy",
        embedding_dim=4,
        load_extension=False,
    )
    dst_results = dst_store.query([1.0, 0.0, 0.0, 0.0], top_k=1)
    dst_store.close()
    assert dst_results
    assert dst_results[0].chunk_id == "c1"
