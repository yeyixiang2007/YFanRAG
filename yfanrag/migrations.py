"""Migration helpers across storage backends."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import List
import sqlite3
import struct

from .models import Chunk
from .sql_utils import connect_sqlite, validate_identifier
from .vectorstores.duckdb_vss import DuckDbVssStore
from .vectorstores.sqlite_vec1 import SqliteVec1Store

try:
    import duckdb
except ImportError:  # pragma: no cover - optional dependency
    duckdb = None

_MIGRATION_BATCH_SIZE = 256


def migrate_sqlite_vec0_to_vec1(
    path: str,
    source_table: str = "vec_chunks",
    target_table: str = "vec1_chunks_data",
    target_index_table: str = "vec1_chunks_index",
    load_extension: bool = True,
    extension_path: str | None = None,
    extension_whitelist: List[str] | None = None,
) -> int:
    """Migrate rows from sqlite-vec vec0 table into vec1 adapter tables."""
    source_table = validate_identifier(source_table, label="source table")
    target_table = validate_identifier(target_table, label="target table")
    target_index_table = validate_identifier(target_index_table, label="target index table")

    store: SqliteVec1Store | None = None
    migrated = 0
    conn = connect_sqlite(path)
    try:
        for rows in _iter_sqlite_rows(
            conn,
            f"SELECT chunk_id, doc_id, start, end, text, embedding FROM {source_table}",
        ):
            chunks: List[Chunk] = []
            embeddings: List[List[float]] = []
            for row in rows:
                vector = _deserialize_float32(row["embedding"])
                chunks.append(
                    Chunk(
                        chunk_id=row["chunk_id"],
                        doc_id=row["doc_id"],
                        text=row["text"],
                        start=row["start"],
                        end=row["end"],
                    )
                )
                embeddings.append(vector)
            if not chunks:
                continue
            if store is None:
                store = SqliteVec1Store(
                    path=path,
                    table=target_table,
                    index_table=target_index_table,
                    embedding_dim=len(embeddings[0]),
                    load_extension=load_extension,
                    extension_path=extension_path,
                    extension_whitelist=extension_whitelist,
                )
            store.add(chunks, embeddings)
            migrated += len(chunks)
    finally:
        conn.close()
        if store is not None:
            store.close()
    return migrated


def migrate_sqlite_vec1_to_duckdb_vss(
    sqlite_path: str,
    duckdb_path: str,
    source_table: str = "vec1_chunks_data",
    target_table: str = "vss_chunks",
    enable_vss: bool = True,
    persistent_index: bool = False,
) -> int:
    """Migrate sqlite vec1-table rows to DuckDB VSS table."""
    source_table = validate_identifier(source_table, label="source table")
    target_table = validate_identifier(target_table, label="target table")

    store: DuckDbVssStore | None = None
    migrated = 0
    conn = connect_sqlite(sqlite_path)
    try:
        for rows in _iter_sqlite_rows(
            conn,
            f"SELECT chunk_id, doc_id, start, end_pos, meta_index, text, embedding FROM {source_table}",
        ):
            chunks: List[Chunk] = []
            embeddings: List[List[float]] = []
            for row in rows:
                chunks.append(
                    Chunk(
                        chunk_id=row["chunk_id"],
                        doc_id=row["doc_id"],
                        text=row["text"],
                        start=row["start"],
                        end=row["end_pos"],
                        metadata={"index": row["meta_index"]} if row["meta_index"] is not None else {},
                    )
                )
                embeddings.append(_deserialize_float32(row["embedding"]))
            if not chunks:
                continue
            if store is None:
                store = DuckDbVssStore(
                    path=duckdb_path,
                    table=target_table,
                    embedding_dim=len(embeddings[0]),
                    enable_vss=enable_vss,
                    persistent_index=persistent_index,
                    fail_if_no_vss=False,
                )
            store.add(chunks, embeddings)
            migrated += len(chunks)
    finally:
        conn.close()
        if store is not None:
            store.close()
    return migrated


def migrate_duckdb_vss_to_sqlite_vec1(
    duckdb_path: str,
    sqlite_path: str,
    source_table: str = "vss_chunks",
    target_table: str = "vec1_chunks_data",
    target_index_table: str = "vec1_chunks_index",
    load_extension: bool = True,
    extension_path: str | None = None,
    extension_whitelist: List[str] | None = None,
) -> int:
    """Migrate DuckDB VSS table rows to sqlite vec1 adapter tables."""
    if duckdb is None:
        raise RuntimeError("duckdb is not installed. Install with `pip install duckdb`.")

    source_table = validate_identifier(source_table, label="source table")
    target_table = validate_identifier(target_table, label="target table")
    target_index_table = validate_identifier(target_index_table, label="target index table")

    store: SqliteVec1Store | None = None
    migrated = 0
    conn = duckdb.connect(duckdb_path)
    try:
        for rows in _iter_duckdb_rows(
            conn,
            f"SELECT chunk_id, doc_id, start_pos, end_pos, meta_index, text, embedding FROM {source_table}",
        ):
            chunks: List[Chunk] = []
            embeddings: List[List[float]] = []
            for row in rows:
                chunk_id, doc_id, start_pos, end_pos, meta_index, text, embedding = row
                chunks.append(
                    Chunk(
                        chunk_id=str(chunk_id),
                        doc_id=str(doc_id),
                        text=str(text),
                        start=int(start_pos),
                        end=int(end_pos),
                        metadata={"index": int(meta_index)} if meta_index is not None else {},
                    )
                )
                embeddings.append([float(x) for x in embedding])
            if not chunks:
                continue
            if store is None:
                store = SqliteVec1Store(
                    path=sqlite_path,
                    table=target_table,
                    index_table=target_index_table,
                    embedding_dim=len(embeddings[0]),
                    load_extension=load_extension,
                    extension_path=extension_path,
                    extension_whitelist=extension_whitelist,
                )
            store.add(chunks, embeddings)
            migrated += len(chunks)
    finally:
        conn.close()
        if store is not None:
            store.close()
    return migrated


def _iter_sqlite_rows(
    conn: sqlite3.Connection,
    sql: str,
    *,
    batch_size: int = _MIGRATION_BATCH_SIZE,
) -> Iterator[list[sqlite3.Row]]:
    cursor = conn.execute(sql)
    while True:
        rows = cursor.fetchmany(max(1, int(batch_size)))
        if not rows:
            break
        yield rows


def _iter_duckdb_rows(
    conn: object,
    sql: str,
    *,
    batch_size: int = _MIGRATION_BATCH_SIZE,
) -> Iterator[list[Sequence[object]]]:
    cursor = conn.execute(sql)
    while True:
        rows = cursor.fetchmany(max(1, int(batch_size)))
        if not rows:
            break
        yield rows


def _deserialize_float32(blob: bytes) -> List[float]:
    if not blob:
        return []
    if len(blob) % 4 != 0:
        raise ValueError("invalid float32 blob length")
    count = len(blob) // 4
    return list(struct.unpack("<" + "f" * count, blob))
