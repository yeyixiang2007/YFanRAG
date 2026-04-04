"""Migration helpers across storage backends."""

from __future__ import annotations

from typing import List
import sqlite3
import struct

from .models import Chunk
from .vectorstores.duckdb_vss import DuckDbVssStore
from .vectorstores.sqlite_vec1 import SqliteVec1Store

try:
    import duckdb
except ImportError:  # pragma: no cover - optional dependency
    duckdb = None


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
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            f"SELECT chunk_id, doc_id, start, end, text, embedding FROM {source_table}"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return 0

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

    store = SqliteVec1Store(
        path=path,
        table=target_table,
        index_table=target_index_table,
        embedding_dim=len(embeddings[0]),
        load_extension=load_extension,
        extension_path=extension_path,
        extension_whitelist=extension_whitelist,
    )
    try:
        store.add(chunks, embeddings)
    finally:
        store.close()
    return len(chunks)


def migrate_sqlite_vec1_to_duckdb_vss(
    sqlite_path: str,
    duckdb_path: str,
    source_table: str = "vec1_chunks_data",
    target_table: str = "vss_chunks",
    enable_vss: bool = True,
    persistent_index: bool = False,
) -> int:
    """Migrate sqlite vec1-table rows to DuckDB VSS table."""
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            f"SELECT chunk_id, doc_id, start, end_pos, meta_index, text, embedding FROM {source_table}"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return 0

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

    store = DuckDbVssStore(
        path=duckdb_path,
        table=target_table,
        embedding_dim=len(embeddings[0]),
        enable_vss=enable_vss,
        persistent_index=persistent_index,
        fail_if_no_vss=False,
    )
    try:
        store.add(chunks, embeddings)
    finally:
        store.close()
    return len(chunks)


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

    conn = duckdb.connect(duckdb_path)
    try:
        rows = conn.execute(
            f"SELECT chunk_id, doc_id, start_pos, end_pos, meta_index, text, embedding FROM {source_table}"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return 0

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

    store = SqliteVec1Store(
        path=sqlite_path,
        table=target_table,
        index_table=target_index_table,
        embedding_dim=len(embeddings[0]),
        load_extension=load_extension,
        extension_path=extension_path,
        extension_whitelist=extension_whitelist,
    )
    try:
        store.add(chunks, embeddings)
    finally:
        store.close()
    return len(chunks)


def _deserialize_float32(blob: bytes) -> List[float]:
    if not blob:
        return []
    if len(blob) % 4 != 0:
        raise ValueError("invalid float32 blob length")
    count = len(blob) // 4
    return list(struct.unpack("<" + "f" * count, blob))
