"""Migration helpers across storage backends."""

from __future__ import annotations

from typing import List
import sqlite3
import struct

from .models import Chunk
from .vectorstores.sqlite_vec1 import SqliteVec1Store


def migrate_sqlite_vec0_to_vec1(
    path: str,
    source_table: str = "vec_chunks",
    target_table: str = "vec1_chunks_data",
    target_index_table: str = "vec1_chunks_index",
    load_extension: bool = True,
    extension_path: str | None = None,
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
