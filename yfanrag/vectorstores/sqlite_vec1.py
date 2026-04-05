"""SQLite vec1-backed vector store with graceful fallback."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Dict, List, Sequence
import json
import math
import sqlite3
import struct
from time import perf_counter

from ..interfaces import FieldFilters, RangeFilters
from ..models import Chunk
from ..observability import log_slow_query
from ..security import ensure_path_in_whitelist, whitelist_from_env
from ..sql_utils import connect_sqlite, delete_by_doc_ids_batched, validate_identifier


@dataclass
class SqliteVec1Store:
    path: str = "yfanrag.db"
    table: str = "vec1_chunks_data"
    index_table: str = "vec1_chunks_index"
    embedding_dim: int | None = None
    distance_metric: str = "l2"
    load_extension: bool = True
    extension_path: str | None = None
    extension_whitelist: Sequence[str] | None = None

    _conn: sqlite3.Connection = field(init=False, repr=False)
    _lock: RLock = field(init=False, repr=False, default_factory=RLock)
    _vec1_enabled: bool = field(init=False, default=False, repr=False)

    _eq_filter_columns = {
        "chunk_id": "chunk_id",
        "doc_id": "doc_id",
        "start": "start",
        "end": "end_pos",
        "index": "meta_index",
    }
    _range_filter_columns = {
        "start": "start",
        "end": "end_pos",
        "index": "meta_index",
    }

    def __post_init__(self) -> None:
        self.table = validate_identifier(self.table, label="vector table")
        self.index_table = validate_identifier(self.index_table, label="vector index table")
        self._conn = connect_sqlite(self.path)
        with self._lock:
            self._ensure_data_schema()

            self._vec1_enabled = False
            if self.load_extension:
                self._vec1_enabled = self._load_vec1_extension()
                if self._vec1_enabled:
                    self._ensure_vec1_schema()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        if not chunks:
            return

        dim = self._infer_dim(embeddings)
        with self._lock:
            self._ensure_dim(dim)

            rows = []
            for chunk, embedding in zip(chunks, embeddings):
                rows.append(
                    (
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.start,
                        chunk.end,
                        chunk.metadata.get("index"),
                        chunk.text,
                        self._serialize_float32(embedding),
                    )
                )
            self._conn.executemany(
                f"INSERT INTO {self.table} "
                "(chunk_id, doc_id, start, end_pos, meta_index, text, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            self._conn.commit()

            if self._vec1_enabled:
                self._sync_vec1_index()

    def query(
        self,
        embedding: Sequence[float],
        top_k: int,
        filters: FieldFilters | None = None,
        range_filters: RangeFilters | None = None,
    ) -> List[Chunk]:
        start_ts = perf_counter()
        if top_k <= 0:
            return []
        if self.embedding_dim is None:
            self.embedding_dim = len(embedding)
        if len(embedding) != self.embedding_dim:
            raise ValueError(
                f"embedding dimension mismatch: expected {self.embedding_dim}, got {len(embedding)}"
            )

        with self._lock:
            if self._vec1_enabled:
                try:
                    ids = self._query_vec1_rowids(embedding, top_k, filters, range_filters)
                    if ids:
                        result = self._load_chunks_by_ids(ids, embedding)
                        elapsed_ms = (perf_counter() - start_ts) * 1000.0
                        log_slow_query("SqliteVec1Store.query", elapsed_ms, f"rows={len(result)}")
                        return result
                except sqlite3.Error:
                    # If vec1 query shape changes or extension is unavailable at runtime,
                    # continue with deterministic exact search fallback.
                    pass

            result = self._query_exhaustive(embedding, top_k, filters, range_filters)
        elapsed_ms = (perf_counter() - start_ts) * 1000.0
        log_slow_query("SqliteVec1Store.query", elapsed_ms, f"rows={len(result)}")
        return result

    def delete_by_doc_ids(self, doc_ids: Sequence[str]) -> int:
        ids = [doc_id for doc_id in doc_ids if doc_id]
        if not ids:
            return 0
        with self._lock:
            deleted = delete_by_doc_ids_batched(self._conn, self.table, ids)
            if self._vec1_enabled:
                self._sync_vec1_index()
            return deleted

    def _ensure_data_schema(self) -> None:
        self._conn.execute(
            f"CREATE TABLE IF NOT EXISTS {self.table} ("
            "id INTEGER PRIMARY KEY, "
            "chunk_id TEXT UNIQUE, "
            "doc_id TEXT, "
            "start INTEGER, "
            "end_pos INTEGER, "
            "meta_index INTEGER, "
            "text TEXT, "
            "embedding BLOB"
            ")"
        )
        self._conn.commit()

    def _ensure_vec1_schema(self) -> None:
        self._conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {self.index_table} "
            "USING vec1(vector, doc_id, start_pos, end_pos, meta_index, chunk_id)"
        )
        self._conn.commit()
        self._sync_vec1_index()

    def _load_vec1_extension(self) -> bool:
        if not hasattr(self._conn, "enable_load_extension"):
            return False
        try:
            self._conn.enable_load_extension(True)
            if self.extension_path:
                whitelist = list(self.extension_whitelist or [])
                if not whitelist:
                    whitelist = whitelist_from_env("YFANRAG_EXTENSION_WHITELIST")
                ensure_path_in_whitelist(
                    self.extension_path,
                    whitelist,
                    label="sqlite extension path",
                )
                self._conn.load_extension(self.extension_path)
            else:
                self._conn.load_extension("vec1")
            return True
        except sqlite3.Error:
            return False
        finally:
            try:
                self._conn.enable_load_extension(False)
            except sqlite3.Error:
                pass

    def _sync_vec1_index(self) -> None:
        if not self._vec1_enabled:
            return
        self._conn.execute(f"DELETE FROM {self.index_table}")
        self._conn.execute(
            f"INSERT INTO {self.index_table}(rowid, vector, doc_id, start_pos, end_pos, meta_index, chunk_id) "
            f"SELECT id, embedding, doc_id, start, end_pos, meta_index, chunk_id FROM {self.table}"
        )
        mode = {"index": "flat", "distance": self._vec1_distance_name()}
        self._conn.execute(
            f"INSERT INTO {self.index_table}(cmd, arg) VALUES('rebuild', ?)",
            (json.dumps(mode),),
        )
        self._conn.commit()

    def _query_vec1_rowids(
        self,
        embedding: Sequence[float],
        top_k: int,
        filters: FieldFilters | None,
        range_filters: RangeFilters | None,
    ) -> List[int]:
        where = []
        params: list[object] = [
            self._serialize_float32(embedding),
            json.dumps({"k": top_k}),
        ]
        for key, value in (filters or {}).items():
            col = self._resolve_filter_column(key, is_range=False)
            where.append(f"{col} = ?")
            params.append(value)
        for key, (lower, upper) in (range_filters or {}).items():
            col = self._resolve_filter_column(key, is_range=True)
            if lower is not None:
                where.append(f"{col} >= ?")
                params.append(lower)
            if upper is not None:
                where.append(f"{col} <= ?")
                params.append(upper)

        sql = f"SELECT rowid FROM {self.index_table}(?, ?)"
        if where:
            sql += " WHERE " + " AND ".join(where)
        rows = self._conn.execute(sql, params).fetchall()
        return [int(row["rowid"]) for row in rows]

    def _load_chunks_by_ids(
        self,
        ids: Sequence[int],
        query_embedding: Sequence[float],
    ) -> List[Chunk]:
        if not ids:
            return []
        placeholders = ", ".join(["?"] * len(ids))
        rows = self._conn.execute(
            f"SELECT id, chunk_id, doc_id, start, end_pos, meta_index, text, embedding "
            f"FROM {self.table} WHERE id IN ({placeholders})",
            list(ids),
        ).fetchall()
        row_map = {int(row["id"]): row for row in rows}
        chunks: List[Chunk] = []
        for item_id in ids:
            row = row_map.get(item_id)
            if row is None:
                continue
            vec = self._deserialize_float32(row["embedding"])
            distance = self._distance(query_embedding, vec)
            metadata: Dict[str, object] = {"distance": distance}
            if row["meta_index"] is not None:
                metadata["index"] = row["meta_index"]
            chunks.append(
                Chunk(
                    chunk_id=row["chunk_id"],
                    doc_id=row["doc_id"],
                    text=row["text"],
                    start=row["start"],
                    end=row["end_pos"],
                    metadata=metadata,
                )
            )
        return chunks

    def _query_exhaustive(
        self,
        embedding: Sequence[float],
        top_k: int,
        filters: FieldFilters | None,
        range_filters: RangeFilters | None,
    ) -> List[Chunk]:
        where = []
        params: list[object] = []
        for key, value in (filters or {}).items():
            col = self._resolve_filter_column(key, is_range=False)
            where.append(f"{col} = ?")
            params.append(value)
        for key, (lower, upper) in (range_filters or {}).items():
            col = self._resolve_filter_column(key, is_range=True)
            if lower is not None:
                where.append(f"{col} >= ?")
                params.append(lower)
            if upper is not None:
                where.append(f"{col} <= ?")
                params.append(upper)

        sql = (
            f"SELECT chunk_id, doc_id, start, end_pos, meta_index, text, embedding "
            f"FROM {self.table}"
        )
        if where:
            sql += " WHERE " + " AND ".join(where)
        rows = self._conn.execute(sql, params).fetchall()

        scored = []
        for row in rows:
            vec = self._deserialize_float32(row["embedding"])
            distance = self._distance(embedding, vec)
            metadata: Dict[str, object] = {"distance": distance}
            if row["meta_index"] is not None:
                metadata["index"] = row["meta_index"]
            scored.append(
                (
                    distance,
                    Chunk(
                        chunk_id=row["chunk_id"],
                        doc_id=row["doc_id"],
                        text=row["text"],
                        start=row["start"],
                        end=row["end_pos"],
                        metadata=metadata,
                    ),
                )
            )
        scored.sort(key=lambda item: item[0])
        return [chunk for _, chunk in scored[:top_k]]

    def _resolve_filter_column(self, key: str, is_range: bool) -> str:
        mapping = self._range_filter_columns if is_range else self._eq_filter_columns
        if key not in mapping:
            kind = "range" if is_range else "field"
            raise ValueError(f"unsupported {kind} filter key: {key}")
        return mapping[key]

    def _ensure_dim(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError("embedding dimension must be positive")
        if self.embedding_dim is None:
            self.embedding_dim = dim
        if self.embedding_dim != dim:
            raise ValueError(
                f"embedding dimension mismatch: expected {self.embedding_dim}, got {dim}"
            )

    @staticmethod
    def _infer_dim(embeddings: Sequence[Sequence[float]]) -> int:
        first = embeddings[0]
        if not first:
            raise ValueError("empty embedding vector")
        dim = len(first)
        for emb in embeddings[1:]:
            if len(emb) != dim:
                raise ValueError("inconsistent embedding dimensions")
        return dim

    @staticmethod
    def _serialize_float32(embedding: Sequence[float]) -> bytes:
        return struct.pack("<" + "f" * len(embedding), *[float(x) for x in embedding])

    @staticmethod
    def _deserialize_float32(blob: bytes) -> List[float]:
        if not blob:
            return []
        if len(blob) % 4 != 0:
            raise ValueError("invalid float32 blob length")
        count = len(blob) // 4
        return list(struct.unpack("<" + "f" * count, blob))

    def _vec1_distance_name(self) -> str:
        if self.distance_metric == "cosine":
            return "cos"
        if self.distance_metric == "l2":
            return "l2"
        raise ValueError(f"unsupported distance metric for vec1: {self.distance_metric}")

    def _distance(self, a: Sequence[float], b: Sequence[float]) -> float:
        if len(a) != len(b):
            raise ValueError("embedding dimension mismatch")
        if self.distance_metric == "l2":
            return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
        if self.distance_metric == "cosine":
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(y * y for y in b))
            if na == 0 or nb == 0:
                return 1.0
            return 1.0 - dot / (na * nb)
        raise ValueError(f"unsupported distance metric: {self.distance_metric}")
