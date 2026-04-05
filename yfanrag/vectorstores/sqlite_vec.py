"""SQLite vec0-backed vector store."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import List, Sequence
from time import perf_counter
import sqlite3

from ..interfaces import FieldFilters, RangeFilters
from ..models import Chunk
from ..observability import log_slow_query
from ..sql_utils import connect_sqlite, delete_by_doc_ids_batched, validate_identifier

try:
    import sqlite_vec
except ImportError:  # pragma: no cover - optional dependency
    sqlite_vec = None


@dataclass
class SqliteVecStore:
    path: str = "yfanrag.db"
    table: str = "vec_chunks"
    embedding_dim: int | None = None
    distance_metric: str | None = None

    _conn: sqlite3.Connection = field(init=False, repr=False)
    _lock: RLock = field(init=False, repr=False, default_factory=RLock)

    _eq_filter_columns = {
        "chunk_id": "chunk_id",
        "doc_id": "doc_id",
        "start": "start",
        "end": "end",
        "index": "meta_index",
    }
    _range_filter_columns = {
        "start": "start",
        "end": "end",
        "index": "meta_index",
    }

    def __post_init__(self) -> None:
        self.table = validate_identifier(self.table, label="vector table")
        self._conn = connect_sqlite(self.path)
        with self._lock:
            self._load_extension()
            if self.embedding_dim is not None:
                self._ensure_schema(self.embedding_dim)

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
            self._ensure_schema(dim)

            rows = []
            has_index_col = self._has_column("meta_index")
            for chunk, embedding in zip(chunks, embeddings):
                index_value = chunk.metadata.get("index")
                if has_index_col:
                    rows.append(
                        (
                            chunk.chunk_id,
                            chunk.doc_id,
                            chunk.start,
                            chunk.end,
                            index_value,
                            self._serialize(embedding),
                            chunk.text,
                        )
                    )
                else:
                    rows.append(
                        (
                            chunk.chunk_id,
                            chunk.doc_id,
                            chunk.start,
                            chunk.end,
                            self._serialize(embedding),
                            chunk.text,
                        )
                    )
            if has_index_col:
                sql = (
                    f"INSERT INTO {self.table} "
                    "(chunk_id, doc_id, start, end, meta_index, embedding, text) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)"
                )
            else:
                sql = (
                    f"INSERT INTO {self.table} "
                    "(chunk_id, doc_id, start, end, embedding, text) "
                    "VALUES (?, ?, ?, ?, ?, ?)"
                )
            self._conn.executemany(sql, rows)
            self._conn.commit()

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
        dim = len(embedding)
        with self._lock:
            self._ensure_schema(dim)
            where_clauses = ["embedding MATCH ?", "k = ?"]
            params: list[object] = [self._serialize(embedding), top_k]

            for key, value in (filters or {}).items():
                column = self._resolve_filter_column(key, is_range=False)
                where_clauses.append(f"{column} = ?")
                params.append(value)

            for key, (lower, upper) in (range_filters or {}).items():
                column = self._resolve_filter_column(key, is_range=True)
                if lower is not None:
                    where_clauses.append(f"{column} >= ?")
                    params.append(lower)
                if upper is not None:
                    where_clauses.append(f"{column} <= ?")
                    params.append(upper)

            select_cols = "chunk_id, doc_id, start, end, text, distance"
            has_index_col = self._has_column("meta_index")
            if has_index_col:
                select_cols += ", meta_index"

            sql = (
                f"SELECT {select_cols} "
                f"FROM {self.table} "
                f"WHERE {' AND '.join(where_clauses)}"
            )
            rows = self._conn.execute(sql, params).fetchall()
        results: List[Chunk] = []
        for row in rows:
            metadata = {"distance": row["distance"]}
            if has_index_col and row["meta_index"] is not None:
                metadata["index"] = row["meta_index"]
            results.append(
                Chunk(
                    chunk_id=row["chunk_id"],
                    doc_id=row["doc_id"],
                    text=row["text"],
                    start=row["start"],
                    end=row["end"],
                    metadata=metadata,
                )
            )
        elapsed_ms = (perf_counter() - start_ts) * 1000.0
        log_slow_query("SqliteVecStore.query", elapsed_ms, f"rows={len(results)}")
        return results

    def delete_by_doc_ids(self, doc_ids: Sequence[str]) -> int:
        ids = [doc_id for doc_id in doc_ids if doc_id]
        if not ids:
            return 0
        with self._lock:
            if not self._table_exists():
                return 0
            return delete_by_doc_ids_batched(self._conn, self.table, ids)

    def _load_extension(self) -> None:
        if sqlite_vec is None:
            raise RuntimeError(
                "sqlite-vec is not installed. Install with `pip install sqlite-vec`."
            )
        if not hasattr(self._conn, "enable_load_extension"):
            raise RuntimeError("SQLite extension loading is not supported in this build")
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)

    def _ensure_schema(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError("embedding dimension must be positive")
        if self.embedding_dim is None:
            self.embedding_dim = dim
        if self.embedding_dim != dim:
            raise ValueError(
                f"embedding dimension mismatch: expected {self.embedding_dim}, got {dim}"
            )

        distance = ""
        if self.distance_metric:
            distance = f" distance_metric={self.distance_metric}"

        sql = (
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {self.table} USING vec0("
            "chunk_id TEXT, "
            "doc_id TEXT, "
            "start INTEGER, "
            "end INTEGER, "
            "meta_index INTEGER, "
            f"embedding float[{self.embedding_dim}]{distance}, "
            "+text TEXT"
            ")"
        )
        self._conn.execute(sql)
        self._conn.commit()

    def _resolve_filter_column(self, key: str, is_range: bool) -> str:
        mapping = self._range_filter_columns if is_range else self._eq_filter_columns
        if key not in mapping:
            kind = "range" if is_range else "field"
            raise ValueError(f"unsupported {kind} filter key: {key}")
        column = mapping[key]
        if not self._has_column(column):
            raise ValueError(
                f"filter key '{key}' requires column '{column}' which is not available; "
                "please re-ingest data with current schema"
            )
        return column

    def _table_exists(self) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (self.table,),
        ).fetchone()
        return row is not None

    def _has_column(self, column_name: str) -> bool:
        if not self._table_exists():
            return False
        rows = self._conn.execute(f"PRAGMA table_info({self.table})").fetchall()
        for row in rows:
            if row["name"] == column_name:
                return True
        return False

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
    def _serialize(embedding: Sequence[float]) -> bytes:
        if sqlite_vec is None:  # pragma: no cover - defensive
            raise RuntimeError("sqlite-vec not installed")
        return sqlite_vec.serialize_float32(list(embedding))
