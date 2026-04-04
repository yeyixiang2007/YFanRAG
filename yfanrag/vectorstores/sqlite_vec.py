"""SQLite vec0-backed vector store."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence
import sqlite3

from ..models import Chunk

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

    def __post_init__(self) -> None:
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._load_extension()
        if self.embedding_dim is not None:
            self._ensure_schema(self.embedding_dim)

    def close(self) -> None:
        self._conn.close()

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        if not chunks:
            return

        dim = self._infer_dim(embeddings)
        self._ensure_schema(dim)

        rows = []
        for chunk, embedding in zip(chunks, embeddings):
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
        sql = (
            f"INSERT INTO {self.table} "
            "(chunk_id, doc_id, start, end, embedding, text) "
            "VALUES (?, ?, ?, ?, ?, ?)"
        )
        self._conn.executemany(sql, rows)
        self._conn.commit()

    def query(self, embedding: Sequence[float], top_k: int) -> List[Chunk]:
        if top_k <= 0:
            return []
        dim = len(embedding)
        self._ensure_schema(dim)

        sql = (
            f"SELECT chunk_id, doc_id, start, end, text, distance "
            f"FROM {self.table} "
            "WHERE embedding MATCH ? AND k = ?"
        )
        rows = self._conn.execute(sql, (self._serialize(embedding), top_k)).fetchall()
        results: List[Chunk] = []
        for row in rows:
            results.append(
                Chunk(
                    chunk_id=row["chunk_id"],
                    doc_id=row["doc_id"],
                    text=row["text"],
                    start=row["start"],
                    end=row["end"],
                    metadata={"distance": row["distance"]},
                )
            )
        return results

    def delete_by_doc_ids(self, doc_ids: Sequence[str]) -> int:
        ids = [doc_id for doc_id in doc_ids if doc_id]
        if not ids:
            return 0
        if not self._table_exists():
            return 0

        placeholders = ", ".join(["?"] * len(ids))
        sql = f"DELETE FROM {self.table} WHERE doc_id IN ({placeholders})"
        cursor = self._conn.execute(sql, ids)
        self._conn.commit()
        return cursor.rowcount

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
            f"embedding float[{self.embedding_dim}]{distance}, "
            "+text TEXT"
            ")"
        )
        self._conn.execute(sql)
        self._conn.commit()

    def _table_exists(self) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (self.table,),
        ).fetchone()
        return row is not None

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
