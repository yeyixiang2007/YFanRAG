"""DuckDB VSS-backed vector store with safe fallback behaviors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence
from time import perf_counter

from ..interfaces import FieldFilters, RangeFilters
from ..models import Chunk
from ..observability import log_slow_query

try:
    import duckdb
except ImportError:  # pragma: no cover - optional dependency
    duckdb = None


@dataclass
class DuckDbVssStore:
    path: str = "yfanrag.duckdb"
    table: str = "vss_chunks"
    embedding_dim: int | None = None
    distance_metric: str = "l2"
    enable_vss: bool = True
    persistent_index: bool = False
    fail_if_no_vss: bool = False

    _conn: object = field(init=False, repr=False)
    _vss_enabled: bool = field(init=False, default=False, repr=False)
    _index_ready: bool = field(init=False, default=False, repr=False)

    _eq_filter_columns = {
        "chunk_id": "chunk_id",
        "doc_id": "doc_id",
        "start": "start_pos",
        "end": "end_pos",
        "index": "meta_index",
    }
    _range_filter_columns = {
        "start": "start_pos",
        "end": "end_pos",
        "index": "meta_index",
    }

    def __post_init__(self) -> None:
        if duckdb is None:
            raise RuntimeError(
                "duckdb is not installed. Install with `pip install duckdb`."
            )
        self._conn = duckdb.connect(self.path)
        if self.persistent_index:
            self._conn.execute("SET hnsw_enable_experimental_persistence = true")

        self._vss_enabled = False
        if self.enable_vss:
            self._vss_enabled = self._try_enable_vss()
            if not self._vss_enabled and self.fail_if_no_vss:
                raise RuntimeError("DuckDB vss extension is not available")

    def close(self) -> None:
        self._conn.close()

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        if not chunks:
            return

        dim = self._infer_dim(embeddings)
        self._ensure_schema(dim)

        sql = (
            f"INSERT INTO {self.table} "
            "(chunk_id, doc_id, start_pos, end_pos, meta_index, text, embedding) "
            f"VALUES (?, ?, ?, ?, ?, ?, ?::FLOAT[{self.embedding_dim}])"
        )
        rows = []
        for chunk, emb in zip(chunks, embeddings):
            rows.append(
                (
                    chunk.chunk_id,
                    chunk.doc_id,
                    chunk.start,
                    chunk.end,
                    chunk.metadata.get("index"),
                    chunk.text,
                    [float(x) for x in emb],
                )
            )
        self._conn.executemany(sql, rows)

        if self._vss_enabled:
            self._ensure_hnsw_index()

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
        self._ensure_dim(len(embedding))

        where = []
        params: list[object] = [[float(x) for x in embedding]]
        for key, value in (filters or {}).items():
            where.append(f"{self._resolve_filter_column(key, is_range=False)} = ?")
            params.append(value)
        for key, (lower, upper) in (range_filters or {}).items():
            col = self._resolve_filter_column(key, is_range=True)
            if lower is not None:
                where.append(f"{col} >= ?")
                params.append(lower)
            if upper is not None:
                where.append(f"{col} <= ?")
                params.append(upper)
        params.append(top_k)

        distance_fn = self._distance_fn_name()
        sql = (
            f"SELECT chunk_id, doc_id, start_pos, end_pos, meta_index, text, "
            f"{distance_fn}(embedding, ?::FLOAT[{self.embedding_dim}]) AS distance "
            f"FROM {self.table}"
        )
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY distance LIMIT ?"

        rows = self._conn.execute(sql, params).fetchall()
        results: List[Chunk] = []
        for row in rows:
            metadata: dict[str, object] = {"distance": float(row[6])}
            if row[4] is not None:
                metadata["index"] = int(row[4])
            results.append(
                Chunk(
                    chunk_id=row[0],
                    doc_id=row[1],
                    text=row[5],
                    start=int(row[2]),
                    end=int(row[3]),
                    metadata=metadata,
                )
            )
        elapsed_ms = (perf_counter() - start_ts) * 1000.0
        log_slow_query("DuckDbVssStore.query", elapsed_ms, f"rows={len(results)}")
        return results

    def delete_by_doc_ids(self, doc_ids: Sequence[str]) -> int:
        ids = [doc_id for doc_id in doc_ids if doc_id]
        if not ids:
            return 0
        placeholders = ", ".join(["?"] * len(ids))
        sql = f"DELETE FROM {self.table} WHERE doc_id IN ({placeholders})"
        cursor = self._conn.execute(sql, ids)
        try:
            return int(cursor.rowcount)
        except Exception:
            return 0

    def _ensure_schema(self, dim: int) -> None:
        self._ensure_dim(dim)
        self._conn.execute(
            f"CREATE TABLE IF NOT EXISTS {self.table} ("
            "chunk_id VARCHAR, "
            "doc_id VARCHAR, "
            "start_pos INTEGER, "
            "end_pos INTEGER, "
            "meta_index INTEGER, "
            "text VARCHAR, "
            f"embedding FLOAT[{self.embedding_dim}]"
            ")"
        )

    def _ensure_hnsw_index(self) -> None:
        if self._index_ready:
            return
        metric = self._hnsw_metric_name()
        index_name = f"idx_{self.table}_embedding_hnsw"
        try:
            self._conn.execute(
                f"CREATE INDEX {index_name} ON {self.table} USING HNSW (embedding) "
                f"WITH (metric='{metric}')"
            )
        except Exception as exc:
            msg = str(exc).lower()
            if "already exists" not in msg:
                return
        self._index_ready = True

    def _try_enable_vss(self) -> bool:
        try:
            self._conn.execute("INSTALL vss")
            self._conn.execute("LOAD vss")
            return True
        except Exception:
            return False

    def _ensure_dim(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError("embedding dimension must be positive")
        if self.embedding_dim is None:
            self.embedding_dim = dim
        if self.embedding_dim != dim:
            raise ValueError(
                f"embedding dimension mismatch: expected {self.embedding_dim}, got {dim}"
            )

    def _resolve_filter_column(self, key: str, is_range: bool) -> str:
        mapping = self._range_filter_columns if is_range else self._eq_filter_columns
        if key not in mapping:
            kind = "range" if is_range else "field"
            raise ValueError(f"unsupported {kind} filter key: {key}")
        return mapping[key]

    def _distance_fn_name(self) -> str:
        if self.distance_metric == "l2":
            return "array_distance"
        if self.distance_metric == "cosine":
            return "array_cosine_distance"
        raise ValueError(f"unsupported distance metric: {self.distance_metric}")

    def _hnsw_metric_name(self) -> str:
        if self.distance_metric == "l2":
            return "l2sq"
        if self.distance_metric == "cosine":
            return "cosine"
        raise ValueError(f"unsupported distance metric for HNSW: {self.distance_metric}")

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
