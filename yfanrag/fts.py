"""SQLite FTS5 index helper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence
import re
import sqlite3
from time import perf_counter

from .observability import log_slow_query

_MATCH_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+")


@dataclass(frozen=True)
class FtsMatch:
    chunk_id: str
    doc_id: str | None
    text: str
    score: float


@dataclass
class SqliteFtsIndex:
    path: str = "yfanrag.db"
    table: str = "fts_chunks"

    _conn: sqlite3.Connection = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def close(self) -> None:
        self._conn.close()

    def add(self, chunks: List[object]) -> None:
        rows = []
        for chunk in chunks:
            rows.append((chunk.chunk_id, chunk.doc_id, chunk.text))
        self._conn.executemany(
            f"INSERT INTO {self.table}(chunk_id, doc_id, text) VALUES (?, ?, ?)",
            rows,
        )
        self._conn.commit()

    def query(self, query: str, top_k: int = 5) -> List[FtsMatch]:
        start_ts = perf_counter()
        if top_k <= 0:
            return []
        sql = (
            f"SELECT chunk_id, doc_id, text, bm25({self.table}) as score "
            f"FROM {self.table} "
            f"WHERE {self.table} MATCH ? "
            "ORDER BY score LIMIT ?"
        )
        try:
            rows = self._conn.execute(sql, (query, top_k)).fetchall()
        except sqlite3.OperationalError as exc:
            safe_query = self._sanitize_match_query(query)
            if not safe_query or safe_query == query.strip():
                raise
            rows = self._conn.execute(sql, (safe_query, top_k)).fetchall()
        results = [
            FtsMatch(
                chunk_id=row["chunk_id"],
                doc_id=row["doc_id"],
                text=row["text"],
                score=row["score"],
            )
            for row in rows
        ]
        elapsed_ms = (perf_counter() - start_ts) * 1000.0
        log_slow_query("SqliteFtsIndex.query", elapsed_ms, f"rows={len(results)}")
        return results

    @staticmethod
    def _sanitize_match_query(query: str) -> str:
        terms = _MATCH_TOKEN_RE.findall(query or "")
        if not terms:
            return ""
        quoted_terms = [f'"{term.replace("\"", "\"\"")}"' for term in terms[:12]]
        return " OR ".join(quoted_terms)

    def delete_by_doc_ids(self, doc_ids: Sequence[str]) -> int:
        ids = [doc_id for doc_id in doc_ids if doc_id]
        if not ids:
            return 0
        placeholders = ", ".join(["?"] * len(ids))
        sql = f"DELETE FROM {self.table} WHERE doc_id IN ({placeholders})"
        cursor = self._conn.execute(sql, ids)
        self._conn.commit()
        return cursor.rowcount

    def _ensure_schema(self) -> None:
        try:
            self._conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {self.table} "
                "USING fts5(chunk_id, doc_id, text)"
            )
        except sqlite3.OperationalError as exc:
            raise RuntimeError("SQLite build does not support FTS5") from exc
        self._conn.commit()
