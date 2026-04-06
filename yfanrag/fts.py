"""SQLite FTS5 index helper."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import List, Sequence
import re
import sqlite3
from time import perf_counter

from .observability import log_slow_query
from .sql_utils import connect_sqlite, delete_by_doc_ids_batched, validate_identifier

_SEARCH_TOKEN_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./#+:-]*|[\u4e00-\u9fff]+")
_CODE_SYMBOL_ALIASES = {
    "c++": ("cpp", "cplusplus", "c"),
    "cpp": ("cpp", "cplusplus", "c"),
    "c#": ("csharp", "c"),
    "f#": ("fsharp", "f"),
}
_MAX_MATCH_GROUPS = 12


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
    _lock: RLock = field(init=False, repr=False, default_factory=RLock)
    _has_search_text: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        self.table = validate_identifier(self.table, label="fts table")
        self._conn = connect_sqlite(self.path)
        with self._lock:
            self._ensure_schema()
            self._has_search_text = self._has_column("search_text")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def add(self, chunks: List[object]) -> None:
        with self._lock:
            self._insert_chunks(chunks)
            self._conn.commit()

    def query(self, query: str, top_k: int = 5) -> List[FtsMatch]:
        start_ts = perf_counter()
        if top_k <= 0:
            return []

        rows: list[sqlite3.Row] = []
        last_error: sqlite3.OperationalError | None = None
        candidates = self._build_match_queries(query)
        with self._lock:
            for candidate in candidates:
                try:
                    current = self._conn.execute(
                        f"SELECT chunk_id, doc_id, text, bm25({self.table}) AS score "
                        f"FROM {self.table} "
                        f"WHERE {self.table} MATCH ? "
                        "ORDER BY score LIMIT ?",
                        (candidate, top_k),
                    ).fetchall()
                except sqlite3.OperationalError as exc:
                    last_error = exc
                    continue
                rows = current
                if rows:
                    break
        if not rows and last_error is not None and len(candidates) == 1:
            raise last_error

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

    @classmethod
    def _sanitize_match_query(cls, query: str) -> str:
        return cls._build_match_query(query, strict=False)

    def delete_by_doc_ids(self, doc_ids: Sequence[str]) -> int:
        ids = [doc_id for doc_id in doc_ids if doc_id]
        if not ids:
            return 0
        with self._lock:
            return delete_by_doc_ids_batched(self._conn, self.table, ids)

    def replace_by_doc_ids(self, doc_ids: Sequence[str], chunks: Sequence[object]) -> int:
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                deleted = delete_by_doc_ids_batched(
                    self._conn,
                    self.table,
                    doc_ids,
                    commit=False,
                )
                self._insert_chunks(chunks)
                self._conn.commit()
                return deleted
            except sqlite3.Error:
                self._conn.rollback()
                raise

    def _ensure_schema(self) -> None:
        try:
            self._conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {self.table} "
                "USING fts5(chunk_id UNINDEXED, doc_id UNINDEXED, text UNINDEXED, search_text)"
            )
        except sqlite3.OperationalError as exc:
            raise RuntimeError("SQLite build does not support FTS5") from exc
        self._conn.commit()

    def _has_column(self, column_name: str) -> bool:
        rows = self._conn.execute(f"PRAGMA table_info({self.table})").fetchall()
        for row in rows:
            if row["name"] == column_name:
                return True
        return False

    def _insert_chunks(self, chunks: Sequence[object]) -> None:
        rows = []
        for chunk in chunks:
            search_text = self._build_search_text(str(chunk.text))
            if self._has_search_text:
                rows.append((chunk.chunk_id, chunk.doc_id, chunk.text, search_text))
            else:
                rows.append((chunk.chunk_id, chunk.doc_id, chunk.text))
        if not rows:
            return
        if self._has_search_text:
            self._conn.executemany(
                f"INSERT INTO {self.table}(chunk_id, doc_id, text, search_text) "
                "VALUES (?, ?, ?, ?)",
                rows,
            )
        else:
            self._conn.executemany(
                f"INSERT INTO {self.table}(chunk_id, doc_id, text) VALUES (?, ?, ?)",
                rows,
            )

    @classmethod
    def _build_match_queries(cls, query: str) -> list[str]:
        candidates: list[str] = []
        for candidate in (
            (query or "").strip(),
            cls._build_match_query(query, strict=True),
            cls._build_match_query(query, strict=False),
        ):
            value = candidate.strip()
            if value and value not in candidates:
                candidates.append(value)
        return candidates

    @classmethod
    def _build_match_query(cls, query: str, *, strict: bool) -> str:
        groups = cls._extract_search_groups(query)
        if not groups:
            return ""
        if strict:
            parts = []
            for group in groups[:_MAX_MATCH_GROUPS]:
                if len(group) == 1:
                    parts.append(cls._quote_term(group[0]))
                else:
                    quoted = " OR ".join(cls._quote_term(term) for term in group)
                    parts.append(f"({quoted})")
            return " ".join(parts)

        terms: list[str] = []
        for group in groups[:_MAX_MATCH_GROUPS]:
            for term in group:
                if term not in terms:
                    terms.append(term)
        return " OR ".join(cls._quote_term(term) for term in terms)

    @classmethod
    def _build_search_text(cls, text: str) -> str:
        compact = " ".join((text or "").split()).strip()
        if not compact:
            return ""
        terms: list[str] = []
        for group in cls._extract_search_groups(compact):
            for term in group:
                if term not in terms:
                    terms.append(term)
        if not terms:
            return compact
        return compact + "\n" + " ".join(terms)

    @classmethod
    def _extract_search_groups(cls, text: str) -> list[list[str]]:
        groups: list[list[str]] = []
        for raw_token in _SEARCH_TOKEN_RE.findall(text or ""):
            token = cls._clean_token(raw_token)
            if not token:
                continue
            if cls._is_cjk(token):
                for gram in cls._cjk_terms(token):
                    groups.append([gram])
                continue
            group = cls._ascii_terms(token)
            if group:
                groups.append(group)
        return groups

    @staticmethod
    def _quote_term(term: str) -> str:
        escaped = term.replace('"', '""')
        return f'"{escaped}"'

    @staticmethod
    def _clean_token(token: str) -> str:
        value = (token or "").strip()
        value = value.strip()
        value = value.lstrip("`'\"([{<")
        value = value.rstrip("`'\"\\])}>.,;:!?")
        return value

    @staticmethod
    def _is_cjk(token: str) -> bool:
        return bool(token) and all("\u4e00" <= ch <= "\u9fff" for ch in token)

    @classmethod
    def _ascii_terms(cls, token: str) -> list[str]:
        value = token.lower()
        terms: list[str] = []
        for alias in _CODE_SYMBOL_ALIASES.get(value, ()):
            if alias not in terms:
                terms.append(alias)

        parts = [
            item.lower()
            for item in re.split(r"[^A-Za-z0-9_]+", value)
            if item
        ]
        compact = "".join(parts)
        if compact and len(compact) > 1 and compact not in terms:
            terms.append(compact)
        if re.fullmatch(r"[A-Za-z0-9_]+", value) and len(value) > 1 and value not in terms:
            terms.append(value)
        for part in parts:
            if (len(part) > 1 or part.isdigit()) and part not in terms:
                terms.append(part)
        return terms

    @staticmethod
    def _cjk_terms(token: str) -> list[str]:
        if len(token) <= 2:
            return [token]
        grams: list[str] = []
        for idx in range(0, len(token) - 1):
            gram = token[idx : idx + 2]
            if gram not in grams:
                grams.append(gram)
        return grams
