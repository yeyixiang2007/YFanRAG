"""Shared SQL helper utilities."""

from __future__ import annotations

from collections.abc import Sequence
import re
import sqlite3

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SQLITE_DELETE_CHUNK_SIZE = 500


def validate_identifier(name: str, label: str = "identifier") -> str:
    value = (name or "").strip()
    if not value:
        raise ValueError(f"{label} cannot be empty")
    if not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(
            f"{label} contains invalid characters: {value!r}; use letters, digits, underscore"
        )
    return value


def connect_sqlite(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False, timeout=30.0)
    _configure_sqlite_connection(conn, path)
    conn.row_factory = sqlite3.Row
    return conn


def delete_by_doc_ids_batched(
    conn: sqlite3.Connection,
    table: str,
    doc_ids: Sequence[str],
    *,
    chunk_size: int = _SQLITE_DELETE_CHUNK_SIZE,
) -> int:
    ids = [doc_id for doc_id in doc_ids if doc_id]
    if not ids:
        return 0

    total = 0
    batch_size = max(1, int(chunk_size))
    for start in range(0, len(ids), batch_size):
        batch = ids[start : start + batch_size]
        placeholders = ", ".join("?" for _ in batch)
        cursor = conn.execute(
            f"DELETE FROM {table} WHERE doc_id IN ({placeholders})",
            batch,
        )
        rowcount = getattr(cursor, "rowcount", 0)
        if isinstance(rowcount, int) and rowcount > 0:
            total += rowcount
    conn.commit()
    return total


def _configure_sqlite_connection(conn: sqlite3.Connection, path: str) -> None:
    conn.execute("PRAGMA busy_timeout = 5000")
    if _is_in_memory_sqlite(path):
        return
    try:
        conn.execute("PRAGMA journal_mode = WAL")
    except sqlite3.Error:
        return
    try:
        conn.execute("PRAGMA synchronous = NORMAL")
    except sqlite3.Error:
        return


def _is_in_memory_sqlite(path: str) -> bool:
    normalized = (path or "").strip().lower()
    if normalized == ":memory:":
        return True
    if normalized.startswith("file:") and "mode=memory" in normalized:
        return True
    return False
