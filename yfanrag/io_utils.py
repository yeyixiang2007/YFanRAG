"""Filesystem helpers for durable writes."""

from __future__ import annotations

from pathlib import Path
import os
import tempfile


def write_text_atomic(path: str | Path, text: str, *, encoding: str = "utf-8") -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def append_text_atomic(path: str | Path, text: str, *, encoding: str = "utf-8") -> None:
    target = Path(path)
    existing = ""
    if target.exists():
        existing = target.read_text(encoding=encoding)
    write_text_atomic(target, existing + text, encoding=encoding)
