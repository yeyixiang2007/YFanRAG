"""Filesystem helpers for durable writes."""

from __future__ import annotations

from contextlib import contextmanager
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
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = text.encode(encoding)
    with target.open("a+b") as handle:
        with _exclusive_file_lock(handle):
            handle.seek(0, os.SEEK_END)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())


@contextmanager
def _exclusive_file_lock(handle: object):
    lock_impl = None
    if os.name == "nt":
        import msvcrt

        def _lock() -> None:
            handle.seek(0)  # type: ignore[attr-defined]
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)  # type: ignore[attr-defined]

        def _unlock() -> None:
            handle.seek(0)  # type: ignore[attr-defined]
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]

        lock_impl = (_lock, _unlock)
    else:
        import fcntl

        def _lock() -> None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)  # type: ignore[attr-defined]

        def _unlock() -> None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]

        lock_impl = (_lock, _unlock)

    lock, unlock = lock_impl
    lock()
    try:
        yield
    finally:
        unlock()
