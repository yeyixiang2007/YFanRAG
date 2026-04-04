"""Security helpers for path whitelist and extension loading isolation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence
import os


def parse_whitelist(value: str | None) -> List[str]:
    if not value:
        return []
    parts = [item.strip() for item in value.split(os.pathsep)]
    return [item for item in parts if item]


def whitelist_from_env(env_name: str) -> List[str]:
    return parse_whitelist(os.getenv(env_name))


def ensure_path_in_whitelist(
    raw_path: str | Path,
    whitelist: Sequence[str] | None,
    label: str = "path",
) -> None:
    if not whitelist:
        return
    target = Path(raw_path).resolve()
    for allowed in whitelist:
        base = Path(allowed).resolve()
        if _is_relative_to(target, base):
            return
    raise ValueError(f"{label} is not in whitelist: {target.as_posix()}")


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False
