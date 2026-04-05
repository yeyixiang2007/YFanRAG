"""Plain text and source-code loader."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence

from .base import BaseLoader
from ..models import Document
from ..security import ensure_path_in_whitelist, whitelist_from_env

DEFAULT_TEXT_EXTENSIONS: tuple[str, ...] = (
    ".txt",
    ".md",
    ".rst",
    ".py",
    ".gd",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".java",
    ".go",
    ".rs",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".cs",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".sql",
    ".sh",
    ".ps1",
    ".bat",
)


@dataclass
class TextFileLoader(BaseLoader):
    paths: Sequence[str]
    encoding: str = "utf-8"
    allow_extensions: Sequence[str] = DEFAULT_TEXT_EXTENSIONS
    path_whitelist: Sequence[str] | None = None

    def load(self) -> List[Document]:
        documents: List[Document] = []
        for path in self._iter_paths():
            text = path.read_text(encoding=self.encoding)
            stat = path.stat()
            documents.append(
                Document(
                    doc_id=f"file:{path.as_posix()}",
                    text=text,
                    metadata={
                        "path": path.as_posix(),
                        "size": stat.st_size,
                        "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    },
                    source=path.as_posix(),
                    title=path.stem,
                )
            )
        return documents

    def _iter_paths(self) -> Iterable[Path]:
        allow = {ext.lower() for ext in self.allow_extensions}
        whitelist = list(self.path_whitelist or [])
        if not whitelist:
            whitelist = whitelist_from_env("YFANRAG_PATH_WHITELIST")
        for raw in self.paths:
            path = Path(raw)
            ensure_path_in_whitelist(path, whitelist, label="loader path")
            if path.is_dir():
                yield from self._iter_dir(path, allow)
            elif path.is_file() and path.suffix.lower() in allow:
                yield path

    @staticmethod
    def _iter_dir(root: Path, allow: set[str]) -> Iterable[Path]:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in allow:
                yield path
