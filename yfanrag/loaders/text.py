"""Plain text and Markdown loader."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence

from .base import BaseLoader
from ..models import Document


@dataclass
class TextFileLoader(BaseLoader):
    paths: Sequence[str]
    encoding: str = "utf-8"
    allow_extensions: Sequence[str] = (".txt", ".md")

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
        for raw in self.paths:
            path = Path(raw)
            if path.is_dir():
                yield from self._iter_dir(path, allow)
            elif path.is_file() and path.suffix.lower() in allow:
                yield path

    @staticmethod
    def _iter_dir(root: Path, allow: set[str]) -> Iterable[Path]:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in allow:
                yield path
