"""Loader base classes."""

from __future__ import annotations

from typing import List

from ..interfaces import DocumentLoader
from ..models import Document


class BaseLoader(DocumentLoader):
    def load(self) -> List[Document]:
        raise NotImplementedError
