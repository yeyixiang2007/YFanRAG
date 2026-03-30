"""Core data models for YFanRAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class Document:
    """A normalized document unit used across loaders and pipelines."""

    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    title: Optional[str] = None
    created_at: Optional[datetime] = None


@dataclass(frozen=True)
class Chunk:
    """A chunk derived from a document."""

    chunk_id: str
    doc_id: str
    text: str
    start: int
    end: int
    metadata: Dict[str, Any] = field(default_factory=dict)
