"""GUI worker event models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkerEvent:
    """Event message exchanged between worker thread and UI thread."""

    kind: str  # delta | done | error
    text: str = ""
    stopped: bool = False
