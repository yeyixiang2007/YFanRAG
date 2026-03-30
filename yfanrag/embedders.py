"""Reference embedder implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence
import math


@dataclass
class HashingEmbedder:
    """Deterministic, dependency-free embedder for tests and demos."""

    dims: int = 8

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if self.dims <= 0:
            raise ValueError("dims must be positive")

        vectors: List[List[float]] = []
        for text in texts:
            vec = [0.0] * self.dims
            for ch in text:
                idx = ord(ch) % self.dims
                vec[idx] += 1.0
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            vectors.append([x / norm for x in vec])
        return vectors
