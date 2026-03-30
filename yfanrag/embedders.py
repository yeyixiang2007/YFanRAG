"""Reference embedder implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence
import math
import json
import os
import urllib.request
import urllib.error


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


@dataclass
class HttpEmbedder:
    """Generic HTTP embedder for API-backed embeddings."""

    endpoint: str
    model: str | None = None
    api_key_env: str | None = None
    timeout: int = 30

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not self.endpoint:
            raise ValueError("endpoint is required for HttpEmbedder")

        payload = {"texts": list(texts)}
        if self.model:
            payload["model"] = self.model

        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}

        if self.api_key_env:
            api_key = os.getenv(self.api_key_env)
            if not api_key:
                raise ValueError(f"missing API key in env var: {self.api_key_env}")
            headers["Authorization"] = f"Bearer {api_key}"

        request = urllib.request.Request(
            self.endpoint,
            data=data,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.URLError as exc:
            raise RuntimeError(f"failed to call embedding endpoint: {exc}") from exc

        payload = json.loads(raw.decode("utf-8"))
        if "embeddings" in payload:
            return payload["embeddings"]
        if "data" in payload:
            return [item["embedding"] for item in payload["data"]]
        raise ValueError("invalid embedding response payload")


class EmbedderFactory:
    """Factory for building embedders from config-like data."""

    @staticmethod
    def from_config(config: object) -> HashingEmbedder | HttpEmbedder:
        provider = getattr(config, "provider", "local")
        provider = provider.lower()
        if provider in {"local", "hashing"}:
            dims = getattr(config, "dims", 8) or 8
            return HashingEmbedder(dims=dims)
        if provider in {"http", "api"}:
            endpoint = getattr(config, "endpoint", None)
            model = getattr(config, "model", None)
            api_key_env = getattr(config, "api_key_env", None)
            return HttpEmbedder(
                endpoint=endpoint,
                model=model,
                api_key_env=api_key_env,
            )
        raise ValueError(f"unknown embedder provider: {provider}")
