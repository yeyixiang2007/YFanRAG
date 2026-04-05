"""Reference embedder implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence
import importlib
import json
import math
import os
import urllib.error
import urllib.request


DEFAULT_FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"
_FASTEMBED_MODEL_DIMS = {
    DEFAULT_FASTEMBED_MODEL: 384,
}


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

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return self.embed(texts)

    def embed_queries(self, texts: Sequence[str]) -> List[List[float]]:
        return self.embed(texts)


@dataclass
class FastEmbedder:
    """Lightweight local semantic embedder backed by FastEmbed."""

    model_name: str = DEFAULT_FASTEMBED_MODEL
    batch_size: int = 256
    cache_dir: str | None = None
    threads: int | None = None

    _model: object | None = field(init=False, repr=False, default=None)

    @property
    def dims(self) -> int | None:
        return fastembed_model_dims(self.model_name)

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return self._embed_with_method("passage_embed", texts)

    def embed_queries(self, texts: Sequence[str]) -> List[List[float]]:
        return self._embed_with_method("query_embed", texts)

    def _embed_with_method(self, method_name: str, texts: Sequence[str]) -> List[List[float]]:
        items = list(texts)
        if not items:
            return []
        model = self._get_model()
        method = getattr(model, method_name, None)
        if not callable(method):
            method = getattr(model, "embed", None)
        if not callable(method):  # pragma: no cover - defensive
            raise RuntimeError("FastEmbed model does not expose an embedding method")
        try:
            output = method(items, batch_size=self.batch_size)
        except TypeError:
            output = method(items)
        return [_coerce_vector(vector) for vector in output]

    def _get_model(self) -> object:
        if self._model is not None:
            return self._model
        module = _import_fastembed_module()
        text_embedding = getattr(module, "TextEmbedding", None)
        if text_embedding is None:  # pragma: no cover - defensive
            raise RuntimeError("fastembed.TextEmbedding is unavailable")
        kwargs: dict[str, object] = {"model_name": self.model_name}
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir
        if self.threads is not None:
            kwargs["threads"] = self.threads
        try:
            self._model = text_embedding(**kwargs)
        except TypeError:
            kwargs.pop("threads", None)
            kwargs.pop("cache_dir", None)
            self._model = text_embedding(**kwargs)
        return self._model


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

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return self.embed(texts)

    def embed_queries(self, texts: Sequence[str]) -> List[List[float]]:
        return self.embed(texts)


class EmbedderFactory:
    """Factory for building embedders from config-like data."""

    @staticmethod
    def from_config(config: object) -> HashingEmbedder | FastEmbedder | HttpEmbedder:
        provider = str(getattr(config, "provider", "local") or "local").lower()
        if provider in {"auto", "fastembed"}:
            model = getattr(config, "model", None) or DEFAULT_FASTEMBED_MODEL
            batch_size = int(getattr(config, "batch_size", 256) or 256)
            cache_dir = getattr(config, "cache_dir", None)
            try:
                _import_fastembed_module()
                return FastEmbedder(
                    model_name=model,
                    batch_size=batch_size,
                    cache_dir=cache_dir,
                )
            except RuntimeError:
                if provider == "fastembed":
                    raise
        if provider in {"auto", "local", "hashing"}:
            dims = getattr(config, "dims", 8) or 8
            return HashingEmbedder(dims=int(dims))
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


def embed_documents(embedder: object, texts: Sequence[str]) -> List[List[float]]:
    method = getattr(embedder, "embed_documents", None)
    if callable(method):
        return [_coerce_vector(vector) for vector in method(texts)]
    method = getattr(embedder, "embed", None)
    if callable(method):
        return [_coerce_vector(vector) for vector in method(texts)]
    raise ValueError("embedder does not expose embed_documents or embed")


def embed_queries(embedder: object, texts: Sequence[str]) -> List[List[float]]:
    method = getattr(embedder, "embed_queries", None)
    if callable(method):
        return [_coerce_vector(vector) for vector in method(texts)]
    method = getattr(embedder, "embed", None)
    if callable(method):
        return [_coerce_vector(vector) for vector in method(texts)]
    raise ValueError("embedder does not expose embed_queries or embed")


def embedder_dims(embedder: object) -> int | None:
    dims = getattr(embedder, "dims", None)
    if isinstance(dims, int) and dims > 0:
        return dims
    return None


def fastembed_model_dims(model_name: str) -> int | None:
    return _FASTEMBED_MODEL_DIMS.get((model_name or "").strip())


def _import_fastembed_module() -> object:
    try:
        return importlib.import_module("fastembed")
    except ImportError as exc:
        raise RuntimeError(
            "fastembed is not installed. Install with `pip install fastembed`."
        ) from exc


def _coerce_vector(vector: Sequence[float] | object) -> List[float]:
    if hasattr(vector, "tolist"):
        values = vector.tolist()
    else:
        values = list(vector)  # type: ignore[arg-type]
    return [float(value) for value in values]
