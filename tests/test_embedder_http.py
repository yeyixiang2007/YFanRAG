import json
import urllib.request

import pytest

from yfanrag.config import EmbeddingConfig
from yfanrag.embedders import EmbedderFactory, HashingEmbedder, HttpEmbedder


class DummyResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_embedder_factory_local():
    config = EmbeddingConfig(provider="local", dims=6)
    embedder = EmbedderFactory.from_config(config)
    assert isinstance(embedder, HashingEmbedder)
    assert embedder.dims == 6


def test_http_embedder_reads_embeddings(monkeypatch):
    payload = {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}

    def fake_urlopen(request, timeout=0):
        assert request.get_method() == "POST"
        return DummyResponse(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    embedder = HttpEmbedder(endpoint="http://example.com")
    result = embedder.embed(["a", "b"])
    assert result == payload["embeddings"]


def test_http_embedder_reads_data_format(monkeypatch):
    payload = {"data": [{"embedding": [1.0, 0.0]}]}

    def fake_urlopen(request, timeout=0):
        return DummyResponse(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    embedder = HttpEmbedder(endpoint="http://example.com")
    result = embedder.embed(["x"])
    assert result == [[1.0, 0.0]]


def test_http_embedder_requires_endpoint():
    with pytest.raises(ValueError):
        HttpEmbedder(endpoint="").embed(["x"])
