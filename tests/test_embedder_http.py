import json
import types
import urllib.request

import pytest

from yfanrag.config import EmbeddingConfig
import yfanrag.embedders as embedders_module
from yfanrag.embedders import EmbedderFactory, FastEmbedder, HashingEmbedder, HttpEmbedder


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


def test_embedder_factory_auto_falls_back_to_hashing(monkeypatch):
    def _missing_fastembed():
        raise RuntimeError("fastembed missing")

    monkeypatch.setattr(embedders_module, "_import_fastembed_module", _missing_fastembed)

    config = EmbeddingConfig(provider="auto", dims=12)
    embedder = EmbedderFactory.from_config(config)

    assert isinstance(embedder, HashingEmbedder)
    assert embedder.dims == 12


def test_fastembedder_uses_query_and_passage_methods(monkeypatch):
    class DummyTextEmbedding:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def passage_embed(self, texts, batch_size=None):
            return [[float(len(text)), float(batch_size or 0)] for text in texts]

        def query_embed(self, texts, batch_size=None):
            return [[float(len(text) * 10), float(batch_size or 0)] for text in texts]

    monkeypatch.setattr(
        embedders_module,
        "_import_fastembed_module",
        lambda: types.SimpleNamespace(TextEmbedding=DummyTextEmbedding),
    )

    embedder = FastEmbedder(model_name="BAAI/bge-small-en-v1.5", batch_size=32)

    assert embedder.embed_documents(["abc"]) == [[3.0, 32.0]]
    assert embedder.embed_queries(["abc"]) == [[30.0, 32.0]]


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
