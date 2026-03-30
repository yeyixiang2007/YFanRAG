from yfanrag.config import YFanRAGConfig


def test_config_roundtrip():
    config = YFanRAGConfig()
    data = config.to_dict()
    loaded = YFanRAGConfig.from_dict(data)
    assert loaded.chunking.chunk_size == config.chunking.chunk_size
    assert loaded.embedding.provider == config.embedding.provider
