from yfanrag.observability import DEFAULT_SLOW_QUERY_MS, slow_query_threshold_ms


def test_slow_query_threshold_default(monkeypatch):
    monkeypatch.delenv("YFANRAG_SLOW_QUERY_MS", raising=False)
    assert slow_query_threshold_ms() == DEFAULT_SLOW_QUERY_MS


def test_slow_query_threshold_from_env(monkeypatch):
    monkeypatch.setenv("YFANRAG_SLOW_QUERY_MS", "123.5")
    assert slow_query_threshold_ms() == 123.5
