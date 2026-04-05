from pathlib import Path
import json

import pytest

import yfanrag.secure_config as secure_config_module
from yfanrag.secure_config import SecureConfigStore


def test_secure_config_roundtrip_fallback(tmp_path: Path) -> None:
    path = tmp_path / "api.enc.json"
    store = SecureConfigStore(path=str(path), backend="fallback")
    payload = {
        "provider_key": "deepseek",
        "endpoint": "https://api.deepseek.com/chat/completions",
        "model": "deepseek-chat",
        "api_key": "sk-secret-123",
        "temperature": "0.2",
    }

    store.save(payload)
    assert path.exists()

    raw = path.read_text(encoding="utf-8")
    assert "sk-secret-123" not in raw
    assert "deepseek-chat" not in raw

    loaded = store.load()
    assert loaded == payload


def test_secure_config_tamper_detected(tmp_path: Path) -> None:
    path = tmp_path / "api.enc.json"
    store = SecureConfigStore(path=str(path), backend="fallback")
    store.save({"api_key": "sk-secret-abc"})

    record = json.loads(path.read_text(encoding="utf-8"))
    cipher = record["ciphertext_b64"]
    if cipher.endswith("A"):
        cipher = cipher[:-1] + "B"
    else:
        cipher = cipher[:-1] + "A"
    record["ciphertext_b64"] = cipher
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(Exception):
        store.load()


def test_secure_config_missing_file_returns_empty(tmp_path: Path) -> None:
    path = tmp_path / "not_exists.enc.json"
    store = SecureConfigStore(path=str(path), backend="fallback")
    assert store.load() == {}


def test_secure_config_fallback_uses_local_secret_when_keyring_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(secure_config_module, "keyring", None)
    path = tmp_path / "api.enc.json"
    store = SecureConfigStore(path=str(path), backend="fallback")

    store.save({"api_key": "sk-test"})

    assert (tmp_path / ".api.enc.json.key").exists()
    assert store.load() == {"api_key": "sk-test"}
