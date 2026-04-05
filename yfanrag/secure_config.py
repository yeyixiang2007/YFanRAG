"""Encrypted local config storage for desktop UI."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
import base64
import binascii
import ctypes
import getpass
import hashlib
import hmac
import json
import os
import platform
import secrets

from .io_utils import write_text_atomic

try:
    import keyring
    from keyring.errors import KeyringError
except ImportError:  # pragma: no cover - optional dependency
    keyring = None

    class KeyringError(Exception):
        """Fallback keyring error placeholder."""

        pass


_FALLBACK_MAGIC = b"YFSC1"
_FALLBACK_SALT_LEN = 16
_FALLBACK_NONCE_LEN = 16
_FALLBACK_TAG_LEN = 32
_FALLBACK_PBKDF2_ITER = 200_000
_FALLBACK_SECRET_BYTES = 32


@dataclass
class SecureConfigStore:
    """Stores JSON payloads in encrypted form on local disk."""

    path: str | None = None
    app_name: str = "YFanRAG Chat Studio"
    backend: str = "auto"  # auto | dpapi | fallback

    def __post_init__(self) -> None:
        if self.path is None:
            self.path = str(Path.home() / ".yfanrag" / "chat_api_config.enc.json")
        backend = self.backend.strip().lower()
        if backend not in {"auto", "dpapi", "fallback"}:
            raise ValueError(f"unsupported backend: {self.backend}")
        self.backend = backend

    def save(self, payload: Dict[str, Any]) -> None:
        content = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        backend = self._resolve_backend()
        if backend == "dpapi":
            ciphertext = self._encrypt_dpapi(content)
        else:
            ciphertext = self._encrypt_fallback(content)

        record = {
            "version": 1,
            "app": self.app_name,
            "backend": backend,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "ciphertext_b64": base64.b64encode(ciphertext).decode("ascii"),
        }
        target = Path(self.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(
            target,
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def load(self) -> Dict[str, Any]:
        target = Path(self.path)
        if not target.exists():
            return {}

        record = json.loads(target.read_text(encoding="utf-8"))
        backend = str(record.get("backend", "")).strip().lower()
        ciphertext_b64 = str(record.get("ciphertext_b64", "")).strip()
        if not backend or not ciphertext_b64:
            raise ValueError("invalid secure config file format")
        ciphertext = base64.b64decode(ciphertext_b64.encode("ascii"))
        if backend == "dpapi":
            plaintext = self._decrypt_dpapi(ciphertext)
        elif backend == "fallback":
            plaintext = self._decrypt_fallback(ciphertext)
        else:
            raise ValueError(f"unsupported secure config backend: {backend}")

        payload = json.loads(plaintext.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("invalid secure config payload")
        return payload

    def _resolve_backend(self) -> str:
        if self.backend == "dpapi":
            return "dpapi"
        if self.backend == "fallback":
            return "fallback"
        if self._dpapi_available():
            return "dpapi"
        return "fallback"

    @staticmethod
    def _dpapi_available() -> bool:
        return platform.system().lower().startswith("win")

    def _encrypt_fallback(self, plaintext: bytes) -> bytes:
        salt = secrets.token_bytes(_FALLBACK_SALT_LEN)
        nonce = secrets.token_bytes(_FALLBACK_NONCE_LEN)
        enc_key, mac_key = self._fallback_keys(salt, allow_create=True)
        keystream = self._keystream(enc_key, nonce, len(plaintext))
        ciphertext = bytes(a ^ b for a, b in zip(plaintext, keystream))
        tag = hmac.new(mac_key, nonce + ciphertext, hashlib.sha256).digest()
        return _FALLBACK_MAGIC + salt + nonce + tag + ciphertext

    def _decrypt_fallback(self, payload: bytes) -> bytes:
        min_len = (
            len(_FALLBACK_MAGIC)
            + _FALLBACK_SALT_LEN
            + _FALLBACK_NONCE_LEN
            + _FALLBACK_TAG_LEN
        )
        if len(payload) < min_len:
            raise ValueError("encrypted payload too short")
        if not payload.startswith(_FALLBACK_MAGIC):
            raise ValueError("invalid encrypted payload header")
        cursor = len(_FALLBACK_MAGIC)
        salt = payload[cursor : cursor + _FALLBACK_SALT_LEN]
        cursor += _FALLBACK_SALT_LEN
        nonce = payload[cursor : cursor + _FALLBACK_NONCE_LEN]
        cursor += _FALLBACK_NONCE_LEN
        expected_tag = payload[cursor : cursor + _FALLBACK_TAG_LEN]
        cursor += _FALLBACK_TAG_LEN
        ciphertext = payload[cursor:]

        enc_key, mac_key = self._fallback_keys(salt, allow_create=False)
        actual_tag = hmac.new(mac_key, nonce + ciphertext, hashlib.sha256).digest()
        if not hmac.compare_digest(expected_tag, actual_tag):
            raise ValueError("encrypted payload integrity check failed")
        keystream = self._keystream(enc_key, nonce, len(ciphertext))
        return bytes(a ^ b for a, b in zip(ciphertext, keystream))

    @staticmethod
    def _keystream(key: bytes, nonce: bytes, size: int) -> bytes:
        blocks: list[bytes] = []
        counter = 0
        while sum(len(item) for item in blocks) < size:
            counter_bytes = counter.to_bytes(8, byteorder="big", signed=False)
            block = hmac.new(key, nonce + counter_bytes, hashlib.sha256).digest()
            blocks.append(block)
            counter += 1
        stream = b"".join(blocks)
        return stream[:size]

    def _fallback_keys(self, salt: bytes, *, allow_create: bool) -> tuple[bytes, bytes]:
        secret = self._fallback_secret(allow_create=allow_create)
        key_material = hashlib.pbkdf2_hmac(
            "sha256",
            secret,
            salt,
            _FALLBACK_PBKDF2_ITER,
            dklen=64,
        )
        return key_material[:32], key_material[32:]

    def _fallback_secret(self, *, allow_create: bool) -> bytes:
        secret = self._fallback_secret_from_keyring(allow_create=allow_create)
        if secret is not None:
            return secret
        secret = self._fallback_secret_from_file(allow_create=allow_create)
        if secret is not None:
            return secret
        raise ValueError("fallback secret is unavailable")

    def _fallback_secret_from_keyring(self, *, allow_create: bool) -> bytes | None:
        if keyring is None:
            return None
        service_name = f"{self.app_name} SecureConfig"
        account_name = f"{getpass.getuser()}@{platform.node()}"
        try:
            stored = keyring.get_password(service_name, account_name)
            if stored:
                return self._decode_secret(stored)
            if not allow_create:
                return None
            secret = secrets.token_bytes(_FALLBACK_SECRET_BYTES)
            encoded = base64.b64encode(secret).decode("ascii")
            keyring.set_password(service_name, account_name, encoded)
            return secret
        except (KeyringError, ValueError, TypeError, binascii.Error):
            return None

    def _fallback_secret_from_file(self, *, allow_create: bool) -> bytes | None:
        target = self._fallback_secret_path()
        if target.exists():
            return self._decode_secret(target.read_text(encoding="utf-8"))
        if not allow_create:
            return None

        secret = secrets.token_bytes(_FALLBACK_SECRET_BYTES)
        write_text_atomic(
            target,
            base64.b64encode(secret).decode("ascii") + "\n",
            encoding="utf-8",
        )
        try:
            os.chmod(target, 0o600)
        except OSError:
            pass
        return secret

    def _fallback_secret_path(self) -> Path:
        target = Path(self.path)
        return target.parent / f".{target.name}.key"

    @staticmethod
    def _decode_secret(value: str) -> bytes:
        decoded = base64.b64decode(value.strip().encode("ascii"), validate=True)
        if len(decoded) < _FALLBACK_SECRET_BYTES:
            raise ValueError("fallback secret is too short")
        return decoded

    def _encrypt_dpapi(self, plaintext: bytes) -> bytes:
        if not self._dpapi_available():
            raise RuntimeError("dpapi is only available on Windows")
        return _dpapi_protect(plaintext, entropy=self.app_name.encode("utf-8"))

    def _decrypt_dpapi(self, ciphertext: bytes) -> bytes:
        if not self._dpapi_available():
            raise RuntimeError("dpapi is only available on Windows")
        return _dpapi_unprotect(ciphertext, entropy=self.app_name.encode("utf-8"))


def _dpapi_protect(data: bytes, entropy: bytes | None = None) -> bytes:
    return _dpapi_crypt(data, entropy=entropy, protect=True)


def _dpapi_unprotect(data: bytes, entropy: bytes | None = None) -> bytes:
    return _dpapi_crypt(data, entropy=entropy, protect=False)


def _dpapi_crypt(data: bytes, entropy: bytes | None, protect: bool) -> bytes:
    if not platform.system().lower().startswith("win"):
        raise RuntimeError("dpapi is only available on Windows")

    from ctypes import wintypes

    class DATA_BLOB(ctypes.Structure):
        _fields_ = [
            ("cbData", wintypes.DWORD),
            ("pbData", ctypes.POINTER(ctypes.c_byte)),
        ]

    def _make_blob(raw: bytes) -> tuple[DATA_BLOB, ctypes.Array[ctypes.c_byte]]:
        if not raw:
            raw = b"\x00"
        buffer = ctypes.create_string_buffer(raw)
        blob = DATA_BLOB(
            cbData=len(raw),
            pbData=ctypes.cast(buffer, ctypes.POINTER(ctypes.c_byte)),
        )
        return blob, buffer

    in_blob, in_buf = _make_blob(data)
    out_blob = DATA_BLOB()

    entropy_blob = None
    entropy_buf = None
    if entropy:
        entropy_blob, entropy_buf = _make_blob(entropy)

    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    flags = 0x01  # CRYPTPROTECT_UI_FORBIDDEN
    if protect:
        ok = crypt32.CryptProtectData(
            ctypes.byref(in_blob),
            ctypes.c_wchar_p("YFanRAG API Config"),
            ctypes.byref(entropy_blob) if entropy_blob is not None else None,
            None,
            None,
            flags,
            ctypes.byref(out_blob),
        )
    else:
        ok = crypt32.CryptUnprotectData(
            ctypes.byref(in_blob),
            None,
            ctypes.byref(entropy_blob) if entropy_blob is not None else None,
            None,
            None,
            flags,
            ctypes.byref(out_blob),
        )
    if not ok:
        raise RuntimeError("DPAPI encryption/decryption failed")

    try:
        if out_blob.cbData <= 0:
            return b""
        ptr = ctypes.cast(out_blob.pbData, ctypes.POINTER(ctypes.c_ubyte))
        return bytes(ptr[i] for i in range(out_blob.cbData))
    finally:
        if out_blob.pbData:
            kernel32.LocalFree(out_blob.pbData)
        _ = in_buf
        _ = entropy_buf
