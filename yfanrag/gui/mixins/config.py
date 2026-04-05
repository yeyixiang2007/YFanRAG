"""API settings persistence helpers for the Tk chat app."""

from __future__ import annotations

from tkinter import END

from ...chat_providers import PROVIDER_PRESETS, parse_json_dict
from ...knowledge_base import QUERY_MODE_CHOICES, STORE_CHOICES


class AppConfigMixin:
    def _collect_api_config_payload(self) -> dict[str, object]:
        return {
            "version": 1,
            "provider_key": self._provider_key(),
            "endpoint": self.endpoint_var.get(),
            "model": self.model_var.get(),
            "api_key": self.api_key_var.get(),
            "api_key_header": self.api_key_header_var.get(),
            "temperature": self.temperature_var.get(),
            "max_tokens": self.max_tokens_var.get(),
            "timeout_seconds": self.timeout_var.get(),
            "stream": bool(self.stream_var.get()),
            "system_prompt": self.system_text.get("1.0", END),
            "extra_headers_json": self.extra_headers_text.get("1.0", END),
            "extra_body_json": self.extra_body_text.get("1.0", END),
            "kb": {
                "db_path": self.kb_db_var.get(),
                "store": self.kb_store_var.get(),
                "enable_fts": bool(self.kb_enable_fts_var.get()),
                "chunker": self.kb_chunker_var.get(),
                "chunk_size": self.kb_chunk_size_var.get(),
                "chunk_overlap": self.kb_chunk_overlap_var.get(),
                "dims": self.kb_dims_var.get(),
                "query_mode": self.kb_query_mode_var.get(),
                "query_top_k": self.kb_top_k_var.get(),
                "use_context_in_chat": bool(self.kb_use_context_var.get()),
                "context_top_k": self.kb_context_top_k_var.get(),
            },
        }

    def _apply_api_config_payload(self, payload: dict[str, object]) -> None:
        provider_key = str(payload.get("provider_key", "")).strip() or "openai_compatible"
        if provider_key not in PROVIDER_PRESETS:
            provider_key = "openai_compatible"
        self.provider_display_var.set(self._display_for_provider(provider_key))
        endpoint = payload.get("endpoint")
        if isinstance(endpoint, str):
            self.endpoint_var.set(endpoint)
        model = payload.get("model")
        if isinstance(model, str):
            self.model_var.set(model)
        api_key = payload.get("api_key")
        if isinstance(api_key, str):
            self.api_key_var.set(api_key)
        api_key_header = payload.get("api_key_header")
        if isinstance(api_key_header, str):
            self.api_key_header_var.set(api_key_header)
        temperature = payload.get("temperature")
        if isinstance(temperature, str):
            self.temperature_var.set(temperature)
        max_tokens = payload.get("max_tokens")
        if isinstance(max_tokens, str):
            self.max_tokens_var.set(max_tokens)
        timeout_seconds = payload.get("timeout_seconds")
        if isinstance(timeout_seconds, str):
            self.timeout_var.set(timeout_seconds)
        self.stream_var.set(bool(payload.get("stream", self.stream_var.get())))

        system_prompt = payload.get("system_prompt")
        if isinstance(system_prompt, str):
            self.system_text.delete("1.0", END)
            self.system_text.insert("1.0", system_prompt)

        extra_headers_json = payload.get("extra_headers_json")
        if isinstance(extra_headers_json, str):
            self.extra_headers_text.delete("1.0", END)
            self.extra_headers_text.insert("1.0", extra_headers_json)

        extra_body_json = payload.get("extra_body_json")
        if isinstance(extra_body_json, str):
            self.extra_body_text.delete("1.0", END)
            self.extra_body_text.insert("1.0", extra_body_json)

        kb_payload = payload.get("kb")
        if isinstance(kb_payload, dict):
            db_path = kb_payload.get("db_path")
            if isinstance(db_path, str) and db_path.strip():
                self.kb_db_var.set(db_path)

            store = kb_payload.get("store")
            if isinstance(store, str) and store in STORE_CHOICES:
                self.kb_store_var.set(store)

            chunker = kb_payload.get("chunker")
            if isinstance(chunker, str) and chunker in {"fixed", "recursive", "structured"}:
                self.kb_chunker_var.set(chunker)

            query_mode = kb_payload.get("query_mode")
            if isinstance(query_mode, str) and query_mode in QUERY_MODE_CHOICES:
                self.kb_query_mode_var.set(query_mode)

            for field, var in (
                ("chunk_size", self.kb_chunk_size_var),
                ("chunk_overlap", self.kb_chunk_overlap_var),
                ("dims", self.kb_dims_var),
                ("query_top_k", self.kb_top_k_var),
                ("context_top_k", self.kb_context_top_k_var),
            ):
                value = kb_payload.get(field)
                if isinstance(value, int):
                    var.set(str(value))
                elif isinstance(value, str) and value.strip():
                    var.set(value)

            enable_fts = kb_payload.get("enable_fts")
            if isinstance(enable_fts, bool):
                self.kb_enable_fts_var.set(enable_fts)

            use_context = kb_payload.get("use_context_in_chat")
            if isinstance(use_context, bool):
                self.kb_use_context_var.set(use_context)
        self._refresh_provider_meta()

    def _save_api_config(self, verbose: bool) -> None:
        try:
            payload = self._collect_api_config_payload()
            self.config_store.save(payload)
            if verbose:
                self._set_status("API config saved (encrypted)", tone="ok")
                self._append_log("system", "Saved encrypted API config to local disk.")
        except Exception as exc:
            if verbose:
                self._append_log("error", f"Save API config failed: {exc}")
            self._set_status("Save API config failed", tone="error")

    def _load_api_config(self, announce: bool) -> None:
        try:
            payload = self.config_store.load()
        except Exception as exc:
            if announce:
                self._append_log("error", f"Load API config failed: {exc}")
            self._set_status("Load API config failed", tone="warn")
            return

        if not payload:
            if announce:
                self._append_log("system", "No local API config found.")
            return
        try:
            self._apply_api_config_payload(payload)
        except Exception as exc:
            if announce:
                self._append_log("error", f"Apply API config failed: {exc}")
            self._set_status("Apply API config failed", tone="warn")
            return
        self._set_status("Encrypted API config loaded", tone="ok")
        if announce:
            self._append_log("system", "Loaded encrypted API config from local disk.")
