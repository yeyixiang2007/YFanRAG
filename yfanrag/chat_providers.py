"""Provider adapters for real-model chat APIs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Sequence
import json
import urllib.error
import urllib.request


PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "openai_compatible": {
        "label": "OpenAI-Compatible",
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-4.1-mini",
        "api_key_header": "",
    },
    "deepseek": {
        "label": "DeepSeek (OpenAI-Compatible)",
        "endpoint": "https://api.deepseek.com/chat/completions",
        "model": "deepseek-chat",
        "api_key_header": "",
    },
    "openai_responses": {
        "label": "OpenAI Responses",
        "endpoint": "https://api.openai.com/v1/responses",
        "model": "gpt-4.1-mini",
        "api_key_header": "",
    },
    "anthropic": {
        "label": "Anthropic Messages",
        "endpoint": "https://api.anthropic.com/v1/messages",
        "model": "claude-3-5-sonnet-latest",
        "api_key_header": "x-api-key",
    },
}


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass
class ChatApiSettings:
    provider: str = "openai_compatible"
    endpoint: str = PROVIDER_PRESETS["openai_compatible"]["endpoint"]
    model: str = PROVIDER_PRESETS["openai_compatible"]["model"]
    api_key: str = ""
    api_key_header: str = ""
    system_prompt: str = ""
    temperature: float | None = 0.2
    max_tokens: int | None = 1024
    timeout_seconds: int = 120
    anthropic_version: str = "2023-06-01"
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)


class ChatApiClient:
    """HTTP client that adapts payload/response across common providers."""

    def build_request(
        self,
        settings: ChatApiSettings,
        messages: Sequence[ChatMessage],
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        provider = settings.provider.strip()
        if provider in {"openai_compatible", "deepseek"}:
            return self._build_openai_chat_request(settings, messages)
        if provider == "openai_responses":
            return self._build_openai_responses_request(settings, messages)
        if provider == "anthropic":
            return self._build_anthropic_request(settings, messages)
        raise ValueError(f"unsupported provider: {provider}")

    def chat(self, settings: ChatApiSettings, messages: Sequence[ChatMessage]) -> str:
        endpoint, headers, payload = self.build_request(settings, messages)
        response = self._post_json(
            endpoint=endpoint,
            headers=headers,
            payload=payload,
            timeout_seconds=settings.timeout_seconds,
        )
        text = self.extract_text(settings.provider, response)
        if not text.strip():
            raise RuntimeError("empty assistant response")
        return text

    def chat_stream(
        self,
        settings: ChatApiSettings,
        messages: Sequence[ChatMessage],
        on_delta: Callable[[str], None],
        should_stop: Callable[[], bool] | None = None,
    ) -> tuple[str, bool]:
        endpoint, headers, payload = self.build_request(settings, messages)
        stream_payload = dict(payload)
        stream_payload["stream"] = True

        parts: List[str] = []
        stop_check = should_stop or (lambda: False)

        def _on_event(event: Mapping[str, Any]) -> None:
            delta = self.extract_stream_delta(settings.provider, event)
            if delta:
                parts.append(delta)
                on_delta(delta)

        stopped = self._post_stream_sse(
            endpoint=endpoint,
            headers=headers,
            payload=stream_payload,
            timeout_seconds=settings.timeout_seconds,
            on_event=_on_event,
            should_stop=stop_check,
        )

        if not parts and not stopped:
            # Fallback for providers that do not expose SSE chunks in the expected format.
            full_text = self.chat(settings=settings, messages=messages)
            if full_text:
                on_delta(full_text)
            return full_text, False

        return "".join(parts), stopped

    def extract_text(self, provider: str, response: Mapping[str, Any]) -> str:
        if "error" in response:
            raise RuntimeError(_json_dumps(response["error"]))

        if provider == "anthropic":
            content = response.get("content")
            if isinstance(content, list):
                parts = [_flatten_content(item) for item in content]
                return "\n".join(part for part in parts if part.strip()).strip()

        output_text = response.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, Mapping):
                message = first.get("message")
                if isinstance(message, Mapping):
                    return _flatten_content(message.get("content")).strip()
                text = first.get("text")
                if isinstance(text, str):
                    return text.strip()

        output = response.get("output")
        if isinstance(output, list):
            chunks = [_flatten_content(item) for item in output]
            merged = "\n".join(chunk for chunk in chunks if chunk.strip()).strip()
            if merged:
                return merged

        content = response.get("content")
        if isinstance(content, (str, list, Mapping)):
            merged = _flatten_content(content).strip()
            if merged:
                return merged

        message = response.get("message")
        if isinstance(message, Mapping):
            merged = _flatten_content(message.get("content")).strip()
            if merged:
                return merged

        return _json_dumps(response)

    def extract_stream_delta(self, provider: str, event: Mapping[str, Any]) -> str:
        if provider in {"openai_compatible", "deepseek"}:
            choices = event.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, Mapping):
                    delta = first.get("delta")
                    if isinstance(delta, Mapping):
                        return _flatten_content(delta.get("content")).strip()
                    message = first.get("message")
                    if isinstance(message, Mapping):
                        return _flatten_content(message.get("content")).strip()
            return ""

        if provider == "openai_responses":
            event_type = str(event.get("type") or "")
            if event_type in {
                "response.output_text.delta",
                "response.refusal.delta",
            }:
                return str(event.get("delta") or "")
            delta = event.get("delta")
            if isinstance(delta, str):
                return delta
            if "output_text" in event and isinstance(event["output_text"], str):
                return event["output_text"]
            return ""

        if provider == "anthropic":
            event_type = str(event.get("type") or event.get("_event") or "")
            if event_type == "content_block_delta":
                delta = event.get("delta")
                if isinstance(delta, Mapping) and delta.get("type") == "text_delta":
                    return str(delta.get("text") or "")
            if event_type == "content_block_start":
                block = event.get("content_block")
                if isinstance(block, Mapping):
                    return str(block.get("text") or "")
            return ""

        return ""

    def _build_openai_chat_request(
        self,
        settings: ChatApiSettings,
        messages: Sequence[ChatMessage],
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        endpoint = _validate_endpoint(settings.endpoint)
        normalized = _normalize_messages(messages)
        if settings.system_prompt.strip():
            normalized = [
                ChatMessage(role="system", content=settings.system_prompt.strip()),
                *normalized,
            ]

        payload: dict[str, Any] = {
            "model": settings.model.strip(),
            "messages": [
                {"role": msg.role, "content": msg.content}
                for msg in normalized
            ],
        }
        if settings.temperature is not None:
            payload["temperature"] = float(settings.temperature)
        if settings.max_tokens is not None:
            payload["max_tokens"] = int(settings.max_tokens)
        payload.update(settings.extra_body)

        headers = _build_base_headers(settings, default_key_header="Authorization")
        return endpoint, headers, payload

    def _build_openai_responses_request(
        self,
        settings: ChatApiSettings,
        messages: Sequence[ChatMessage],
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        endpoint = _validate_endpoint(settings.endpoint)
        normalized = _normalize_messages(messages)
        if settings.system_prompt.strip():
            normalized = [
                ChatMessage(role="system", content=settings.system_prompt.strip()),
                *normalized,
            ]

        payload: dict[str, Any] = {
            "model": settings.model.strip(),
            "input": [
                {
                    "role": msg.role,
                    "content": [{"type": "input_text", "text": msg.content}],
                }
                for msg in normalized
            ],
        }
        if settings.temperature is not None:
            payload["temperature"] = float(settings.temperature)
        if settings.max_tokens is not None:
            payload["max_output_tokens"] = int(settings.max_tokens)
        payload.update(settings.extra_body)

        headers = _build_base_headers(settings, default_key_header="Authorization")
        return endpoint, headers, payload

    def _build_anthropic_request(
        self,
        settings: ChatApiSettings,
        messages: Sequence[ChatMessage],
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        endpoint = _validate_endpoint(settings.endpoint)
        normalized = _normalize_messages(messages)
        conversational = [msg for msg in normalized if msg.role in {"user", "assistant"}]
        if not conversational:
            raise ValueError("anthropic request requires at least one user/assistant message")

        payload: dict[str, Any] = {
            "model": settings.model.strip(),
            "messages": [
                {"role": msg.role, "content": msg.content}
                for msg in conversational
            ],
            "max_tokens": int(settings.max_tokens or 1024),
        }
        if settings.system_prompt.strip():
            payload["system"] = settings.system_prompt.strip()
        if settings.temperature is not None:
            payload["temperature"] = float(settings.temperature)
        payload.update(settings.extra_body)

        headers = _build_base_headers(settings, default_key_header="x-api-key")
        headers.setdefault("anthropic-version", settings.anthropic_version.strip() or "2023-06-01")
        return endpoint, headers, payload

    def _post_json(
        self,
        endpoint: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=body,
            headers=dict(headers),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"http error {exc.code} {exc.reason}: {detail[:1000]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"network error: {exc}") from exc

        try:
            payload_obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid json response: {raw[:1000]}") from exc
        if not isinstance(payload_obj, dict):
            raise RuntimeError("api response must be json object")
        return payload_obj

    def _post_stream_sse(
        self,
        endpoint: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: int,
        on_event: Callable[[Mapping[str, Any]], None],
        should_stop: Callable[[], bool],
    ) -> bool:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        stream_headers = dict(headers)
        stream_headers.setdefault("Accept", "text/event-stream")
        request = urllib.request.Request(
            endpoint,
            data=body,
            headers=stream_headers,
            method="POST",
        )

        stopped = False
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                event_name = ""
                data_lines: List[str] = []
                for raw_line in response:
                    if should_stop():
                        stopped = True
                        break

                    line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                    if line.startswith("event:"):
                        event_name = line[len("event:") :].strip()
                        continue
                    if line.startswith("data:"):
                        data_lines.append(line[len("data:") :].strip())
                        continue
                    if line.strip():
                        continue

                    if data_lines:
                        stopped = self._consume_sse_event(
                            event_name=event_name,
                            data_lines=data_lines,
                            on_event=on_event,
                            should_stop=should_stop,
                        )
                        if stopped:
                            break
                        data_lines = []
                    event_name = ""

                if not stopped and data_lines:
                    stopped = self._consume_sse_event(
                        event_name=event_name,
                        data_lines=data_lines,
                        on_event=on_event,
                        should_stop=should_stop,
                    )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"http error {exc.code} {exc.reason}: {detail[:1000]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"network error: {exc}") from exc
        return stopped

    @staticmethod
    def _consume_sse_event(
        event_name: str,
        data_lines: Sequence[str],
        on_event: Callable[[Mapping[str, Any]], None],
        should_stop: Callable[[], bool],
    ) -> bool:
        if should_stop():
            return True

        raw_payload = "\n".join(data_lines).strip()
        if not raw_payload:
            return False
        if raw_payload == "[DONE]":
            return False

        try:
            event_obj = json.loads(raw_payload)
        except json.JSONDecodeError:
            return False
        if not isinstance(event_obj, dict):
            return False
        if event_name and "type" not in event_obj:
            event_obj["_event"] = event_name
        on_event(event_obj)
        return False


def parse_json_dict(raw: str, field_name: str) -> dict[str, Any]:
    text = raw.strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be valid JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be JSON object")
    return value


def _build_base_headers(
    settings: ChatApiSettings,
    default_key_header: str,
) -> dict[str, str]:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    key = settings.api_key.strip()
    key_header = settings.api_key_header.strip()

    if key:
        if key_header:
            headers[key_header] = key
        elif default_key_header.lower() == "authorization":
            headers["Authorization"] = f"Bearer {key}"
        else:
            headers[default_key_header] = key

    for header_name, header_value in settings.extra_headers.items():
        headers[str(header_name)] = str(header_value)
    return headers


def _validate_endpoint(endpoint: str) -> str:
    value = endpoint.strip()
    if not value:
        raise ValueError("endpoint is required")
    if not value.startswith("http://") and not value.startswith("https://"):
        raise ValueError("endpoint must start with http:// or https://")
    return value


def _normalize_messages(messages: Sequence[ChatMessage]) -> List[ChatMessage]:
    normalized: List[ChatMessage] = []
    for msg in messages:
        role = str(msg.role).strip().lower()
        if role not in {"system", "user", "assistant"}:
            role = "user"
        content = str(msg.content or "").strip()
        if not content:
            continue
        normalized.append(ChatMessage(role=role, content=content))
    if not normalized:
        raise ValueError("messages cannot be empty")
    return normalized


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        for key in ("text", "content", "value", "output_text"):
            if key in content:
                return _flatten_content(content[key])
        return _json_dumps(content)
    if isinstance(content, list):
        parts = [_flatten_content(item) for item in content]
        return "\n".join(part for part in parts if part.strip())
    return str(content)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)
