"""Request pipeline, queue polling, and chat rendering."""

from __future__ import annotations

from queue import Empty
import threading
from typing import Sequence
from tkinter import END

from ...chat_providers import ChatApiSettings, ChatMessage, parse_json_dict
from ..events import WorkerEvent
from ..markdown import (
    MARKDOWN_FENCE_RE,
    MARKDOWN_HEADING_RE,
    MARKDOWN_HR_RE,
    MARKDOWN_INLINE_TOKEN_RE,
    MARKDOWN_LINK_RE,
    MARKDOWN_ORDERED_LIST_RE,
    MARKDOWN_QUOTE_RE,
    MARKDOWN_UNORDERED_LIST_RE,
)


class AppChatMixin:
    def _on_stop(self) -> None:
        if not self.pending:
            return
        self.stop_requested = True
        self._set_status("Stopping...", tone="warn")

    def _on_send(self) -> None:
        if self.pending:
            return
        user_text = self.input_text.get("1.0", END).strip()
        if not user_text:
            return

        outbound_text = user_text
        kb_note = ""
        try:
            outbound_text, kb_note = self._build_kb_context_for_user_text(user_text)
        except Exception as exc:
            self._append_log("error", f"Knowledge base context failed: {exc}")
            self._set_status("KB context failed", tone="warn")
            outbound_text = user_text
            kb_note = ""

        self.input_text.delete("1.0", END)
        self._append_log("user", user_text)
        if kb_note:
            self._append_log("system", kb_note)
        self.messages.append(ChatMessage(role="user", content=outbound_text))

        try:
            settings = self._collect_settings()
        except Exception as exc:
            self._append_log("error", f"Invalid settings: {exc}")
            self._set_status("Invalid settings", tone="error")
            return
        self._save_api_config(verbose=False)

        payload_messages = list(self.messages)
        self.stop_requested = False
        self._start_stream_placeholder()
        self._set_pending(True)
        self._set_status("Request in flight...", tone="warn")

        stream_enabled = bool(self.stream_var.get())
        thread = threading.Thread(
            target=self._request_worker,
            args=(settings, payload_messages, stream_enabled),
            daemon=True,
        )
        thread.start()

    def _collect_settings(self) -> ChatApiSettings:
        provider = self._provider_key()
        endpoint = self.endpoint_var.get().strip()
        model = self.model_var.get().strip()
        api_key = self.api_key_var.get().strip()
        api_key_header = self.api_key_header_var.get().strip()

        temperature_raw = self.temperature_var.get().strip()
        temperature = float(temperature_raw) if temperature_raw else None

        max_tokens_raw = self.max_tokens_var.get().strip()
        max_tokens = int(max_tokens_raw) if max_tokens_raw else None

        timeout_raw = self.timeout_var.get().strip()
        timeout_seconds = int(timeout_raw) if timeout_raw else 120

        system_prompt = self.system_text.get("1.0", END).strip()
        extra_headers = parse_json_dict(
            self.extra_headers_text.get("1.0", END),
            "Extra Headers",
        )
        extra_body = parse_json_dict(
            self.extra_body_text.get("1.0", END),
            "Extra Body",
        )

        self._refresh_provider_meta()
        return ChatApiSettings(
            provider=provider,
            endpoint=endpoint,
            model=model,
            api_key=api_key,
            api_key_header=api_key_header,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            extra_headers={str(k): str(v) for k, v in extra_headers.items()},
            extra_body=extra_body,
        )

    def _request_worker(
        self,
        settings: ChatApiSettings,
        messages: list[ChatMessage],
        stream_enabled: bool,
    ) -> None:
        try:
            if stream_enabled:
                text, stopped = self.client.chat_stream(
                    settings=settings,
                    messages=messages,
                    on_delta=lambda chunk: self.queue.put(
                        WorkerEvent(kind="delta", text=chunk)
                    ),
                    should_stop=lambda: self.stop_requested,
                )
                self.queue.put(WorkerEvent(kind="done", text=text, stopped=stopped))
            else:
                text = self.client.chat(settings=settings, messages=messages)
                self.queue.put(WorkerEvent(kind="done", text=text, stopped=False))
        except Exception as exc:
            self.queue.put(WorkerEvent(kind="error", text=str(exc), stopped=False))

    def _poll_queue(self) -> None:
        try:
            while True:
                event = self.queue.get_nowait()
                if event.kind == "delta":
                    self._append_stream_delta(event.text)
                    continue

                self._set_pending(False)
                if event.kind == "done":
                    self._finalize_stream_message(event.text)
                    if event.stopped:
                        self._set_status("Stopped", tone="warn")
                    else:
                        self._set_status("Response received", tone="ok")
                    self._focus_input()
                    continue

                if event.kind == "error":
                    self._finalize_stream_message("")
                    self._append_log("error", event.text)
                    self._set_status("Request failed", tone="error")
                    self._focus_input()
        except Empty:
            pass
        self.root.after(80, self._poll_queue)

    def _start_stream_placeholder(self) -> None:
        self.transcript.append({"role": "assistant", "text": ""})
        self.stream_message_index = len(self.transcript) - 1
        self._render_chat()

    def _append_stream_delta(self, delta: str) -> None:
        if not delta:
            return
        if self.stream_message_index is None:
            self._start_stream_placeholder()
        if self.stream_message_index is None:
            return
        self.transcript[self.stream_message_index]["text"] += delta
        self._render_chat()

    def _finalize_stream_message(self, final_text: str) -> None:
        if self.stream_message_index is None:
            return
        idx = self.stream_message_index
        existing = self.transcript[idx]["text"].strip()
        fallback = final_text.strip()
        text = existing or fallback
        self.transcript[idx]["text"] = text
        self.stream_message_index = None
        self._render_chat()
        if text:
            self.messages.append(ChatMessage(role="assistant", content=text))

    def _append_log(self, role: str, text: str) -> None:
        self.transcript.append({"role": role, "text": text})
        self._render_chat()

    def _chat_insert(self, text: str, *tags: str) -> None:
        active_tags = tuple(tag for tag in tags if tag)
        if active_tags:
            self.chat_text.insert(END, text, active_tags)
        else:
            self.chat_text.insert(END, text)

    def _render_markdown(self, role_tag: str, text: str) -> None:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = normalized.split("\n")
        in_code_block = False

        for line in lines:
            fence_match = MARKDOWN_FENCE_RE.match(line)
            if fence_match:
                if in_code_block:
                    in_code_block = False
                else:
                    in_code_block = True
                    language = (fence_match.group(1) or "").strip()
                    if language:
                        self._chat_insert(f"[{language}]\n", role_tag, "md_code_meta")
                continue

            if in_code_block:
                self._chat_insert(f"{line}\n", role_tag, "md_code_block")
                continue

            if not line.strip():
                self._chat_insert("\n", role_tag)
                continue

            heading_match = MARKDOWN_HEADING_RE.match(line)
            if heading_match:
                level = min(len(heading_match.group(1)), 4)
                heading_text = heading_match.group(2).strip() or " "
                self._chat_insert(f"{heading_text}\n", role_tag, f"md_h{level}")
                continue

            if MARKDOWN_HR_RE.match(line):
                self._chat_insert("----------------------------------------\n", role_tag, "md_hr")
                continue

            quote_match = MARKDOWN_QUOTE_RE.match(line)
            if quote_match:
                self._chat_insert("| ", role_tag, "md_quote_marker")
                self._render_markdown_inline(
                    role_tag,
                    quote_match.group(1),
                    extra_tags=("md_quote",),
                )
                self._chat_insert("\n", role_tag, "md_quote")
                continue

            unordered_match = MARKDOWN_UNORDERED_LIST_RE.match(line)
            if unordered_match:
                indent = unordered_match.group(1).replace("\t", "    ")
                self._chat_insert(f"{indent}- ", role_tag, "md_list_marker")
                self._render_markdown_inline(
                    role_tag,
                    unordered_match.group(2),
                    extra_tags=("md_list",),
                )
                self._chat_insert("\n", role_tag, "md_list")
                continue

            ordered_match = MARKDOWN_ORDERED_LIST_RE.match(line)
            if ordered_match:
                indent = ordered_match.group(1).replace("\t", "    ")
                index = ordered_match.group(2)
                self._chat_insert(f"{indent}{index}. ", role_tag, "md_list_marker")
                self._render_markdown_inline(
                    role_tag,
                    ordered_match.group(3),
                    extra_tags=("md_list",),
                )
                self._chat_insert("\n", role_tag, "md_list")
                continue

            self._render_markdown_inline(role_tag, line)
            self._chat_insert("\n", role_tag)

    def _render_markdown_inline(
        self,
        role_tag: str,
        text: str,
        extra_tags: tuple[str, ...] = (),
    ) -> None:
        cursor = 0
        for match in MARKDOWN_INLINE_TOKEN_RE.finditer(text):
            start, end = match.span()
            if start > cursor:
                self._chat_insert(text[cursor:start], role_tag, *extra_tags)
            self._render_markdown_token(role_tag, match.group(0), extra_tags)
            cursor = end
        if cursor < len(text):
            self._chat_insert(text[cursor:], role_tag, *extra_tags)

    def _render_markdown_token(
        self,
        role_tag: str,
        token: str,
        extra_tags: tuple[str, ...],
    ) -> None:
        base_tags = (role_tag, *extra_tags)

        if token.startswith("`") and token.endswith("`"):
            self._chat_insert(token[1:-1], *base_tags, "md_code_inline")
            return
        if token.startswith("~~") and token.endswith("~~"):
            self._chat_insert(token[2:-2], *base_tags, "md_strike")
            return

        link_match = MARKDOWN_LINK_RE.match(token)
        if link_match is not None:
            label, url = link_match.groups()
            self._chat_insert(label, *base_tags, "md_link")
            self._chat_insert(f" ({url})", *base_tags, "md_link_url")
            return

        if (token.startswith("***") and token.endswith("***")) or (
            token.startswith("___") and token.endswith("___")
        ):
            self._chat_insert(token[3:-3], *base_tags, "md_bold_italic")
            return
        if (token.startswith("**") and token.endswith("**")) or (
            token.startswith("__") and token.endswith("__")
        ):
            self._chat_insert(token[2:-2], *base_tags, "md_bold")
            return
        if (token.startswith("*") and token.endswith("*")) or (
            token.startswith("_") and token.endswith("_")
        ):
            self._chat_insert(token[1:-1], *base_tags, "md_italic")
            return

        self._chat_insert(token, *base_tags)

    def _render_chat(self) -> None:
        self.chat_text.configure(state="normal")
        self.chat_text.delete("1.0", END)
        for item in self.transcript:
            role = item["role"]
            text = item["text"]
            role_display = role.upper()
            self.chat_text.insert(END, f"{role_display}\n", ("title",))
            tag = role if role in {"user", "assistant", "system", "error"} else "system"
            body = text if text.strip() else "…"
            self._render_markdown(tag, body)
            self._chat_insert("\n", tag)
        self.chat_text.configure(state="disabled")
        self.chat_text.see(END)
