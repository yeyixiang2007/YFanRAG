"""Request pipeline, queue polling, and chat rendering."""

from __future__ import annotations

from pathlib import Path
from queue import Empty
import threading
import re
from typing import Sequence
from tkinter import END

from ...benchmark import RetrievalItem
from ...chat_providers import ChatApiSettings, ChatMessage, parse_json_dict
from ...feedback_loop import FeedbackRecord, FeedbackReference
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
    _TRACE_CONFIDENCE_RE = re.compile(
        r"(?:证据\s*[：:]\s*(?:充分|不足)|evidence\s*[:：]\s*(?:sufficient|insufficient))",
        re.IGNORECASE,
    )
    _TRACE_SOURCE_RE = re.compile(r"(?:doc_id\s*[=:]|chunk_id\s*[=:])", re.IGNORECASE)
    _LOW_CONFIDENCE_HINT_RE = re.compile(
        r"(?:证据不足|不确定|无法确认|未知|可能|insufficient|not sure|unknown|uncertain)",
        re.IGNORECASE,
    )
    _TRACE_ID_TOKEN_RE = re.compile(r"(doc_id|chunk_id)\s*[=:]\s*", re.IGNORECASE)
    _TRACE_LABEL_RE = re.compile(r"^\[(?P<label>[^\]\n]+)\]\s*")
    _TRACE_SOURCE_HEADING_RE = re.compile(r"^\s*(?:引用来源|sources?)\s*[:：]\s*$", re.IGNORECASE)

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

        self._set_feedback_target(None)
        outbound_text = user_text
        kb_note = ""
        self._set_kb_traceability_state(False, ())
        try:
            outbound_text, kb_note = self._build_kb_context_for_user_text(user_text)
        except Exception as exc:
            self._append_log("error", f"Knowledge base context failed: {exc}")
            self._set_status("KB context failed", tone="warn")
            outbound_text = user_text
            kb_note = ""
            self._set_kb_traceability_state(False, ())
            self._clear_kb_feedback_context()

        self.input_text.delete("1.0", END)
        self._append_log("user", user_text)
        if kb_note:
            self._append_log("system", kb_note)
        self.messages.append(ChatMessage(role="user", content=outbound_text))
        self.pending_feedback_context = self._build_pending_feedback_context(user_text)

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
        feedback_context = dict(self.pending_feedback_context or {})
        idx = self.stream_message_index
        existing = self.transcript[idx]["text"].strip()
        fallback = final_text.strip()
        text = existing or fallback
        text = self._ensure_traceable_kb_answer(text)
        self.transcript[idx]["text"] = text
        self.stream_message_index = None
        self._set_kb_traceability_state(False, ())
        self._render_chat()
        if text:
            self.messages.append(ChatMessage(role="assistant", content=text))
            query = str(feedback_context.get("query", "")).strip()
            if query:
                refs = self._merge_feedback_references(
                    structured_refs=feedback_context.get("references", []),
                    answer_text=text,
                )
                self._set_feedback_target(
                    {
                        "query": query,
                        "answer": text,
                        "references": refs,
                        "requested_mode": feedback_context.get("requested_mode", ""),
                        "resolved_mode": feedback_context.get("resolved_mode", ""),
                        "query_type": feedback_context.get("query_type", ""),
                        "plan_summary": feedback_context.get("plan_summary", ""),
                        "kb_db_path": feedback_context.get("kb_db_path", ""),
                    }
                )
        else:
            self._set_feedback_target(None)
        self.pending_feedback_context = None
        self._clear_kb_feedback_context()

    def _set_kb_traceability_state(self, required: bool, refs: Sequence[str]) -> None:
        self.kb_traceability_required = bool(required)
        unique: list[str] = []
        seen: set[str] = set()
        for item in refs:
            value = " ".join(str(item).split()).strip()
            if not value:
                continue
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(value)
        self.kb_traceability_refs = unique[:12]

    def _ensure_traceable_kb_answer(self, text: str) -> str:
        body = (text or "").strip()
        if not body:
            return body
        if not bool(getattr(self, "kb_traceability_required", False)):
            return body

        has_confidence = bool(self._TRACE_CONFIDENCE_RE.search(body))
        has_source = bool(self._TRACE_SOURCE_RE.search(body))
        if has_confidence and has_source:
            return body

        confidence = "证据不足" if self._LOW_CONFIDENCE_HINT_RE.search(body) else "证据充分"
        refs = list(getattr(self, "kb_traceability_refs", []) or [])[:5]
        if not refs:
            refs = ["doc_id=unknown chunk_id=unknown"]
            confidence = "证据不足"
        references = "\n".join(f"- {item}" for item in refs)
        patch = (
            "\n\n可溯源补充:\n"
            f"证据: {confidence}\n"
            "引用来源:\n"
            f"{references}"
        )
        return f"{body}{patch}"

    def _build_pending_feedback_context(self, user_text: str) -> dict[str, object]:
        query = " ".join((user_text or "").split()).strip()
        refs: list[FeedbackReference] = []
        for item in list(getattr(self, "kb_feedback_refs", []) or []):
            if not isinstance(item, dict):
                continue
            doc_id = self._normalize_feedback_ref_id(str(item.get("doc_id", "")))
            chunk_id = self._normalize_feedback_ref_id(str(item.get("chunk_id", "")))
            if not doc_id and not chunk_id:
                continue
            rank_value = item.get("rank")
            rank: int | None = None
            if isinstance(rank_value, int):
                rank = rank_value
            refs.append(
                FeedbackReference(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    rank=rank,
                    source=str(item.get("source", "")).strip() or None,
                )
            )
        return {
            "query": query,
            "references": refs,
            "requested_mode": str(getattr(self, "kb_feedback_requested_mode", "")).strip(),
            "resolved_mode": str(getattr(self, "kb_feedback_resolved_mode", "")).strip(),
            "query_type": str(getattr(self, "kb_feedback_query_type", "")).strip(),
            "plan_summary": str(getattr(self, "kb_feedback_plan_summary", "")).strip(),
            "kb_db_path": str(getattr(self, "kb_feedback_db_path", "")).strip(),
        }

    def _merge_feedback_references(
        self,
        structured_refs: object,
        answer_text: str,
    ) -> list[FeedbackReference]:
        merged: list[FeedbackReference] = []
        seen: set[tuple[str, str]] = set()

        def _add(
            doc_id: str,
            chunk_id: str,
            rank: int | None = None,
            source: str | None = None,
        ) -> None:
            normalized_doc = self._normalize_feedback_ref_id(doc_id)
            normalized_chunk = self._normalize_feedback_ref_id(chunk_id)
            if not normalized_doc and not normalized_chunk:
                return
            key = (normalized_doc.lower(), normalized_chunk.lower())
            if key in seen:
                return
            seen.add(key)
            merged.append(
                FeedbackReference(
                    doc_id=normalized_doc,
                    chunk_id=normalized_chunk,
                    rank=rank,
                    source=source,
                )
            )

        if isinstance(structured_refs, list):
            for ref in structured_refs:
                if isinstance(ref, FeedbackReference):
                    _add(ref.doc_id, ref.chunk_id, ref.rank, ref.source)
                    continue
                if isinstance(ref, dict):
                    rank_value = ref.get("rank")
                    rank: int | None = int(rank_value) if isinstance(rank_value, int) else None
                    _add(
                        str(ref.get("doc_id", "")),
                        str(ref.get("chunk_id", "")),
                        rank=rank,
                        source=str(ref.get("source", "")).strip() or None,
                    )

        for ref in self._extract_traceability_refs_from_text(answer_text):
            _add(ref.doc_id, ref.chunk_id, ref.rank, ref.source)
        return merged

    def _extract_traceability_refs_from_text(self, text: str) -> list[FeedbackReference]:
        refs: list[FeedbackReference] = []
        for line in (text or "").splitlines():
            parsed = self._parse_traceability_line(line)
            if parsed is None:
                continue
            _, doc_id, chunk_id = parsed
            refs.append(
                FeedbackReference(
                    doc_id=self._normalize_feedback_ref_id(doc_id),
                    chunk_id=self._normalize_feedback_ref_id(chunk_id),
                )
            )
        return refs

    @staticmethod
    def _normalize_feedback_ref_id(value: str) -> str:
        normalized = " ".join((value or "").split()).strip()
        lowered = normalized.lower()
        if lowered in {"", "unknown", "n/a", "na", "-", "none", "null"}:
            return ""
        return normalized

    def _set_feedback_target(self, payload: dict[str, object] | None) -> None:
        self.feedback_target = payload
        self._refresh_feedback_buttons()

    def _refresh_feedback_buttons(self) -> None:
        enabled = bool(self.feedback_target) and not self.pending
        for attr in ("feedback_helpful_button", "feedback_unhelpful_button"):
            button = getattr(self, attr, None)
            if button is None:
                continue
            if enabled:
                button.state(["!disabled"])
            else:
                button.state(["disabled"])

    def _on_feedback_helpful(self) -> None:
        self._submit_feedback("helpful")

    def _on_feedback_unhelpful(self) -> None:
        self._submit_feedback("unhelpful")

    def _submit_feedback(self, label: str) -> None:
        target = self.feedback_target
        if not target:
            self._set_status("No answer available for feedback", tone="warn")
            return

        query = str(target.get("query", "")).strip()
        answer = str(target.get("answer", "")).strip()
        refs_raw = target.get("references", [])
        refs = self._merge_feedback_references(refs_raw, answer)
        if not query or not answer:
            self._set_status("Feedback skipped: empty query or answer", tone="warn")
            self._set_feedback_target(None)
            return

        top_k = 5
        try:
            top_k = max(1, int(self.kb_context_top_k_var.get().strip() or "5"))
        except Exception:
            top_k = 5

        retrieve_fn = None
        try:
            config = self._collect_kb_config()
            mode = str(target.get("requested_mode", "")).strip() or (
                self.kb_query_mode_var.get().strip() or "auto"
            )
            if config.store == "duckdb-vss" and mode in {"hybrid", "fts"}:
                mode = "vector"
            retrieve_fn = lambda query_text, case_top_k: self._feedback_retrieve_items(
                query_text=query_text,
                top_k=case_top_k,
                mode=mode,
                config=config,
            )
        except Exception:
            retrieve_fn = None

        record = FeedbackRecord(
            query=query,
            answer=answer,
            label=label,
            references=refs,
            requested_mode=str(target.get("requested_mode", "")).strip() or None,
            resolved_mode=str(target.get("resolved_mode", "")).strip() or None,
            query_type=str(target.get("query_type", "")).strip() or None,
            plan_summary=str(target.get("plan_summary", "")).strip() or None,
            kb_db_path=str(target.get("kb_db_path", "")).strip() or None,
            metadata={
                "provider": self._provider_key(),
                "model": self.model_var.get().strip(),
                "stream": bool(self.stream_var.get()),
            },
        )
        try:
            result = self.feedback_store.record_feedback(
                record,
                retrieve=retrieve_fn,
                default_top_k=top_k,
            )
        except Exception as exc:
            self._append_log("error", f"Feedback logging failed: {exc}")
            self._set_status("Feedback save failed", tone="error")
            return

        label_text = "有帮助" if label == "helpful" else "没帮助"
        summary = (
            f"反馈已记录: {label_text} | log={result.feedback_log_path} "
            f"| hard_cases={result.hard_cases_path}"
        )
        if label == "unhelpful":
            if result.hard_case_added:
                summary += " | hard_case=added"
            else:
                summary += " | hard_case=skipped(no refs)"
        if result.benchmark_report is not None:
            summary += f" | benchmark={self._feedback_benchmark_summary(result.benchmark_report)}"
        elif result.benchmark_error:
            summary += f" | benchmark_error={result.benchmark_error}"
        self._append_log("system", summary)
        self._set_status("Feedback saved", tone="ok")
        self._set_feedback_target(None)

    def _feedback_retrieve_items(
        self,
        query_text: str,
        top_k: int,
        mode: str,
        config: object,
    ) -> list[RetrievalItem]:
        if top_k <= 0:
            return []
        hits = self.kb_manager.query(
            query_text=query_text,
            top_k=top_k,
            mode=mode,
            config=config,
        )
        return [RetrievalItem(chunk_id=hit.chunk_id, doc_id=hit.doc_id) for hit in hits]

    @staticmethod
    def _feedback_benchmark_summary(report: dict[str, object]) -> str:
        total = int(report.get("total_cases", 0))
        hit_rate = float(report.get("hit_rate", 0.0))
        mrr = float(report.get("mrr", 0.0))
        recall = float(report.get("recall", 0.0))
        latency = report.get("latency_ms", {})
        if isinstance(latency, dict):
            p95 = float(latency.get("p95", 0.0))
        else:
            p95 = 0.0
        return (
            f"cases={total} hit_rate={hit_rate:.3f} mrr={mrr:.3f} "
            f"recall={recall:.3f} p95={p95:.1f}ms"
        )

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

            if self._TRACE_SOURCE_HEADING_RE.match(line):
                self._chat_insert("引用来源\n", role_tag, "md_source_heading")
                continue

            if self._render_traceability_line(role_tag, line):
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

    def _render_traceability_line(self, role_tag: str, line: str) -> bool:
        parsed = self._parse_traceability_line(line, require_token_at_start=True)
        if parsed is None:
            return False
        label, doc_id, chunk_id = parsed
        title, meta_text, chunk_text = self._format_traceability_display(doc_id, chunk_id)
        self._chat_insert("- ", role_tag, "md_list_marker")
        if label:
            self._chat_insert(f"[{label}] ", role_tag, "md_source_label")
        self._chat_insert(title, role_tag, "md_source_item")
        if chunk_text:
            self._chat_insert(f" {chunk_text}", role_tag, "md_source_chunk")
        self._chat_insert("\n", role_tag, "md_source_item")
        if meta_text:
            self._chat_insert(f"  {meta_text}\n", role_tag, "md_source_meta")
        return True

    @classmethod
    def _parse_traceability_line(
        cls,
        line: str,
        *,
        require_token_at_start: bool = False,
    ) -> tuple[str | None, str, str] | None:
        body = (line or "").strip()
        if not body:
            return None

        if body[:1] in {"-", "*", "+"}:
            body = body[1:].lstrip()

        label: str | None = None
        label_match = cls._TRACE_LABEL_RE.match(body)
        if label_match is not None:
            label = label_match.group("label").strip() or None
            body = body[label_match.end() :].lstrip()

        matches = list(cls._TRACE_ID_TOKEN_RE.finditer(body))
        if not matches:
            return None
        if require_token_at_start and matches[0].start() != 0:
            return None

        fields: dict[str, str] = {}
        for index, match in enumerate(matches):
            key = match.group(1).lower()
            value_start = match.end()
            value_end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
            value = body[value_start:value_end].strip()
            fields[key] = value

        doc_id = fields.get("doc_id", "")
        chunk_id = fields.get("chunk_id", "")
        if not doc_id and not chunk_id:
            return None
        return label, doc_id, chunk_id

    @classmethod
    def _format_traceability_display(
        cls,
        doc_id: str,
        chunk_id: str,
    ) -> tuple[str, str | None, str | None]:
        normalized_doc = cls._normalize_feedback_ref_id(doc_id)
        normalized_chunk = cls._normalize_feedback_ref_id(chunk_id)

        title = normalized_doc or normalized_chunk or "unknown source"
        meta_text: str | None = None
        if normalized_doc.lower().startswith("file:"):
            raw_path = normalized_doc[5:]
            path = Path(raw_path)
            if path.name:
                title = path.name
            parent = path.parent.as_posix()
            if parent and parent != ".":
                meta_text = cls._shorten_middle(parent, 72)
            else:
                meta_text = cls._shorten_middle(path.as_posix(), 72)
        elif normalized_doc:
            title = cls._shorten_middle(normalized_doc, 72)

        chunk_text: str | None = None
        if normalized_chunk:
            if "#chunk:" in normalized_chunk:
                chunk_text = f"[chunk {normalized_chunk.rsplit(':', 1)[-1]}]"
            else:
                chunk_text = f"[{cls._shorten_middle(normalized_chunk, 36)}]"
        return title, meta_text, chunk_text

    @staticmethod
    def _shorten_middle(text: str, max_chars: int) -> str:
        value = str(text or "")
        if max_chars <= 3 or len(value) <= max_chars:
            return value
        keep_left = max(1, (max_chars - 3) // 2)
        keep_right = max(1, max_chars - 3 - keep_left)
        return f"{value[:keep_left]}...{value[-keep_right:]}"

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
