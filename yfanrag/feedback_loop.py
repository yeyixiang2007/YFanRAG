"""Feedback-loop helpers for GUI thumbs feedback and retrieval regression."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence
import json

from .benchmark import RetrievalItem, evaluate_retrieval_benchmark, load_benchmark_cases

_UNKNOWN_REF_VALUES = {"", "unknown", "n/a", "na", "-", "none", "null"}
_LABEL_ALIASES = {
    "helpful": "helpful",
    "unhelpful": "unhelpful",
    "有帮助": "helpful",
    "没帮助": "unhelpful",
    "无帮助": "unhelpful",
}


@dataclass(frozen=True)
class FeedbackReference:
    doc_id: str
    chunk_id: str = ""
    rank: int | None = None
    source: str | None = None


@dataclass(frozen=True)
class FeedbackRecord:
    query: str
    answer: str
    label: str
    references: Sequence[FeedbackReference] = ()
    requested_mode: str | None = None
    resolved_mode: str | None = None
    query_type: str | None = None
    plan_summary: str | None = None
    kb_db_path: str | None = None
    metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class FeedbackLoopResult:
    feedback_log_path: str
    hard_cases_path: str
    hard_case_added: bool
    benchmark_case_count: int
    benchmark_report: dict[str, object] | None = None
    benchmark_error: str | None = None


class FeedbackLoopStore:
    def __init__(self, root_dir: str | None = None) -> None:
        if root_dir is None:
            root_dir = str(Path.home() / ".yfanrag" / "feedback")
        self.root_dir = Path(root_dir)

    @property
    def feedback_log_path(self) -> Path:
        return self.root_dir / "feedback_events.jsonl"

    @property
    def hard_cases_path(self) -> Path:
        return self.root_dir / "hard_cases.jsonl"

    def record_feedback(
        self,
        record: FeedbackRecord,
        *,
        retrieve: Callable[[str, int], Sequence[RetrievalItem]] | None = None,
        default_top_k: int = 5,
    ) -> FeedbackLoopResult:
        query = record.query.strip()
        if not query:
            raise ValueError("feedback query cannot be empty")
        answer = record.answer.strip()
        if not answer:
            raise ValueError("feedback answer cannot be empty")

        normalized_label = self._normalize_label(record.label)
        top_k = max(1, int(default_top_k))
        timestamp = datetime.now(timezone.utc).isoformat()
        references = self._normalize_references(record.references)

        event_row: dict[str, object] = {
            "ts_utc": timestamp,
            "label": normalized_label,
            "query": query,
            "answer": answer,
            "references": references,
            "requested_mode": (record.requested_mode or "").strip(),
            "resolved_mode": (record.resolved_mode or "").strip(),
            "query_type": (record.query_type or "").strip(),
            "plan_summary": (record.plan_summary or "").strip(),
            "kb_db_path": (record.kb_db_path or "").strip(),
        }
        if record.metadata:
            event_row["metadata"] = record.metadata
        self._append_jsonl(self.feedback_log_path, event_row)

        hard_case_added = False
        if normalized_label == "unhelpful":
            hard_case_row = self._build_hard_case_row(
                query=query,
                references=references,
                top_k=top_k,
            )
            if hard_case_row is not None:
                hard_case_row["label"] = normalized_label
                hard_case_row["ts_utc"] = timestamp
                if event_row["resolved_mode"]:
                    hard_case_row["mode"] = event_row["resolved_mode"]
                self._append_jsonl(self.hard_cases_path, hard_case_row)
                hard_case_added = True

        report: dict[str, object] | None = None
        benchmark_case_count = 0
        benchmark_error: str | None = None
        if retrieve is not None and self.hard_cases_path.exists():
            try:
                cases = load_benchmark_cases(str(self.hard_cases_path))
                benchmark_case_count = len(cases)
                if cases:
                    report = evaluate_retrieval_benchmark(
                        cases=cases,
                        retrieve=retrieve,
                        default_top_k=top_k,
                    )
            except Exception as exc:  # pragma: no cover - defensive path
                benchmark_error = str(exc)

        return FeedbackLoopResult(
            feedback_log_path=str(self.feedback_log_path),
            hard_cases_path=str(self.hard_cases_path),
            hard_case_added=hard_case_added,
            benchmark_case_count=benchmark_case_count,
            benchmark_report=report,
            benchmark_error=benchmark_error,
        )

    @staticmethod
    def _normalize_label(value: str) -> str:
        normalized = _LABEL_ALIASES.get(value.strip().lower()) or _LABEL_ALIASES.get(value.strip())
        if normalized is None:
            raise ValueError(f"unsupported feedback label: {value}")
        return normalized

    @classmethod
    def _normalize_references(
        cls,
        references: Sequence[FeedbackReference],
    ) -> list[dict[str, object]]:
        normalized: list[dict[str, object]] = []
        seen: set[tuple[str, str]] = set()
        for ref in references:
            doc_id = cls._normalize_ref_id(ref.doc_id)
            chunk_id = cls._normalize_ref_id(ref.chunk_id)
            key = (doc_id.lower(), chunk_id.lower())
            if key in seen:
                continue
            seen.add(key)
            item: dict[str, object] = {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
            }
            if ref.rank is not None:
                item["rank"] = int(ref.rank)
            if ref.source:
                item["source"] = str(ref.source).strip()
            normalized.append(item)
        return normalized

    @classmethod
    def _build_hard_case_row(
        cls,
        *,
        query: str,
        references: Sequence[dict[str, object]],
        top_k: int,
    ) -> dict[str, object] | None:
        chunk_ids: list[str] = []
        doc_ids: list[str] = []
        seen_chunk: set[str] = set()
        seen_doc: set[str] = set()

        for item in references:
            chunk_id = cls._normalize_ref_id(str(item.get("chunk_id", "")))
            doc_id = cls._normalize_ref_id(str(item.get("doc_id", "")))
            if chunk_id and chunk_id.lower() not in seen_chunk:
                seen_chunk.add(chunk_id.lower())
                chunk_ids.append(chunk_id)
            if doc_id and doc_id.lower() not in seen_doc:
                seen_doc.add(doc_id.lower())
                doc_ids.append(doc_id)

        if not chunk_ids and not doc_ids:
            return None
        row: dict[str, object] = {
            "query": query,
            "top_k": max(1, int(top_k)),
        }
        if chunk_ids:
            row["expected_chunk_ids"] = chunk_ids
        else:
            row["expected_doc_ids"] = doc_ids
        return row

    @staticmethod
    def _normalize_ref_id(value: str) -> str:
        normalized = " ".join(value.split()).strip()
        if normalized.lower() in _UNKNOWN_REF_VALUES:
            return ""
        return normalized

    @staticmethod
    def _append_jsonl(path: Path, row: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        with path.open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")
