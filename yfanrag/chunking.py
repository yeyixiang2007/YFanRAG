"""Chunking strategies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, List, Sequence, Tuple

from .models import Chunk, Document

_PY_SYMBOL_RE = re.compile(
    r"(?m)^((?:async[ \t]+def)|def|class)[ \t]+([A-Za-z_][A-Za-z0-9_]*)\b"
)
_JS_CLASS_RE = re.compile(
    r"(?m)^(?:export[ \t]+)?(?:default[ \t]+)?class[ \t]+([A-Za-z_$][A-Za-z0-9_$]*)\b"
)
_JS_FUNCTION_RE = re.compile(
    r"(?m)^(?:export[ \t]+)?(?:default[ \t]+)?(?:async[ \t]+)?function[ \t]+([A-Za-z_$][A-Za-z0-9_$]*)\s*\("
)
_JS_ARROW_RE = re.compile(
    r"(?m)^(?:export[ \t]+)?(?:const|let|var)[ \t]+([A-Za-z_$][A-Za-z0-9_$]*)[ \t]*=[ \t]*(?:async[ \t]+)?(?:\([^)\n]*\)|[A-Za-z_$][A-Za-z0-9_$]*)[ \t]*=>"
)
_MARKDOWN_HEADING_LINE_RE = re.compile(r"^[ \t]{0,3}(#{1,6})[ \t]+(.+?)\s*$")
_STRUCTURED_MD_EXTENSIONS = {".md", ".markdown"}
_STRUCTURED_PY_EXTENSIONS = {".py"}
_STRUCTURED_JS_EXTENSIONS = {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}


@dataclass(frozen=True)
class StructuredSpan:
    start: int
    end: int
    section_type: str
    section_title: str | None = None
    section_level: int | None = None


def _validate_chunk_window(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")


def _normalize_spans(spans: Sequence[StructuredSpan], text_len: int) -> List[StructuredSpan]:
    ordered = sorted(spans, key=lambda item: (item.start, item.end))
    normalized: list[StructuredSpan] = []
    cursor = 0
    for item in ordered:
        start = max(0, min(item.start, text_len))
        end = max(start, min(item.end, text_len))
        if end <= start:
            continue
        if start < cursor:
            start = cursor
        if end <= start:
            continue
        normalized.append(
            StructuredSpan(
                start=start,
                end=end,
                section_type=item.section_type,
                section_title=item.section_title,
                section_level=item.section_level,
            )
        )
        cursor = end
    return normalized


@dataclass
class FixedChunker:
    chunk_size: int = 800
    chunk_overlap: int = 120

    def chunk(self, document: Document) -> List[Chunk]:
        _validate_chunk_window(self.chunk_size, self.chunk_overlap)

        text = document.text
        step = self.chunk_size - self.chunk_overlap
        chunks: List[Chunk] = []
        index = 0
        for start in range(0, len(text), step):
            end = min(start + self.chunk_size, len(text))
            if start >= end:
                break
            chunk_text = text[start:end]
            chunk_id = f"{document.doc_id}#chunk:{index}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=document.doc_id,
                    text=chunk_text,
                    start=start,
                    end=end,
                    metadata={"index": index},
                )
            )
            index += 1
        return chunks


@dataclass
class StructureAwareChunker:
    """Uses structure-aware spans for markdown and code, then size-aware sub-chunks."""

    chunk_size: int = 800
    chunk_overlap: int = 120
    separators: Sequence[str] = ("\n\n", "\n", " ", "")

    def chunk(self, document: Document) -> List[Chunk]:
        _validate_chunk_window(self.chunk_size, self.chunk_overlap)
        text = document.text
        if not text:
            return []

        suffix = self._detect_suffix(document)
        spans = self._extract_structured_spans(text=text, suffix=suffix)
        if not spans:
            return self._fallback_chunker().chunk(document)

        chunks = self._materialize_chunks(document=document, spans=spans)
        if chunks:
            return chunks
        return self._fallback_chunker().chunk(document)

    def _fallback_chunker(self) -> RecursiveChunker:
        return RecursiveChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
        )

    @staticmethod
    def _detect_suffix(document: Document) -> str:
        path = ""
        metadata_path = document.metadata.get("path")
        if isinstance(metadata_path, str):
            path = metadata_path.strip()
        if not path and document.source:
            path = document.source.strip()
        if not path and document.doc_id.startswith("file:"):
            path = document.doc_id[len("file:") :].strip()
        if not path:
            return ""
        return Path(path).suffix.lower()

    def _extract_structured_spans(self, text: str, suffix: str) -> List[StructuredSpan]:
        if suffix in _STRUCTURED_MD_EXTENSIONS:
            return self._extract_markdown_spans(text)
        if suffix in _STRUCTURED_PY_EXTENSIONS:
            return self._extract_python_spans(text)
        if suffix in _STRUCTURED_JS_EXTENSIONS:
            return self._extract_javascript_spans(text)
        return []

    @staticmethod
    def _extract_markdown_spans(text: str) -> List[StructuredSpan]:
        headings: list[tuple[int, int, str]] = []
        in_code_block = False
        cursor = 0
        for line in text.splitlines(keepends=True):
            stripped = line.strip()
            if stripped.startswith("```") or stripped.startswith("~~~"):
                in_code_block = not in_code_block
            if not in_code_block:
                match = _MARKDOWN_HEADING_LINE_RE.match(line.rstrip("\r\n"))
                if match is not None:
                    level = len(match.group(1))
                    title = match.group(2).strip()
                    headings.append((cursor, level, title))
            cursor += len(line)
        if not headings:
            return []

        spans: list[StructuredSpan] = []
        first_start = headings[0][0]
        if first_start > 0 and text[:first_start].strip():
            spans.append(
                StructuredSpan(
                    start=0,
                    end=first_start,
                    section_type="markdown.preamble",
                    section_title="Preamble",
                )
            )
        for idx, (start, level, title) in enumerate(headings):
            end = headings[idx + 1][0] if idx + 1 < len(headings) else len(text)
            if end <= start or not text[start:end].strip():
                continue
            spans.append(
                StructuredSpan(
                    start=start,
                    end=end,
                    section_type="markdown.heading",
                    section_title=title,
                    section_level=level,
                )
            )
        return _normalize_spans(spans=spans, text_len=len(text))

    @staticmethod
    def _extract_python_spans(text: str) -> List[StructuredSpan]:
        matches = list(_PY_SYMBOL_RE.finditer(text))
        if not matches:
            return []

        spans: list[StructuredSpan] = []
        first_start = matches[0].start()
        if first_start > 0 and text[:first_start].strip():
            spans.append(
                StructuredSpan(
                    start=0,
                    end=first_start,
                    section_type="python.preamble",
                    section_title="Preamble",
                )
            )
        for idx, match in enumerate(matches):
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            if end <= start or not text[start:end].strip():
                continue
            keyword = match.group(1)
            symbol = match.group(2)
            section_type = "python.class" if keyword == "class" else "python.function"
            spans.append(
                StructuredSpan(
                    start=start,
                    end=end,
                    section_type=section_type,
                    section_title=symbol,
                )
            )
        return _normalize_spans(spans=spans, text_len=len(text))

    @staticmethod
    def _extract_javascript_spans(text: str) -> List[StructuredSpan]:
        declarations: list[tuple[int, str, str]] = []
        declarations.extend(
            (match.start(), "javascript.class", match.group(1))
            for match in _JS_CLASS_RE.finditer(text)
        )
        declarations.extend(
            (match.start(), "javascript.function", match.group(1))
            for match in _JS_FUNCTION_RE.finditer(text)
        )
        declarations.extend(
            (match.start(), "javascript.function", match.group(1))
            for match in _JS_ARROW_RE.finditer(text)
        )
        if not declarations:
            return []

        declarations.sort(key=lambda item: item[0])
        deduped: list[tuple[int, str, str]] = []
        seen_start: set[int] = set()
        for item in declarations:
            if item[0] in seen_start:
                continue
            seen_start.add(item[0])
            deduped.append(item)

        spans: list[StructuredSpan] = []
        first_start = deduped[0][0]
        if first_start > 0 and text[:first_start].strip():
            spans.append(
                StructuredSpan(
                    start=0,
                    end=first_start,
                    section_type="javascript.preamble",
                    section_title="Preamble",
                )
            )
        for idx, (start, section_type, title) in enumerate(deduped):
            end = deduped[idx + 1][0] if idx + 1 < len(deduped) else len(text)
            if end <= start or not text[start:end].strip():
                continue
            spans.append(
                StructuredSpan(
                    start=start,
                    end=end,
                    section_type=section_type,
                    section_title=title,
                )
            )
        return _normalize_spans(spans=spans, text_len=len(text))

    def _materialize_chunks(
        self,
        document: Document,
        spans: Sequence[StructuredSpan],
    ) -> List[Chunk]:
        recursive = self._fallback_chunker()
        chunks: list[Chunk] = []
        index = 0
        for span in spans:
            if span.end <= span.start:
                continue
            section_text = document.text[span.start : span.end]
            if not section_text.strip():
                continue
            local_spans = recursive._split_recursive(section_text, 0, list(recursive.separators))
            local_spans = recursive._apply_overlap(local_spans, len(section_text))
            for piece_index, (local_start, local_end) in enumerate(local_spans):
                abs_start = span.start + local_start
                abs_end = span.start + local_end
                if abs_end <= abs_start:
                    continue
                metadata: dict[str, object] = {
                    "index": index,
                    "section_type": span.section_type,
                }
                if span.section_title:
                    metadata["section_title"] = span.section_title
                if span.section_level is not None:
                    metadata["section_level"] = span.section_level
                if len(local_spans) > 1:
                    metadata["section_piece"] = piece_index
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.doc_id}#chunk:{index}",
                        doc_id=document.doc_id,
                        text=document.text[abs_start:abs_end],
                        start=abs_start,
                        end=abs_end,
                        metadata=metadata,
                    )
                )
                index += 1
        return chunks


@dataclass
class RecursiveChunker:
    chunk_size: int = 800
    chunk_overlap: int = 120
    separators: Sequence[str] = ("\n\n", "\n", " ", "")

    def chunk(self, document: Document) -> List[Chunk]:
        _validate_chunk_window(self.chunk_size, self.chunk_overlap)

        spans = self._split_recursive(document.text, 0, list(self.separators))
        spans = self._apply_overlap(spans, len(document.text))

        chunks: List[Chunk] = []
        for index, (start, end) in enumerate(spans):
            chunk_id = f"{document.doc_id}#chunk:{index}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=document.doc_id,
                    text=document.text[start:end],
                    start=start,
                    end=end,
                    metadata={"index": index},
                )
            )
        return chunks

    def _split_recursive(
        self, text: str, offset: int, separators: List[str]
    ) -> List[Tuple[int, int]]:
        if len(text) <= self.chunk_size:
            return [(offset, offset + len(text))]
        if not separators:
            return self._split_fixed(text, offset)

        sep = separators[0]
        if sep == "":
            return self._split_fixed(text, offset)

        parts = text.split(sep)
        if len(parts) == 1:
            return self._split_recursive(text, offset, separators[1:])

        spans: List[Tuple[int, int]] = []
        cursor = 0
        for index, part in enumerate(parts):
            if part:
                part_offset = offset + cursor
                if len(part) > self.chunk_size:
                    spans.extend(self._split_recursive(part, part_offset, separators[1:]))
                else:
                    spans.append((part_offset, part_offset + len(part)))
                cursor += len(part)
            if index < len(parts) - 1:
                sep_offset = offset + cursor
                spans.append((sep_offset, sep_offset + len(sep)))
                cursor += len(sep)

        return self._merge_spans(spans)

    def _split_fixed(self, text: str, offset: int) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        for start in range(0, len(text), self.chunk_size):
            end = min(start + self.chunk_size, len(text))
            if start >= end:
                break
            spans.append((offset + start, offset + end))
        return spans

    def _merge_spans(self, spans: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
        spans_list = list(spans)
        if not spans_list:
            return []
        merged: List[Tuple[int, int]] = []
        chunk_start, chunk_end = spans_list[0]
        for start, end in spans_list[1:]:
            if end - chunk_start <= self.chunk_size:
                chunk_end = end
            else:
                merged.append((chunk_start, chunk_end))
                chunk_start, chunk_end = start, end
        merged.append((chunk_start, chunk_end))
        return merged

    def _apply_overlap(
        self, spans: List[Tuple[int, int]], text_len: int
    ) -> List[Tuple[int, int]]:
        if self.chunk_overlap <= 0:
            return spans

        adjusted: List[Tuple[int, int]] = []
        for start, end in spans:
            new_start = max(0, start - self.chunk_overlap)
            if end - new_start > self.chunk_size:
                new_start = max(0, end - self.chunk_size)
            adjusted.append((new_start, min(end, text_len)))
        return adjusted
