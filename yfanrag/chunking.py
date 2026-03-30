"""Chunking strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from .models import Chunk, Document


@dataclass
class FixedChunker:
    chunk_size: int = 800
    chunk_overlap: int = 120

    def chunk(self, document: Document) -> List[Chunk]:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

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
class RecursiveChunker:
    chunk_size: int = 800
    chunk_overlap: int = 120
    separators: Sequence[str] = ("\n\n", "\n", " ", "")

    def chunk(self, document: Document) -> List[Chunk]:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

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
