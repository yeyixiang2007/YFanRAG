"""Markdown parsing regexes for Tk chat rendering."""

from __future__ import annotations

import re

MARKDOWN_FENCE_RE = re.compile(r"^\s*```([\w.+-]+)?\s*$")
MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*)$")
MARKDOWN_QUOTE_RE = re.compile(r"^\s{0,3}>\s?(.*)$")
MARKDOWN_UNORDERED_LIST_RE = re.compile(r"^(\s*)[-*+]\s+(.*)$")
MARKDOWN_ORDERED_LIST_RE = re.compile(r"^(\s*)(\d+)[\.)]\s+(.*)$")
MARKDOWN_HR_RE = re.compile(r"^\s{0,3}(?:-{3,}|\*{3,}|_{3,})\s*$")
MARKDOWN_INLINE_TOKEN_RE = re.compile(
    r"(`[^`\n]+`|\[[^\]\n]+\]\([^)]+\)|\*\*\*.+?\*\*\*|___.+?___|\*\*.+?\*\*|__.+?__|~~.+?~~|\*[^*\n]+\*|_[^_\n]+_)"
)
MARKDOWN_LINK_RE = re.compile(r"^\[([^\]\n]+)\]\(([^)\n]+)\)$")
