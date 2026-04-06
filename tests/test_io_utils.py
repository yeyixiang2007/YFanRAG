from pathlib import Path

import pytest

from yfanrag.io_utils import append_text_atomic


def test_append_text_atomic_appends_without_reading_existing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text("first\n", encoding="utf-8")

    def _unexpected_read(*args, **kwargs):
        raise AssertionError("append_text_atomic should not read the whole file")

    monkeypatch.setattr(Path, "read_text", _unexpected_read)

    append_text_atomic(path, "second\n", encoding="utf-8")

    with path.open("r", encoding="utf-8") as handle:
        assert handle.read() == "first\nsecond\n"
