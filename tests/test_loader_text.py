from pathlib import Path

import pytest

from yfanrag.loaders.text import TextFileLoader


def test_text_loader_reads_txt_and_md(tmp_path: Path):
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "b.md").write_text("# title", encoding="utf-8")
    (tmp_path / "c.log").write_text("ignore", encoding="utf-8")

    loader = TextFileLoader(paths=[str(tmp_path)])
    docs = loader.load()

    assert len(docs) == 2
    assert all(doc.doc_id.startswith("file:") for doc in docs)
    paths = {doc.metadata["path"] for doc in docs}
    assert (tmp_path / "a.txt").as_posix() in paths
    assert (tmp_path / "b.md").as_posix() in paths


def test_text_loader_with_path_whitelist(tmp_path: Path):
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    loader = TextFileLoader(
        paths=[str(tmp_path)],
        path_whitelist=[str(tmp_path)],
    )
    docs = loader.load()
    assert len(docs) == 1


def test_text_loader_path_whitelist_blocks_unauthorized(tmp_path: Path):
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    outside = tmp_path.parent
    loader = TextFileLoader(
        paths=[str(tmp_path)],
        path_whitelist=[str(outside / "not-allowed")],
    )
    with pytest.raises(ValueError):
        loader.load()
