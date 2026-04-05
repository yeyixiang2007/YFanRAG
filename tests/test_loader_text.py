from pathlib import Path

import pytest

from yfanrag.loaders.text import TextFileLoader


def test_text_loader_reads_txt_md_and_code(tmp_path: Path):
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "b.md").write_text("# title", encoding="utf-8")
    (tmp_path / "c.py").write_text("def x():\n    return 1\n", encoding="utf-8")
    (tmp_path / "d.gd").write_text("extends Node\n", encoding="utf-8")
    (tmp_path / "c.log").write_text("ignore", encoding="utf-8")

    loader = TextFileLoader(paths=[str(tmp_path)])
    docs = loader.load()

    assert len(docs) == 4
    assert all(doc.doc_id.startswith("file:") for doc in docs)
    paths = {doc.metadata["path"] for doc in docs}
    assert (tmp_path / "a.txt").as_posix() in paths
    assert (tmp_path / "b.md").as_posix() in paths
    assert (tmp_path / "c.py").as_posix() in paths
    assert (tmp_path / "d.gd").as_posix() in paths


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


def test_text_loader_falls_back_for_gbk_files(tmp_path: Path):
    path = tmp_path / "note.txt"
    path.write_bytes("中文内容".encode("gb18030"))

    loader = TextFileLoader(paths=[str(path)])
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].text == "中文内容"
    assert docs[0].metadata["encoding"] == "gb18030"


def test_text_loader_skips_large_files(tmp_path: Path):
    (tmp_path / "small.txt").write_text("ok", encoding="utf-8")
    (tmp_path / "large.txt").write_text("0123456789", encoding="utf-8")

    loader = TextFileLoader(
        paths=[str(tmp_path)],
        max_file_size_bytes=5,
    )
    docs = loader.load()

    assert [doc.title for doc in docs] == ["small"]
