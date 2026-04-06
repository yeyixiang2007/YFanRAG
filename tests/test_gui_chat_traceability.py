from yfanrag.gui.mixins.chat import AppChatMixin


class _DummyChatApp(AppChatMixin):
    def __init__(self) -> None:
        self.rendered: list[tuple[str, tuple[str, ...]]] = []

    def _chat_insert(self, text: str, *tags: str) -> None:
        self.rendered.append((text, tuple(tag for tag in tags if tag)))


def test_extract_traceability_refs_preserves_paths_with_spaces() -> None:
    app = _DummyChatApp()
    text = (
        "引用来源:\n"
        "- [来源1] doc_id=file:D:/StudentBrain/大学物理/§ 11.1 简谐振动.md "
        "chunk_id=file:D:/StudentBrain/大学物理/§ 11.1 简谐振动.md#chunk:3"
    )

    refs = app._extract_traceability_refs_from_text(text)

    assert len(refs) == 1
    assert refs[0].doc_id == "file:D:/StudentBrain/大学物理/§ 11.1 简谐振动.md"
    assert refs[0].chunk_id == "file:D:/StudentBrain/大学物理/§ 11.1 简谐振动.md#chunk:3"


def test_render_markdown_formats_traceability_lines_for_readability() -> None:
    app = _DummyChatApp()
    text = (
        "引用来源:\n"
        "- [来源1] doc_id=file:D:/StudentBrain/大学物理/§ 11.1 简谐振动.md "
        "chunk_id=file:D:/StudentBrain/大学物理/§ 11.1 简谐振动.md#chunk:3"
    )

    app._render_markdown("assistant", text)
    rendered = "".join(part for part, _ in app.rendered)

    assert "引用来源" in rendered
    assert "§ 11.1 简谐振动.md" in rendered
    assert "[chunk 3]" in rendered
    assert "doc_id=" not in rendered
    assert "chunk_id=" not in rendered
    assert any("md_source_item" in tags for _, tags in app.rendered)
