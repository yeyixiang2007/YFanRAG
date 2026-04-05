from yfanrag.chunking import FixedChunker, RecursiveChunker, StructureAwareChunker
from yfanrag.models import Document


def test_fixed_chunker_overlap():
    doc = Document(doc_id="d1", text="abcdefghij")
    chunker = FixedChunker(chunk_size=4, chunk_overlap=1)
    chunks = chunker.chunk(doc)

    assert chunks[0].text == "abcd"
    assert chunks[1].text == "defg"
    assert chunks[-1].text == "j"


def test_recursive_chunker_limits_size():
    text = "aa bb cc dd ee ff gg hh ii jj"
    doc = Document(doc_id="d2", text=text)
    chunker = RecursiveChunker(chunk_size=6, chunk_overlap=1)
    chunks = chunker.chunk(doc)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 6 for chunk in chunks)


def test_structure_aware_chunker_markdown_headings():
    doc = Document(
        doc_id="d3",
        text=(
            "# Overview\n"
            "Intro text.\n\n"
            "## Usage\n"
            "Step one.\n"
            "Step two.\n"
        ),
        metadata={"path": "demo.md"},
    )
    chunker = StructureAwareChunker(chunk_size=200, chunk_overlap=0)
    chunks = chunker.chunk(doc)

    assert len(chunks) >= 2
    assert any(chunk.metadata.get("section_type") == "markdown.heading" for chunk in chunks)
    assert any(chunk.metadata.get("section_level") == 1 for chunk in chunks)
    assert any(chunk.metadata.get("section_title") == "Usage" for chunk in chunks)


def test_structure_aware_chunker_python_symbols():
    doc = Document(
        doc_id="d4",
        text=(
            "import os\n\n"
            "class Service:\n"
            "    pass\n\n"
            "def run_job():\n"
            "    return True\n"
        ),
        metadata={"path": "demo.py"},
    )
    chunker = StructureAwareChunker(chunk_size=200, chunk_overlap=0)
    chunks = chunker.chunk(doc)
    tags = {(chunk.metadata.get("section_type"), chunk.metadata.get("section_title")) for chunk in chunks}

    assert ("python.class", "Service") in tags
    assert ("python.function", "run_job") in tags


def test_structure_aware_chunker_javascript_symbols():
    doc = Document(
        doc_id="d5",
        text=(
            "const value = 1;\n\n"
            "class Engine {\n"
            "  start() { return value; }\n"
            "}\n\n"
            "function helper() {\n"
            "  return 42;\n"
            "}\n\n"
            "const run = () => helper();\n"
        ),
        metadata={"path": "demo.js"},
    )
    chunker = StructureAwareChunker(chunk_size=240, chunk_overlap=0)
    chunks = chunker.chunk(doc)
    tags = {(chunk.metadata.get("section_type"), chunk.metadata.get("section_title")) for chunk in chunks}

    assert ("javascript.class", "Engine") in tags
    assert ("javascript.function", "helper") in tags
    assert ("javascript.function", "run") in tags


def test_structure_aware_chunker_falls_back_for_txt():
    doc = Document(
        doc_id="d6",
        text="aa bb cc dd ee ff gg hh ii jj",
        metadata={"path": "demo.txt"},
    )
    structured = StructureAwareChunker(chunk_size=6, chunk_overlap=1)
    recursive = RecursiveChunker(chunk_size=6, chunk_overlap=1)

    structured_chunks = structured.chunk(doc)
    recursive_chunks = recursive.chunk(doc)

    assert [chunk.text for chunk in structured_chunks] == [chunk.text for chunk in recursive_chunks]
