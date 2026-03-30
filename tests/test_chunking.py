from yfanrag.chunking import FixedChunker, RecursiveChunker
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
