import pytest

from yfanrag.models import Chunk
from yfanrag.vectorstores.duckdb_vss import DuckDbVssStore

duckdb = pytest.importorskip("duckdb")


def test_duckdb_vss_store_roundtrip_without_vss(tmp_path):
    db_path = tmp_path / "vss.duckdb"
    store = DuckDbVssStore(
        path=str(db_path),
        embedding_dim=4,
        enable_vss=False,
        distance_metric="l2",
    )

    chunks = [
        Chunk(chunk_id="c1", doc_id="d1", text="hello", start=0, end=5, metadata={"index": 0}),
        Chunk(chunk_id="c2", doc_id="d2", text="world", start=6, end=11, metadata={"index": 1}),
    ]
    embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
    store.add(chunks, embeddings)

    top1 = store.query([1.0, 0.0, 0.0, 0.0], top_k=1)
    by_doc = store.query([1.0, 0.0, 0.0, 0.0], top_k=5, filters={"doc_id": "d2"})
    deleted = store.delete_by_doc_ids(["d1"])
    remain = store.query([1.0, 0.0, 0.0, 0.0], top_k=5)
    store.close()

    assert [chunk.chunk_id for chunk in top1] == ["c1"]
    assert [chunk.chunk_id for chunk in by_doc] == ["c2"]
    assert deleted >= 0
    assert all(chunk.doc_id != "d1" for chunk in remain)
