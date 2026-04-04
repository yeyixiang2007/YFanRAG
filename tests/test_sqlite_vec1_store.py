from yfanrag.models import Chunk
from yfanrag.vectorstores.sqlite_vec1 import SqliteVec1Store


def test_sqlite_vec1_store_roundtrip_without_extension(tmp_path):
    db_path = tmp_path / "vec1.db"
    store = SqliteVec1Store(
        path=str(db_path),
        embedding_dim=4,
        load_extension=False,
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

    results = store.query([1.0, 0.0, 0.0, 0.0], top_k=1)
    by_doc = store.query([1.0, 0.0, 0.0, 0.0], top_k=5, filters={"doc_id": "d2"})
    by_range = store.query(
        [1.0, 0.0, 0.0, 0.0],
        top_k=5,
        range_filters={"start": (5, None)},
    )
    deleted = store.delete_by_doc_ids(["d1"])
    remaining = store.query([1.0, 0.0, 0.0, 0.0], top_k=5)
    store.close()

    assert [chunk.chunk_id for chunk in results] == ["c1"]
    assert [chunk.chunk_id for chunk in by_doc] == ["c2"]
    assert [chunk.chunk_id for chunk in by_range] == ["c2"]
    assert deleted == 1
    assert all(chunk.doc_id != "d1" for chunk in remaining)
