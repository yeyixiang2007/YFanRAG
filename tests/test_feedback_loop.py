from yfanrag.benchmark import RetrievalItem
from yfanrag.feedback_loop import FeedbackLoopStore, FeedbackRecord, FeedbackReference


def test_record_unhelpful_feedback_generates_hard_case_and_benchmark(tmp_path):
    store = FeedbackLoopStore(root_dir=str(tmp_path))
    record = FeedbackRecord(
        query="how to configure sqlite-vec1",
        answer="not useful answer",
        label="没帮助",
        references=[
            FeedbackReference(doc_id="file:docs/TECHNICAL.md", chunk_id="chunk-1"),
            FeedbackReference(doc_id="file:docs/TECHNICAL.md", chunk_id="chunk-1"),
        ],
        resolved_mode="auto",
    )

    report = store.record_feedback(
        record,
        default_top_k=5,
        retrieve=lambda query, top_k: [
            RetrievalItem(chunk_id="chunk-1", doc_id="file:docs/TECHNICAL.md"),
            RetrievalItem(chunk_id="chunk-2", doc_id="file:docs/OTHER.md"),
        ][:top_k],
    )

    assert report.hard_case_added is True
    assert report.benchmark_case_count == 1
    assert report.benchmark_report is not None
    assert report.benchmark_report["hit_rate"] == 1.0
    assert store.feedback_log_path.exists()
    assert store.hard_cases_path.exists()


def test_record_unhelpful_feedback_without_refs_skips_hard_case(tmp_path):
    store = FeedbackLoopStore(root_dir=str(tmp_path))
    record = FeedbackRecord(
        query="why is this answer wrong",
        answer="still wrong",
        label="unhelpful",
        references=[],
    )

    report = store.record_feedback(record, default_top_k=3, retrieve=None)

    assert report.hard_case_added is False
    assert report.benchmark_case_count == 0
    assert report.benchmark_report is None
    assert store.feedback_log_path.exists()
    assert not store.hard_cases_path.exists()
