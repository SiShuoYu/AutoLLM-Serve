import json
from pathlib import Path

from citetune.readiness import assess_data_readiness


def test_readiness_reports_missing_primary_data_and_ignores_reserves(tmp_path: Path) -> None:
    queue = tmp_path / "queue.jsonl"
    queue_rows = [
        {
            "task_id": "train-answerable-0001",
            "split": "train",
            "task_type": "answerable",
            "queue_role": "primary",
        },
        {
            "task_id": "train-answerable-reserve-0001",
            "split": "train",
            "task_type": "answerable",
            "queue_role": "reserve",
        },
        {
            "task_id": "test-insufficient-evidence-0001",
            "split": "test",
            "task_type": "insufficient_evidence",
            "queue_role": "primary",
        },
    ]
    queue.write_text("\n".join(json.dumps(row) for row in queue_rows) + "\n", encoding="utf-8")
    submissions = tmp_path / "submissions.jsonl"
    submissions.write_text(
        json.dumps(
            {
                "task_id": "train-answerable-0001",
                "review": {"status": "approved", "reviewer_type": "model"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report = assess_data_readiness(queue, submissions)
    assert report.approved_counts["train:answerable"] == 1
    assert report.missing_approved_counts["test:insufficient_evidence"] == 1
    assert report.ready_for_gpu is False
