import json
from pathlib import Path

from citetune.candidate_audit import audit_candidate_drafts


def test_candidate_audit_reports_risks_and_writes_bounded_packet(tmp_path: Path) -> None:
    queue = tmp_path / "queue.jsonl"
    queue.write_text(
        "\n".join(
            json.dumps(
                {
                    "task_id": task_id,
                    "split": "train",
                    "task_type": "answerable",
                    "source_chunk": {
                        "chunk_id": f"chunk-{index}",
                        "document_id": "doc",
                        "text": "Pod 是 Kubernetes 中最小的可部署单元。",
                        "source_path": "pods.md",
                        "source_url": "https://example.test/pods",
                    },
                },
                ensure_ascii=False,
            )
            for index, task_id in enumerate(("train-answerable-0001", "train-answerable-0002"), 1)
        )
        + "\n",
        encoding="utf-8",
    )
    submissions = tmp_path / "candidates.jsonl"
    submissions.write_text(
        "\n".join(
            json.dumps(
                {
                    "task_id": task_id,
                    "author_id": "model:generator",
                    "question": "Pod 是什么？",
                    "reference_answer": answer,
                    "reference_citation_ids": [f"chunk-{index}"],
                    "review": {"status": "needs_revision", "notes": "待审核"},
                },
                ensure_ascii=False,
            )
            for index, (task_id, answer) in enumerate(
                (
                    ("train-answerable-0001", "Pod 是 Kubernetes 最小的可部署单元。"),
                    ("train-answerable-0002", "短答"),
                ),
                1,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    report = audit_candidate_drafts(
        queue, submissions, tmp_path / "report.json", tmp_path / "packet.md", review_limit=1
    )
    assert report.screened_candidate_count == 2
    assert report.flagged_task_count == 2
    assert report.issue_counts["duplicate_question"] == 2
    assert report.issue_counts["answer_too_short"] == 1
    packet = (tmp_path / "packet.md").read_text(encoding="utf-8")
    assert packet.count("## train-answerable") == 1
