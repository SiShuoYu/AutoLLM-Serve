import json
from pathlib import Path

from citetune.candidate_curation import (
    curate_heldout_candidate_drafts,
    curate_train_candidate_drafts,
)


def test_curation_rejects_only_extra_duplicates_and_generic_answers(tmp_path: Path) -> None:
    queue = tmp_path / "queue.jsonl"
    queue.write_text(
        "\n".join(
            json.dumps(
                {
                    "task_id": task_id,
                    "split": "train",
                    "task_type": "answerable",
                    "queue_role": role,
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
            for index, (task_id, role) in enumerate(
                (
                    ("train-answerable-0001", "primary"),
                    ("train-answerable-0002", "primary"),
                    ("train-answerable-0003", "primary"),
                ),
                1,
            )
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
                    "question": question,
                    "reference_answer": answer,
                    "reference_citation_ids": [f"chunk-{index}"],
                    "review": {"status": "needs_revision", "notes": "待审核"},
                },
                ensure_ascii=False,
            )
            for index, (task_id, question, answer) in enumerate(
                (
                    (
                        "train-answerable-0001",
                        "Pod 是什么？",
                        "Pod 是 Kubernetes 中最小的可部署单元。",
                    ),
                    (
                        "train-answerable-0002",
                        "Pod 是什么？",
                        "Pod 是 Kubernetes 中最小的可部署单元。",
                    ),
                    ("train-answerable-0003", "如何使用 Pod？", "抱歉，我无法回答。"),
                ),
                1,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "curated.jsonl"
    plan = curate_train_candidate_drafts(queue, submissions, output)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert plan.auto_rejected_task_count == 2
    assert plan.auto_rejection_counts == {
        "duplicate_question": 1,
        "generic_or_abstaining_answer": 1,
    }
    assert rows[0]["review"]["status"] == "needs_revision"
    assert rows[1]["review"]["status"] == "rejected"
    assert rows[2]["review"]["status"] == "rejected"


def test_heldout_curation_never_deduplicates_across_splits(tmp_path: Path) -> None:
    queue = tmp_path / "queue.jsonl"
    queue.write_text(
        "\n".join(
            json.dumps(
                {
                    "task_id": task_id,
                    "split": split,
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
            for index, (task_id, split) in enumerate(
                (("validation-answerable-0001", "validation"), ("test-answerable-0001", "test")), 1
            )
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
                    "reference_answer": "Pod 是 Kubernetes 中最小的可部署单元。",
                    "reference_citation_ids": [f"chunk-{index}"],
                    "review": {"status": "needs_revision", "notes": "待审核"},
                },
                ensure_ascii=False,
            )
            for index, task_id in enumerate(
                ("validation-answerable-0001", "test-answerable-0001"), 1
            )
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "curated.jsonl"
    plan = curate_heldout_candidate_drafts(queue, submissions, output)
    assert plan.auto_rejected_task_count == 0
    assert plan.replacement_requested_counts == {}
