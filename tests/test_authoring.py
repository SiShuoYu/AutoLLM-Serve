import json
from pathlib import Path

import pytest

from citetune.authoring import (
    build_authoring_queue,
    compile_reviewed_dataset,
    render_review_packet,
    validate_authoring_submissions,
)
from citetune.dataset import load_dataset


def test_authoring_queue_locks_sources_to_their_existing_splits(tmp_path: Path) -> None:
    splits = tmp_path / "splits"
    splits.mkdir()
    for split in ("train", "validation", "test"):
        with (splits / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for number in range(3):
                handle.write(json.dumps({"chunk_id": f"{split}-{number}", "text": "证据"}) + "\n")
    output = tmp_path / "authoring.jsonl"
    manifest = build_authoring_queue(
        splits,
        output,
        tmp_path / "authoring-manifest.json",
        train_count=2,
        validation_count=1,
        test_answerable_count=2,
        train_insufficient_evidence_count=1,
        validation_insufficient_evidence_count=1,
        test_insufficient_evidence_count=1,
        train_reserve_count=1,
        validation_reserve_count=1,
        test_reserve_count=1,
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    answerable = [row for row in rows if row["task_type"] == "answerable"]
    assert len(answerable) == 8
    assert all(row["source_chunk"]["chunk_id"].startswith(row["split"]) for row in answerable)
    assert len([row for row in rows if row["task_type"] == "insufficient_evidence"]) == 3
    assert manifest.answerable_task_counts == {"train": 2, "validation": 1, "test": 2}
    assert manifest.reserve_task_counts == {"train": 1, "validation": 1, "test": 1}
    assert manifest.insufficient_evidence_task_counts == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }
    assert len([row for row in rows if row["queue_role"] == "reserve"]) == 3


def test_authoring_queue_reduces_reserves_without_reusing_source_chunks(tmp_path: Path) -> None:
    splits = tmp_path / "splits"
    splits.mkdir()
    for split, count in (("train", 3), ("validation", 2), ("test", 2)):
        (splits / f"{split}.jsonl").write_text(
            "\n".join(
                json.dumps({"chunk_id": f"{split}-{number}", "text": "证据"})
                for number in range(count)
            )
            + "\n",
            encoding="utf-8",
        )
    output = tmp_path / "authoring.jsonl"
    manifest = build_authoring_queue(
        splits,
        output,
        tmp_path / "authoring-manifest.json",
        train_count=2,
        validation_count=1,
        test_answerable_count=1,
        train_reserve_count=4,
        validation_reserve_count=4,
        test_reserve_count=4,
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    answerable = [row for row in rows if row["task_type"] == "answerable"]
    assert manifest.reserve_task_counts == {"train": 1, "validation": 1, "test": 1}
    assert len({row["source_chunk"]["chunk_id"] for row in answerable}) == len(answerable)


def test_reviewed_authoring_must_match_locked_evidence(tmp_path: Path) -> None:
    queue = tmp_path / "queue.jsonl"
    queue.write_text(
        json.dumps(
            {
                "task_id": "train-answerable-0001",
                "split": "train",
                "task_type": "answerable",
                "source_chunk": {
                    "chunk_id": "chunk-1",
                    "document_id": "document-1",
                    "text": "证据内容",
                    "source_path": "content/zh-cn/docs/example.md",
                    "source_url": "https://example.test/example.md",
                },
            }
        )
        + "\n"
        + json.dumps(
            {
                "task_id": "test-insufficient-evidence-0001",
                "split": "test",
                "task_type": "insufficient_evidence",
                "source_chunk": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    submissions = tmp_path / "reviewed.jsonl"
    submissions.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in (
                {
                    "task_id": "train-answerable-0001",
                    "question": "问题",
                    "reference_answer": "回答",
                    "reference_citation_ids": ["chunk-1"],
                    "author_id": "author",
                    "review": {
                        "status": "approved",
                        "reviewer_id": "reviewer",
                        "reviewer_type": "human",
                        "evidence_supported": True,
                    },
                },
                {
                    "task_id": "test-insufficient-evidence-0001",
                    "question": "问题",
                    "reference_answer": "提供的证据不足以回答此问题。",
                    "reference_citation_ids": [],
                    "author_id": "author",
                    "review": {
                        "status": "approved",
                        "reviewer_id": "reviewer",
                        "reviewer_type": "human",
                        "evidence_supported": True,
                    },
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    report = validate_authoring_submissions(queue, submissions, require_complete=True)
    assert report.approved_task_count == 2

    compiled = tmp_path / "dataset.jsonl"
    compile_reviewed_dataset(queue, submissions, compiled, require_complete=True)
    examples = load_dataset(compiled)
    assert examples[0].expected_behavior == "answer"
    assert examples[1].expected_behavior == "abstain_evidence_insufficient"
    assert examples[1].evidence == ()

    invalid = submissions.read_text(encoding="utf-8").replace('"chunk-1"', '"other-chunk"')
    submissions.write_text(invalid, encoding="utf-8")
    with pytest.raises(ValueError, match="assigned source"):
        validate_authoring_submissions(queue, submissions)


def test_rejected_source_needs_reason_but_not_invented_qa(tmp_path: Path) -> None:
    queue = tmp_path / "queue.jsonl"
    queue.write_text(
        json.dumps(
            {
                "task_id": "train-answerable-0001",
                "split": "train",
                "task_type": "answerable",
                "source_chunk": {"chunk_id": "ambiguous"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    submission = tmp_path / "submission.jsonl"
    submission.write_text(
        json.dumps(
            {
                "task_id": "train-answerable-0001",
                "author_id": "author",
                "review": {"status": "rejected", "notes": "上下文不完整"},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    report = validate_authoring_submissions(queue, submission)
    assert report.approved_task_count == 0
    assert report.submitted_task_count == 1


def test_review_packet_contains_question_answer_and_assigned_evidence(tmp_path: Path) -> None:
    queue = tmp_path / "queue.jsonl"
    queue.write_text(
        json.dumps(
            {
                "task_id": "train-answerable-0001",
                "split": "train",
                "queue_role": "primary",
                "task_type": "answerable",
                "source_chunk": {
                    "chunk_id": "chunk-1",
                    "document_id": "document-1",
                    "text": "Pod 是可部署单元。   \n   \n第二行\t",
                    "source_path": "pods.md",
                    "source_url": "https://example.test/pods",
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    submissions = tmp_path / "candidates.jsonl"
    submissions.write_text(
        json.dumps(
            {
                "task_id": "train-answerable-0001",
                "question": "Pod 是什么？",
                "reference_answer": "Pod 是可部署单元。",
                "reference_citation_ids": ["chunk-1"],
                "author_id": "candidate-author",
                "review": {"status": "needs_revision", "notes": "待审核"},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "review.md"
    render_review_packet(queue, submissions, output)
    rendered = output.read_text(encoding="utf-8")
    assert "Pod 是什么？" in rendered
    assert "Pod 是可部署单元。" in rendered
    assert "批准：问题自然" in rendered
    assert all(line == line.rstrip() for line in rendered.splitlines())


def test_held_out_approval_requires_independent_human_reviewer(tmp_path: Path) -> None:
    queue = tmp_path / "queue.jsonl"
    queue.write_text(
        json.dumps(
            {
                "task_id": "test-answerable-0001",
                "split": "test",
                "task_type": "answerable",
                "source_chunk": {
                    "chunk_id": "chunk-1",
                    "document_id": "doc-1",
                    "text": "证据",
                    "source_path": "doc.md",
                    "source_url": "https://example.test/doc",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    submissions = tmp_path / "submission.jsonl"
    submissions.write_text(
        json.dumps(
            {
                "task_id": "test-answerable-0001",
                "question": "问题",
                "reference_answer": "答案",
                "reference_citation_ids": ["chunk-1"],
                "author_id": "author",
                "review": {
                    "status": "approved",
                    "reviewer_id": "reviewer",
                    "reviewer_type": "model",
                    "evidence_supported": True,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="human reviewer"):
        validate_authoring_submissions(queue, submissions)
