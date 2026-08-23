import json
from pathlib import Path
from typing import Any

import pytest

from citetune.model_review import (
    ReviewDecision,
    load_model_review_config,
    parse_review_output,
    review_train_candidates,
)


def _queue_row(task_id: str, split: str) -> dict[str, object]:
    return {
        "task_id": task_id,
        "split": split,
        "task_type": "answerable",
        "source_chunk": {
            "chunk_id": f"chunk-{task_id}",
            "document_id": "doc",
            "text": "Pod 是 Kubernetes 中最小的可部署单元。",
            "source_path": "pods.md",
            "source_url": "https://example.test/pods",
        },
    }


class ApprovingReviewer:
    reviewer_id = "model:independent-reviewer@1234567890123456789012345678901234567890"

    def review(self, task: dict[str, Any], candidate: dict[str, Any]) -> ReviewDecision:
        return ReviewDecision("approved", True, "答案受指定证据支持。")


def test_model_review_updates_train_but_not_held_out_rows(tmp_path: Path) -> None:
    queue = tmp_path / "queue.jsonl"
    queue.write_text(
        "\n".join(
            json.dumps(_queue_row(task_id, split), ensure_ascii=False)
            for task_id, split in (
                ("train-answerable-0001", "train"),
                ("test-answerable-0001", "test"),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    submissions = tmp_path / "drafts.jsonl"
    submissions.write_text(
        "\n".join(
            json.dumps(
                {
                    "task_id": task_id,
                    "author_id": "model:draft-generator",
                    "question": "Pod 是什么？",
                    "reference_answer": "Pod 是 Kubernetes 中最小的可部署单元。",
                    "reference_citation_ids": [f"chunk-{task_id}"],
                    "review": {"status": "needs_revision", "notes": "待审核"},
                },
                ensure_ascii=False,
            )
            for task_id in ("train-answerable-0001", "test-answerable-0001")
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "reviewed.jsonl"
    report = review_train_candidates(queue, submissions, output, ApprovingReviewer())
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert report.reviewed_train_candidate_count == 1
    assert report.verdict_counts == {"approved": 1}
    assert rows[0]["review"]["reviewer_type"] == "model"
    assert rows[1]["review"]["status"] == "needs_revision"


def test_review_output_parser_requires_supported_approval() -> None:
    assert parse_review_output(
        '{"verdict":"approved","evidence_supported":true,"notes":"证据充分"}'
    ) == ReviewDecision("approved", True, "证据充分")
    with pytest.raises(ValueError, match="evidence supported"):
        parse_review_output('{"verdict":"approved","evidence_supported":false,"notes":"矛盾"}')


def test_review_config_requires_pinned_revision(tmp_path: Path) -> None:
    config = tmp_path / "review.yaml"
    config.write_text(
        """review:
  model_id: Qwen/Qwen2.5-0.5B-Instruct
  model_revision: main
  max_new_tokens: 96
  temperature: 0.0
  seed: 42
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="full commit hash"):
        load_model_review_config(config)
