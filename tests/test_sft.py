import json
from pathlib import Path

import pytest

from citetune.sft import export_candidate_smoke_sft, export_sft_dataset, verify_sft_jsonl


def _answerable(example_id: str, split: str) -> dict[str, object]:
    return {
        "example_id": example_id,
        "split": split,
        "question": "Pod 是什么？",
        "reference_answer": "Pod 是可部署单元。",
        "reference_citation_ids": ["pods"],
        "evidence": [
            {
                "evidence_id": "pods",
                "document_id": "doc-pods",
                "text": "Pod 是 Kubernetes 中最小的可部署单元。",
                "source_title": "pods",
            }
        ],
    }


def test_sft_export_includes_train_and_excludes_validation_and_test(tmp_path: Path) -> None:
    dataset = tmp_path / "reviewed.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in (
                _answerable("train-1", "train"),
                _answerable("validation-1", "validation"),
                _answerable("test-1", "test"),
                {
                    "example_id": "train-abstain",
                    "split": "train",
                    "question": "证据中没有答案的问题",
                    "reference_answer": "提供的证据不足以回答此问题。",
                    "reference_citation_ids": [],
                    "evidence": [],
                    "expected_behavior": "abstain_evidence_insufficient",
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "sft.jsonl"
    manifest = export_sft_dataset(dataset, output, tmp_path / "manifest.json")
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert manifest.exported_train_count == 2
    assert manifest.excluded_split_counts == {"test": 1, "validation": 1}
    assert all(row["metadata"]["split"] == "train" for row in rows)
    assert [message["role"] for message in rows[0]["prompt"]] == ["system", "user"]
    assert "[pods]" in rows[0]["completion"][0]["content"]
    assert "证据不足" in rows[1]["completion"][0]["content"]


def test_sft_verifier_rejects_non_train_rows(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps(
            {
                "prompt": [
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": "u"},
                ],
                "completion": [
                    {"role": "assistant", "content": "a"},
                ],
                "metadata": {"split": "test"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-train"):
        verify_sft_jsonl(path)


def test_candidate_smoke_export_is_train_only_and_explicitly_unreviewed(tmp_path: Path) -> None:
    queue = tmp_path / "queue.jsonl"
    queue.write_text(
        json.dumps(
            {
                "task_id": "train-answerable-0001",
                "split": "train",
                "task_type": "answerable",
                "source_chunk": {
                    "chunk_id": "chunk-1",
                    "document_id": "doc-1",
                    "text": "Pod 是 Kubernetes 最小的可部署单元。",
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
                "author_id": "model:generator",
                "question": "Pod 是什么？",
                "reference_answer": "Pod 是 Kubernetes 最小的可部署单元。",
                "reference_citation_ids": ["chunk-1"],
                "review": {"status": "needs_revision", "notes": "待审核"},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "smoke.jsonl"
    manifest = export_candidate_smoke_sft(
        queue, submissions, output, tmp_path / "smoke_manifest.json", limit=1
    )
    row = json.loads(output.read_text(encoding="utf-8"))
    assert manifest.purpose == "pipeline_smoke_only_unreviewed_candidate_drafts"
    assert manifest.exported_train_count == 1
    assert row["metadata"]["dataset_status"] == "candidate_smoke_unreviewed"
    assert row["metadata"]["purpose"] == "pipeline_smoke_only"
