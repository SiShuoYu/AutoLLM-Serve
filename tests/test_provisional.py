import json
from pathlib import Path

from citetune.provisional import export_provisional_heldout_dataset


def test_provisional_heldout_export_is_explicitly_not_a_quality_benchmark(tmp_path: Path) -> None:
    queue = tmp_path / "queue.jsonl"
    queue.write_text(
        "\n".join(
            json.dumps(
                {
                    "task_id": f"{split}-answerable-0001",
                    "split": split,
                    "task_type": "answerable",
                    "source_chunk": {
                        "chunk_id": f"{split}-chunk",
                        "document_id": f"{split}-document",
                        "text": "这是一段可由问题回答的中文证据。",
                        "source_path": "source.md",
                        "source_url": "https://example.test/source",
                    },
                },
                ensure_ascii=False,
            )
            for split in ("train", "validation", "test")
        )
        + "\n",
        encoding="utf-8",
    )
    submissions = tmp_path / "submissions.jsonl"
    submissions.write_text(
        "\n".join(
            json.dumps(
                {
                    "task_id": f"{split}-answerable-0001",
                    "author_id": "model:generator",
                    "question": "证据说明了什么？",
                    "reference_answer": "证据说明了可回答的问题。",
                    "reference_citation_ids": [f"{split}-chunk"],
                    "review": {"status": "needs_revision", "notes": "未审核草稿"},
                },
                ensure_ascii=False,
            )
            for split in ("train", "validation", "test")
        )
        + "\n",
        encoding="utf-8",
    )

    output = tmp_path / "provisional.jsonl"
    manifest = export_provisional_heldout_dataset(
        queue, submissions, output, tmp_path / "manifest.json"
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert manifest.exported_split_counts == {"test": 1, "validation": 1}
    assert {row["split"] for row in rows} == {"validation", "test"}
    assert all(
        row["dataset_role"] == "unreviewed_heldout_drafts_latency_only_not_quality_benchmark"
        for row in rows
    )


def test_provisional_heldout_export_can_create_a_balanced_operational_preflight(
    tmp_path: Path,
) -> None:
    queue = tmp_path / "queue.jsonl"
    submissions = tmp_path / "submissions.jsonl"
    queue_rows = []
    submission_rows = []
    for split in ("validation", "test"):
        for ordinal in range(3):
            task_id = f"{split}-answerable-{ordinal:04d}"
            queue_rows.append(
                {
                    "task_id": task_id,
                    "split": split,
                    "task_type": "answerable",
                    "source_chunk": {
                        "chunk_id": task_id,
                        "document_id": "document",
                        "text": "一段中文证据。",
                        "source_path": "source.md",
                        "source_url": "https://example.test/source",
                    },
                }
            )
            submission_rows.append(
                {
                    "task_id": task_id,
                    "author_id": "model:generator",
                    "question": "问题？",
                    "reference_answer": "答案。",
                    "reference_citation_ids": [task_id],
                    "review": {"status": "needs_revision", "notes": "未审核草稿"},
                }
            )
    queue.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in queue_rows) + "\n",
        encoding="utf-8",
    )
    submissions.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in submission_rows) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "preflight.jsonl"
    manifest = export_provisional_heldout_dataset(
        queue,
        submissions,
        output,
        tmp_path / "manifest.json",
        limit_per_split=2,
    )

    assert manifest.limit_per_split == 2
    assert manifest.exported_split_counts == {"test": 2, "validation": 2}
    assert len(output.read_text(encoding="utf-8").splitlines()) == 4
