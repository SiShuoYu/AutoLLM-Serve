"""Leakage-safe export of approved training examples to Qwen chat SFT JSONL."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .authoring import validate_authoring_submissions
from .dataset import load_dataset
from .schemas import GroundedExample

SFT_FORMAT_VERSION = "qwen-prompt-completion-v1"
SYSTEM_PROMPT = (
    "你是 Kubernetes 中文文档问答助手。只使用给定证据回答；"
    "证据不足时明确回答‘提供的证据不足以回答此问题’，不要猜测。"
    "回答末尾列出引用 ID。"
)


@dataclass(frozen=True, slots=True)
class SFTExportManifest:
    input_sha256: str
    output_sha256: str
    format_version: str
    system_prompt_sha256: str
    exported_train_count: int
    exported_behavior_counts: dict[str, int]
    excluded_split_counts: dict[str, int]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CandidateSmokeExportManifest:
    """Provenance for an explicitly non-final candidate-draft smoke dataset."""

    purpose: str
    queue_sha256: str
    submissions_sha256: str
    output_sha256: str
    format_version: str
    system_prompt_sha256: str
    selected_task_ids_sha256: str
    exported_train_count: int
    seed: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _as_chat_row(example: GroundedExample) -> dict[str, object]:
    if example.split != "train":
        raise ValueError(f"SFT export received non-train example: {example.example_id}")
    if example.evidence:
        evidence_text = "\n\n".join(
            f"[{item.evidence_id}] {item.text}" for item in example.evidence
        )
    else:
        evidence_text = "（无可用证据）"
    citation_suffix = (
        "\n\n引用：" + " ".join(f"[{item}]" for item in example.reference_citation_ids)
        if example.reference_citation_ids
        else ""
    )
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"问题：{example.question}\n\n证据：\n{evidence_text}",
            },
        ],
        "completion": [
            {
                "role": "assistant",
                "content": f"{example.reference_answer}{citation_suffix}",
            },
        ],
        "metadata": {
            "example_id": example.example_id,
            "split": example.split,
            "expected_behavior": example.expected_behavior,
        },
    }


def verify_sft_jsonl(path: str | Path) -> int:
    """Fail closed if any exported record is not explicitly marked as training data."""
    count = 0
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row: Any = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid SFT JSON on line {line_number}") from error
        if not isinstance(row, dict) or not isinstance(row.get("metadata"), dict):
            raise ValueError(f"SFT line {line_number} has no metadata")
        if row["metadata"].get("split") != "train":
            raise ValueError(f"SFT line {line_number} contains non-train data")
        prompt = row.get("prompt")
        completion = row.get("completion")
        if not isinstance(prompt, list) or [item.get("role") for item in prompt] != [
            "system",
            "user",
        ]:
            raise ValueError(f"SFT line {line_number} has invalid chat prompt")
        if not isinstance(completion, list) or [item.get("role") for item in completion] != [
            "assistant"
        ]:
            raise ValueError(f"SFT line {line_number} has invalid chat completion")
        count += 1
    if not count:
        raise ValueError("SFT dataset is empty")
    return count


def export_sft_dataset(
    dataset_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
) -> SFTExportManifest:
    """Export only train rows and record every excluded split."""
    dataset = Path(dataset_path)
    examples = load_dataset(dataset)
    training = [example for example in examples if example.split == "train"]
    if not training:
        raise ValueError("dataset contains no approved train examples")
    excluded = Counter(example.split for example in examples if example.split != "train")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for example in training:
            handle.write(
                json.dumps(_as_chat_row(example), ensure_ascii=False, sort_keys=True) + "\n"
            )
    exported_count = verify_sft_jsonl(output)
    manifest = SFTExportManifest(
        input_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
        output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        format_version=SFT_FORMAT_VERSION,
        system_prompt_sha256=hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        exported_train_count=exported_count,
        exported_behavior_counts=dict(
            sorted(Counter(example.expected_behavior for example in training).items())
        ),
        excluded_split_counts=dict(sorted(excluded.items())),
    )
    manifest_file = Path(manifest_path)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(
        json.dumps(manifest.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def export_candidate_smoke_sft(
    queue_path: str | Path,
    submissions_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
    *,
    limit: int = 100,
    seed: int = 42,
) -> CandidateSmokeExportManifest:
    """Export a small, unreviewed train-only set solely to test the GPU path.

    This intentionally has a separate command, manifest, format marker, and
    output location from ``export_sft_dataset``. It may not be cited as a
    quality experiment or used for validation/test reporting.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")
    queue = Path(queue_path)
    submissions = Path(submissions_path)
    validate_authoring_submissions(queue, submissions)
    queue_rows = {
        row["task_id"]: row
        for row in (json.loads(line) for line in queue.read_text(encoding="utf-8").splitlines())
        if row
    }
    candidate_rows = [
        row
        for row in (
            json.loads(line) for line in submissions.read_text(encoding="utf-8").splitlines()
        )
        if row
        and row.get("review", {}).get("status") == "needs_revision"
        and (task := queue_rows.get(row.get("task_id"))) is not None
        and task.get("split") == "train"
        and task.get("task_type") == "answerable"
    ]
    selected = sorted(
        candidate_rows,
        key=lambda row: hashlib.sha256(f"{seed}:{row['task_id']}".encode()).hexdigest(),
    )[:limit]
    if not selected:
        raise ValueError("no train candidate drafts are available for smoke export")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for candidate in selected:
            task = queue_rows[candidate["task_id"]]
            source = task["source_chunk"]
            example = GroundedExample.from_dict(
                {
                    "example_id": candidate["task_id"],
                    "split": "train",
                    "question": candidate["question"],
                    "reference_answer": candidate["reference_answer"],
                    "reference_citation_ids": candidate["reference_citation_ids"],
                    "evidence": [
                        {
                            "evidence_id": source["chunk_id"],
                            "document_id": source["document_id"],
                            "text": source["text"],
                            "source_title": source["source_path"],
                            "source_url": source["source_url"],
                        }
                    ],
                }
            )
            chat_row = _as_chat_row(example)
            metadata = chat_row["metadata"]
            assert isinstance(metadata, dict)
            metadata["dataset_status"] = "candidate_smoke_unreviewed"
            metadata["purpose"] = "pipeline_smoke_only"
            handle.write(json.dumps(chat_row, ensure_ascii=False, sort_keys=True) + "\n")

    exported_count = verify_sft_jsonl(output)
    selected_ids = "\n".join(row["task_id"] for row in selected).encode()
    manifest = CandidateSmokeExportManifest(
        purpose="pipeline_smoke_only_unreviewed_candidate_drafts",
        queue_sha256=hashlib.sha256(queue.read_bytes()).hexdigest(),
        submissions_sha256=hashlib.sha256(submissions.read_bytes()).hexdigest(),
        output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        format_version="qwen-candidate-smoke-unreviewed-v1",
        system_prompt_sha256=hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        selected_task_ids_sha256=hashlib.sha256(selected_ids).hexdigest(),
        exported_train_count=exported_count,
        seed=seed,
    )
    manifest_file = Path(manifest_path)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(
        json.dumps(manifest.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
