"""Leakage-safe export of approved training examples to Qwen chat SFT JSONL."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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
