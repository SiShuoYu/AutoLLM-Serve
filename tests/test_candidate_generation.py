import json
from pathlib import Path
from typing import Any

import pytest

from citetune.candidate_generation import (
    CandidateGenerationConfig,
    generate_answerable_candidates,
    generation_preflight,
    load_candidate_generation_config,
    parse_candidate_output,
)


class FakeGenerator:
    author_id = "fake-generator"

    def generate(self, task: dict[str, Any]) -> tuple[str, str]:
        return f"{task['task_id']} 的问题？", "仅来自证据的答案。"


class InvalidGenerator:
    author_id = "invalid-generator"

    def generate(self, task: dict[str, Any]) -> tuple[str, str]:
        raise ValueError("missing answer")


def test_generation_is_resumable_and_skips_insufficient_evidence(tmp_path: Path) -> None:
    queue = tmp_path / "queue.jsonl"
    rows = [
        {
            "task_id": "train-answerable-0001",
            "split": "train",
            "queue_role": "primary",
            "task_type": "answerable",
            "source_chunk": {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "text": "证据一",
                "source_path": "one.md",
                "source_url": "https://example.test/one",
            },
        },
        {
            "task_id": "train-insufficient-evidence-0001",
            "split": "train",
            "queue_role": "primary",
            "task_type": "insufficient_evidence",
            "source_chunk": None,
        },
        {
            "task_id": "train-answerable-reserve-0001",
            "split": "train",
            "queue_role": "reserve",
            "task_type": "answerable",
            "source_chunk": {
                "chunk_id": "chunk-reserve",
                "document_id": "doc-reserve",
                "text": "预留证据",
                "source_path": "reserve.md",
                "source_url": "https://example.test/reserve",
            },
        },
    ]
    queue.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "candidates.jsonl"
    assert generate_answerable_candidates(queue, output, FakeGenerator(), limit=10) == 1
    assert generate_answerable_candidates(queue, output, FakeGenerator(), limit=10) == 0
    assert (
        generate_answerable_candidates(
            queue, output, FakeGenerator(), limit=10, include_reserves=True
        )
        == 1
    )
    candidates = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert candidates[0]["reference_citation_ids"] == ["chunk-1"]
    assert candidates[0]["review"]["status"] == "needs_revision"
    assert candidates[1]["reference_citation_ids"] == ["chunk-reserve"]


def test_generation_records_unparsable_model_output_and_continues(tmp_path: Path) -> None:
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
                    "document_id": "doc-1",
                    "text": "证据一",
                    "source_path": "one.md",
                    "source_url": "https://example.test/one",
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "candidates.jsonl"
    assert generate_answerable_candidates(queue, output, InvalidGenerator(), limit=1) == 0
    rejected = json.loads(output.read_text(encoding="utf-8"))
    assert rejected["review"]["status"] == "rejected"
    assert "question" not in rejected

    config = CandidateGenerationConfig(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        model_revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        max_new_tokens=128,
        temperature=0.0,
        seed=42,
    )
    report = generation_preflight(config, queue, output)
    assert report.eligible_answerable_tasks == 1
    assert report.remaining_answerable_tasks == 0
    assert report.gpu_required is True


def test_generation_config_requires_pinned_model_revision(tmp_path: Path) -> None:
    config = tmp_path / "generation.yaml"
    config.write_text(
        """generation:
  model_id: Qwen/Qwen2.5-1.5B-Instruct
  model_revision: 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
  max_new_tokens: 256
  temperature: 0.0
  seed: 42
""",
        encoding="utf-8",
    )
    loaded = load_candidate_generation_config(config)
    assert loaded.max_new_tokens == 256
    assert loaded.temperature == 0.0

    config.write_text(config.read_text(encoding="utf-8").replace(loaded.model_revision, "main"))
    with pytest.raises(ValueError, match="full commit hash"):
        load_candidate_generation_config(config)


def test_candidate_output_parser_accepts_json_and_labelled_fallback() -> None:
    assert parse_candidate_output(
        '```json\n{"question": "Pod 是什么？", "answer": "可部署单元。"}\n```'
    ) == (
        "Pod 是什么？",
        "可部署单元。",
    )
    assert parse_candidate_output("**问题**：Pod 是什么？ **答案**：Pod 是可部署单元。") == (
        "Pod 是什么？",
        "Pod 是可部署单元。",
    )
    with pytest.raises(ValueError, match="no parseable"):
        parse_candidate_output("我无法生成问题")
