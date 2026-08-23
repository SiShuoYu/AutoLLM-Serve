import json
from pathlib import Path
from typing import Any

import pytest

from citetune.candidate_generation import (
    CandidateGenerationConfig,
    generate_answerable_candidates,
    generation_preflight,
    load_candidate_generation_config,
)


class FakeGenerator:
    author_id = "fake-generator"

    def generate(self, task: dict[str, Any]) -> tuple[str, str]:
        return f"{task['task_id']} 的问题？", "仅来自证据的答案。"


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
    ]
    queue.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "candidates.jsonl"
    assert generate_answerable_candidates(queue, output, FakeGenerator(), limit=10) == 1
    assert generate_answerable_candidates(queue, output, FakeGenerator(), limit=10) == 0
    candidates = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert candidates[0]["reference_citation_ids"] == ["chunk-1"]
    assert candidates[0]["review"]["status"] == "needs_revision"

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
