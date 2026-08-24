import json
from pathlib import Path
from typing import Any

import pytest

from citetune.inference import GeneratedAnswer, generate_predictions, load_inference_config


class FakeGenerator:
    peak_gpu_memory_bytes: int | None = 123

    def generate(self, question: str, evidence: list[Any]) -> GeneratedAnswer:
        return GeneratedAnswer("Pod 是可部署单元。 [chunk-1]", 12.5)


def test_inference_generates_predictions_and_real_style_manifest(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps(
            {"chunk_id": "chunk-1", "text": "Pod 是可部署单元。", "source_url": "https://e.test/p"}
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "example_id": "test-1",
                "split": "test",
                "question": "Pod 是什么？",
                "reference_answer": "Pod 是可部署单元。",
                "reference_citation_ids": ["chunk-1"],
                "evidence": [
                    {
                        "evidence_id": "chunk-1",
                        "document_id": "doc",
                        "text": "Pod 是可部署单元。",
                        "source_title": "pods",
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "rag.yaml"
    config_path.write_text(
        f"""inference:
  system_name: rag-test
  mode: rag
  model:
    id: Qwen/Qwen2.5-1.5B-Instruct
    revision: 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
  adapter_path: null
  corpus_path: {corpus.name}
  top_k: 1
  max_new_tokens: 64
  temperature: 0.0
  seed: 42
""",
        encoding="utf-8",
    )
    output = tmp_path / "predictions.jsonl"
    report = generate_predictions(
        load_inference_config(config_path), dataset, output, FakeGenerator()
    )
    row = json.loads(output.read_text(encoding="utf-8"))
    assert report.generated_prediction_count == 1
    assert report.peak_gpu_memory_bytes == 123
    assert row["citation_ids"] == ["chunk-1"]
    assert row["answer"] == "Pod 是可部署单元。"


@pytest.mark.parametrize(
    "config_name, expected_mode, expects_adapter",
    [
        ("inference-base-qwen2.5-1.5b.yaml", "base", False),
        ("inference-rag-qwen2.5-1.5b.yaml", "rag", False),
        ("inference-qlora-qwen2.5-1.5b-synthetic-v1.yaml", "qlora", True),
        ("inference-rag-qlora-qwen2.5-1.5b-synthetic-v1.yaml", "rag_qlora", True),
    ],
)
def test_checked_in_inference_configs_define_the_comparison_matrix(
    config_name: str, expected_mode: str, expects_adapter: bool
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    config = load_inference_config(repository_root / "configs" / config_name)

    assert config.mode == expected_mode
    assert (config.adapter_path is not None) is expects_adapter
    assert config.temperature == 0.0
    assert config.seed == 42
    assert config.max_new_tokens == 256
