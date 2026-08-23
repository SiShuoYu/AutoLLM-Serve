import hashlib
import json
from pathlib import Path

import pytest

from citetune.training import load_qlora_config, training_preflight

MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


def _write_config(tmp_path: Path, revision: str = MODEL_REVISION) -> Path:
    config = tmp_path / "qlora.yaml"
    config.write_text(
        f"""experiment:
  name: pilot
  model:
    id: Qwen/Qwen2.5-1.5B-Instruct
    revision: {revision}
  data:
    train_path: sft.jsonl
    manifest_path: manifest.json
  output_dir: output
  seed: 42
  training:
    epochs: 1
    learning_rate: 0.0002
    batch_size: 1
    gradient_accumulation_steps: 8
    max_length: 512
    mixed_precision: none
  lora:
    r: 8
    alpha: 16
    dropout: 0.05
    target_modules: [q_proj, v_proj]
""",
        encoding="utf-8",
    )
    return config


def test_training_preflight_verifies_pinned_model_and_sft_hash(tmp_path: Path) -> None:
    sft = tmp_path / "sft.jsonl"
    sft.write_text(
        json.dumps(
            {
                "prompt": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "question"},
                ],
                "completion": [{"role": "assistant", "content": "answer"}],
                "metadata": {"split": "train"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(sft.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps({"output_sha256": digest}), encoding="utf-8")
    config = load_qlora_config(_write_config(tmp_path))
    report = training_preflight(config)
    assert report.model_revision == MODEL_REVISION
    assert report.verified_train_rows == 1
    assert report.manifest_matches_train is True
    assert report.gpu_required is True
    assert config.mixed_precision == "none"


def test_training_config_requires_full_model_commit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="full 40-character"):
        load_qlora_config(_write_config(tmp_path, revision="main"))


def test_training_config_rejects_unknown_precision_mode(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    path.write_text(path.read_text(encoding="utf-8").replace("none", "bf16"), encoding="utf-8")
    with pytest.raises(ValueError, match="mixed_precision"):
        load_qlora_config(path)
