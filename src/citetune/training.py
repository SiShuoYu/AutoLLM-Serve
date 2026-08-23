"""QLoRA configuration, CPU-safe preflight, and explicit GPU training entrypoint."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .sft import verify_sft_jsonl


@dataclass(frozen=True, slots=True)
class QLoRAConfig:
    name: str
    model_id: str
    model_revision: str
    train_path: Path
    train_manifest_path: Path
    output_dir: Path
    seed: int
    epochs: float
    learning_rate: float
    batch_size: int
    gradient_accumulation_steps: int
    max_length: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TrainingPreflight:
    name: str
    model_id: str
    model_revision: str
    verified_train_rows: int
    train_sha256: str
    manifest_matches_train: bool
    gpu_required: bool
    gpu_dependencies_present: dict[str, bool]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _positive_number(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be positive")
    return float(value)


def _positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def load_qlora_config(path: str | Path) -> QLoRAConfig:
    config_path = Path(path)
    raw = _mapping(yaml.safe_load(config_path.read_text(encoding="utf-8")), "config")
    experiment = _mapping(raw.get("experiment"), "experiment")
    model = _mapping(experiment.get("model"), "experiment.model")
    data = _mapping(experiment.get("data"), "experiment.data")
    training = _mapping(experiment.get("training"), "experiment.training")
    lora = _mapping(experiment.get("lora"), "experiment.lora")
    revision = _text(model.get("revision"), "experiment.model.revision")
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("experiment.model.revision must be a full 40-character commit hash")
    target_modules = lora.get("target_modules")
    if (
        not isinstance(target_modules, list)
        or not target_modules
        or any(not isinstance(item, str) or not item for item in target_modules)
    ):
        raise ValueError("experiment.lora.target_modules must contain module names")
    root = config_path.parent
    seed = experiment.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("experiment.seed must be an integer")
    batch_size = _positive_int(training.get("batch_size"), "experiment.training.batch_size")
    accumulation = _positive_int(
        training.get("gradient_accumulation_steps"),
        "experiment.training.gradient_accumulation_steps",
    )
    max_length = _positive_int(training.get("max_length"), "experiment.training.max_length")
    lora_r = _positive_int(lora.get("r"), "experiment.lora.r")
    lora_alpha = _positive_int(lora.get("alpha"), "experiment.lora.alpha")
    dropout = lora.get("dropout")
    if not isinstance(dropout, (int, float)) or isinstance(dropout, bool) or not 0 <= dropout < 1:
        raise ValueError("experiment.lora.dropout must be in [0, 1)")
    return QLoRAConfig(
        name=_text(experiment.get("name"), "experiment.name"),
        model_id=_text(model.get("id"), "experiment.model.id"),
        model_revision=revision,
        train_path=(root / _text(data.get("train_path"), "experiment.data.train_path")).resolve(),
        train_manifest_path=(
            root / _text(data.get("manifest_path"), "experiment.data.manifest_path")
        ).resolve(),
        output_dir=(root / _text(experiment.get("output_dir"), "experiment.output_dir")).resolve(),
        seed=seed,
        epochs=_positive_number(training.get("epochs"), "experiment.training.epochs"),
        learning_rate=_positive_number(
            training.get("learning_rate"), "experiment.training.learning_rate"
        ),
        batch_size=batch_size,
        gradient_accumulation_steps=accumulation,
        max_length=max_length,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=float(dropout),
        target_modules=tuple(target_modules),
    )


def training_preflight(config: QLoRAConfig) -> TrainingPreflight:
    row_count = verify_sft_jsonl(config.train_path)
    manifest = json.loads(config.train_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("output_sha256"), str):
        raise ValueError("SFT manifest has no output_sha256")
    train_hash = hashlib.sha256(config.train_path.read_bytes()).hexdigest()
    if manifest["output_sha256"] != train_hash:
        raise ValueError("SFT manifest hash does not match training data")
    dependencies = {
        package: importlib.util.find_spec(package) is not None
        for package in ("torch", "transformers", "datasets", "peft", "trl", "bitsandbytes")
    }
    return TrainingPreflight(
        name=config.name,
        model_id=config.model_id,
        model_revision=config.model_revision,
        verified_train_rows=row_count,
        train_sha256=train_hash,
        manifest_matches_train=True,
        gpu_required=True,
        gpu_dependencies_present=dependencies,
    )


def run_qlora_training(config: QLoRAConfig) -> None:
    """Run the pinned QLoRA experiment; this is intentionally GPU-only."""
    training_preflight(config)
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as error:
        raise RuntimeError(
            "install the project training dependencies before GPU training"
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA execution requires a CUDA GPU; run preflight on CPU instead")
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.model_revision)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        quantization_config=quantization,
        device_map={"": 0},
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=list(config.target_modules),
    )
    train_dataset = load_dataset("json", data_files=str(config.train_path), split="train")
    arguments = SFTConfig(
        output_dir=str(config.output_dir),
        seed=config.seed,
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_length=config.max_length,
        gradient_checkpointing=True,
        fp16=True,
        bf16=False,
        completion_only_loss=True,
        packing=False,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(config.output_dir / "adapter"))
    tokenizer.save_pretrained(str(config.output_dir / "adapter"))
