"""Reproducible evaluation run creation and result artifact persistence."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

from .dataset import dataset_manifest, load_dataset, load_predictions
from .evaluation import EvaluationSummary, evaluate_dataset


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    name: str
    dataset_path: Path
    predictions_path: Path
    output_dir: Path
    seed: int


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("experiment"), dict):
        raise ValueError("config must contain an experiment mapping")
    experiment = raw["experiment"]
    if not isinstance(experiment.get("name"), str) or not experiment["name"].strip():
        raise ValueError("experiment.name must be a non-empty string")
    if not isinstance(experiment.get("seed"), int):
        raise ValueError("experiment.seed must be an integer")
    fields = ("dataset_path", "predictions_path", "output_dir")
    if not all(isinstance(experiment.get(field), str) and experiment[field] for field in fields):
        raise ValueError("experiment must include dataset_path, predictions_path, and output_dir")
    root = config_path.parent
    return ExperimentConfig(
        name=experiment["name"].strip(),
        dataset_path=(root / experiment["dataset_path"]).resolve(),
        predictions_path=(root / experiment["predictions_path"]).resolve(),
        output_dir=(root / experiment["output_dir"]).resolve(),
        seed=experiment["seed"],
    )


def _git_revision() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, check=True, text=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_experiment(config: ExperimentConfig) -> EvaluationSummary:
    """Evaluate one prediction set and persist auditable, immutable-style artifacts."""
    examples = load_dataset(config.dataset_path)
    predictions = load_predictions(config.predictions_path)
    evaluations, summary = evaluate_dataset(examples, predictions)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config_hash = hashlib.sha256(
        json.dumps(asdict(config), default=str, sort_keys=True).encode("utf-8")
    ).hexdigest()
    run_metadata: dict[str, object] = {
        "name": config.name,
        "seed": config.seed,
        "created_at": datetime.now(UTC).isoformat(),
        "git_revision": _git_revision(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config_sha256": config_hash,
        "dataset": dataset_manifest(config.dataset_path, examples).as_dict(),
        "predictions_sha256": hashlib.sha256(config.predictions_path.read_bytes()).hexdigest(),
    }
    _write_json(config.output_dir / "run.json", run_metadata)
    _write_json(config.output_dir / "summary.json", summary.as_dict())
    with (config.output_dir / "per_example.jsonl").open("w", encoding="utf-8") as handle:
        for evaluation in evaluations:
            handle.write(
                json.dumps(evaluation.as_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            )
    return summary
