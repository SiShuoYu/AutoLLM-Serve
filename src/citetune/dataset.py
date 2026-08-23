"""JSONL data loading, validation, and deterministic manifests."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from .schemas import GroundedExample, ModelPrediction


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on line {line_number} of {path}") from error
        if not isinstance(raw, dict):
            raise ValueError(f"line {line_number} of {path} must contain a JSON object")
        rows.append(raw)
    if not rows:
        raise ValueError(f"{path} contains no JSONL rows")
    return rows


def load_dataset(path: str | Path) -> list[GroundedExample]:
    dataset_path = Path(path)
    examples = [GroundedExample.from_dict(row) for row in _read_jsonl(dataset_path)]
    identifiers = [example.example_id for example in examples]
    duplicates = [item for item, count in Counter(identifiers).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate example IDs: {sorted(duplicates)}")
    return examples


def load_predictions(path: str | Path) -> list[ModelPrediction]:
    prediction_path = Path(path)
    predictions = [ModelPrediction.from_dict(row) for row in _read_jsonl(prediction_path)]
    identifiers = [prediction.example_id for prediction in predictions]
    duplicates = [item for item, count in Counter(identifiers).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate prediction example IDs: {sorted(duplicates)}")
    return predictions


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    path: str
    sha256: str
    example_count: int
    split_counts: dict[str, int]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def dataset_manifest(
    path: str | Path, examples: list[GroundedExample] | None = None
) -> DatasetManifest:
    dataset_path = Path(path)
    examples = examples if examples is not None else load_dataset(dataset_path)
    return DatasetManifest(
        path=str(dataset_path),
        sha256=hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        example_count=len(examples),
        split_counts=dict(sorted(Counter(example.split for example in examples).items())),
    )
