"""Data readiness gate before any GPU training or inference work."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class DataReadinessReport:
    queue_task_count: int
    submitted_task_count: int
    status_counts: dict[str, int]
    required_primary_counts: dict[str, int]
    approved_counts: dict[str, int]
    missing_approved_counts: dict[str, int]
    human_approved_test_count: int
    ready_for_gpu: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row: Any = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"line {line_number} of {path} must be an object")
        rows.append(row)
    return rows


def _gate_key(split: object, task_type: object) -> str:
    return f"{split}:{task_type}"


def assess_data_readiness(
    queue_path: str | Path, submissions_path: str | Path
) -> DataReadinessReport:
    queue_rows = _read_jsonl(queue_path)
    submissions = _read_jsonl(submissions_path)
    queue = {row.get("task_id"): row for row in queue_rows}
    required = Counter(
        _gate_key(row.get("split"), row.get("task_type"))
        for row in queue_rows
        if row.get("queue_role", "primary") == "primary"
    )
    status_counts: Counter[str] = Counter()
    approved: Counter[str] = Counter()
    human_test = 0
    for row in submissions:
        task = queue.get(row.get("task_id"))
        if task is None:
            raise ValueError(f"unknown readiness task: {row.get('task_id')}")
        review = row.get("review")
        if not isinstance(review, dict) or not isinstance(review.get("status"), str):
            raise ValueError(f"submission {row.get('task_id')} has no review status")
        status = review["status"]
        status_counts[status] += 1
        if status != "approved" or task.get("queue_role", "primary") != "primary":
            continue
        key = _gate_key(task.get("split"), task.get("task_type"))
        approved[key] += 1
        if task.get("split") == "test" and review.get("reviewer_type") == "human":
            human_test += 1
    missing = {key: max(0, count - approved[key]) for key, count in sorted(required.items())}
    return DataReadinessReport(
        queue_task_count=len(queue_rows),
        submitted_task_count=len(submissions),
        status_counts=dict(sorted(status_counts.items())),
        required_primary_counts=dict(sorted(required.items())),
        approved_counts={key: approved[key] for key in sorted(required)},
        missing_approved_counts=missing,
        human_approved_test_count=human_test,
        ready_for_gpu=not any(missing.values()) and human_test >= 200,
    )
