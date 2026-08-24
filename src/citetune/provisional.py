"""Explicitly non-reportable held-out data exports for inference plumbing checks."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .authoring import validate_authoring_submissions
from .schemas import GroundedExample


@dataclass(frozen=True, slots=True)
class ProvisionalHeldoutManifest:
    """Provenance for a latency-only, unreviewed held-out draft export."""

    dataset_role: str
    queue_sha256: str
    submissions_sha256: str
    output_sha256: str
    format_version: str
    exported_split_counts: dict[str, int]
    excluded_submission_status_counts: dict[str, int]
    limit_per_split: int | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def export_provisional_heldout_dataset(
    queue_path: str | Path,
    submissions_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
    *,
    limit_per_split: int | None = None,
) -> ProvisionalHeldoutManifest:
    """Export unreviewed validation/test drafts only for operational inference checks.

    The output is schema-compatible with prediction generation so that real
    latency and memory can be measured. Its explicit role marker prevents it
    from being mistaken for a quality benchmark or an SFT source.
    """
    if limit_per_split is not None and limit_per_split <= 0:
        raise ValueError("limit_per_split must be positive when provided")
    queue_file = Path(queue_path)
    submissions_file = Path(submissions_path)
    validate_authoring_submissions(queue_file, submissions_file)
    queue = {row["task_id"]: row for row in _rows(queue_file)}
    submissions = _rows(submissions_file)
    selected = [
        row
        for row in submissions
        if row.get("review", {}).get("status") == "needs_revision"
        and queue[row["task_id"]].get("split") in {"validation", "test"}
        and queue[row["task_id"]].get("task_type") == "answerable"
    ]
    selected = sorted(selected, key=lambda row: row["task_id"])
    if limit_per_split is not None:
        limited: list[dict[str, Any]] = []
        for split in ("validation", "test"):
            split_rows = [row for row in selected if queue[row["task_id"]]["split"] == split]
            limited.extend(split_rows[:limit_per_split])
        selected = limited
    split_counts = Counter(str(queue[row["task_id"]]["split"]) for row in selected)
    if not split_counts.get("validation") or not split_counts.get("test"):
        raise ValueError("provisional export requires needs_revision rows in validation and test")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for candidate in selected:
            task = queue[candidate["task_id"]]
            source = task["source_chunk"]
            assert isinstance(source, dict)
            example = GroundedExample.from_dict(
                {
                    "example_id": candidate["task_id"],
                    "split": task["split"],
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
            row = example.as_dict()
            row["dataset_role"] = "unreviewed_heldout_drafts_latency_only_not_quality_benchmark"
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    excluded = Counter(
        str(row.get("review", {}).get("status", "missing"))
        for row in submissions
        if row not in selected
    )
    manifest = ProvisionalHeldoutManifest(
        dataset_role="unreviewed_heldout_drafts_latency_only_not_quality_benchmark",
        queue_sha256=hashlib.sha256(queue_file.read_bytes()).hexdigest(),
        submissions_sha256=hashlib.sha256(submissions_file.read_bytes()).hexdigest(),
        output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        format_version="citetune-provisional-heldout-v1",
        exported_split_counts=dict(sorted(split_counts.items())),
        excluded_submission_status_counts=dict(sorted(excluded.items())),
        limit_per_split=limit_per_split,
    )
    manifest_file = Path(manifest_path)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(
        json.dumps(manifest.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
