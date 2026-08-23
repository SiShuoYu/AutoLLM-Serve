"""Conservative, provenance-preserving cleanup of unreviewed candidate drafts."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .authoring import validate_authoring_submissions


@dataclass(frozen=True, slots=True)
class CandidateCurationPlan:
    input_sha256: str
    curated_input_sha256: str
    auto_rejected_task_count: int
    auto_rejection_counts: dict[str, int]
    replacement_requested_count: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _normalized_question(question: str) -> str:
    return re.sub(r"[\s\W_]+", "", question).lower()


def curate_train_candidate_drafts(
    queue_path: str | Path,
    submissions_path: str | Path,
    output_path: str | Path,
) -> CandidateCurationPlan:
    """Reject only objectively duplicate or generic train drafts in a new file.

    Short answers and weak lexical overlap remain review flags, because neither
    is sufficient proof that a Chinese answer is wrong. The source file is
    never modified, preserving every model draft for audit.
    """
    queue_file = Path(queue_path)
    submissions_file = Path(submissions_path)
    output = Path(output_path)
    if output.exists():
        raise ValueError(f"refusing to overwrite existing curated output: {output}")
    validate_authoring_submissions(queue_file, submissions_file)
    queue = {row["task_id"]: row for row in _rows(queue_file)}
    rows = _rows(submissions_file)
    eligible = [
        row
        for row in rows
        if row.get("review", {}).get("status") == "needs_revision"
        and queue[row["task_id"]].get("split") == "train"
        and queue[row["task_id"]].get("task_type") == "answerable"
    ]
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_question[_normalized_question(row["question"])].append(row)

    reasons: dict[str, set[str]] = defaultdict(set)
    for same_question_rows in by_question.values():
        if len(same_question_rows) < 2:
            continue
        keep = min(
            same_question_rows,
            key=lambda row: (
                queue[row["task_id"]].get("queue_role") != "primary",
                row["task_id"],
            ),
        )
        for row in same_question_rows:
            if row["task_id"] != keep["task_id"]:
                reasons[row["task_id"]].add("duplicate_question")
    generic_markers = ("无法回答", "不知道", "没有提供", "抱歉")
    for row in eligible:
        if any(marker in row["reference_answer"] for marker in generic_markers):
            reasons[row["task_id"]].add("generic_or_abstaining_answer")

    curated: list[dict[str, Any]] = []
    for row in rows:
        task_id = row["task_id"]
        if task_id not in reasons:
            curated.append(row)
            continue
        review = dict(row["review"])
        review["status"] = "rejected"
        review["notes"] = (
            "规则预审拒绝：" + "、".join(sorted(reasons[task_id])) + "。原始草稿保留。"
        )
        curated.append({**row, "review": review})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in curated:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    validate_authoring_submissions(queue_file, output)
    counts = Counter(reason for task_reasons in reasons.values() for reason in task_reasons)
    return CandidateCurationPlan(
        input_sha256=hashlib.sha256(submissions_file.read_bytes()).hexdigest(),
        curated_input_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        auto_rejected_task_count=len(reasons),
        auto_rejection_counts=dict(sorted(counts.items())),
        replacement_requested_count=len(reasons),
    )
