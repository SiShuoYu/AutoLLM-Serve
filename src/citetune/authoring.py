"""Deterministic source selection for human-reviewed QA authoring."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class AuthoringQueueManifest:
    source_split_sha256: dict[str, str]
    seed: int
    answerable_task_counts: dict[str, int]
    reserve_task_counts: dict[str, int]
    insufficient_evidence_task_counts: dict[str, int]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AuthoringValidationReport:
    """Outcome of validating reviewed QA rows against a locked task queue."""

    queue_task_count: int
    submitted_task_count: int
    approved_task_count: int
    pending_task_count: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on line {line_number} of {path}") from error
        if not isinstance(row, dict) or not isinstance(row.get("chunk_id"), str):
            raise ValueError(f"line {line_number} of {path} has no chunk_id")
        rows.append(row)
    return rows


def _stable_choice(rows: list[dict[str, Any]], seed: int, count: int) -> list[dict[str, Any]]:
    if count > len(rows):
        raise ValueError(f"requested {count} authoring tasks from only {len(rows)} source chunks")
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(f"{seed}:{row['chunk_id']}".encode()).hexdigest(),
    )[:count]


def build_authoring_queue(
    split_directory: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
    *,
    seed: int = 42,
    train_count: int = 1_200,
    validation_count: int = 100,
    test_answerable_count: int = 160,
    train_insufficient_evidence_count: int = 120,
    validation_insufficient_evidence_count: int = 20,
    test_insufficient_evidence_count: int = 40,
    train_reserve_count: int = 240,
    validation_reserve_count: int = 20,
    test_reserve_count: int = 32,
) -> AuthoringQueueManifest:
    """Select source chunks before authoring so future answers cannot leak splits.

    The resulting records intentionally contain no model-generated question or
    answer. They are review tasks, not training examples, until a human accepts
    the authored content under the annotation guide.
    """
    requested = {
        "train": train_count,
        "validation": validation_count,
        "test": test_answerable_count,
    }
    reserve_requested = {
        "train": train_reserve_count,
        "validation": validation_reserve_count,
        "test": test_reserve_count,
    }
    insufficient_requested = {
        "train": train_insufficient_evidence_count,
        "validation": validation_insufficient_evidence_count,
        "test": test_insufficient_evidence_count,
    }
    split_root = Path(split_directory)
    selected: dict[str, list[dict[str, Any]]] = {}
    checksums: dict[str, str] = {}
    for split, count in requested.items():
        path = split_root / f"{split}.jsonl"
        rows = _load_jsonl(path)
        selected[split] = _stable_choice(rows, seed, count + reserve_requested[split])
        checksums[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    output_rows: list[dict[str, object]] = []
    for split, rows in selected.items():
        primary_count = requested[split]
        for index, row in enumerate(rows):
            reserve = index >= primary_count
            ordinal = index - primary_count + 1 if reserve else index + 1
            task_suffix = (
                f"answerable-reserve-{ordinal:04d}" if reserve else f"answerable-{ordinal:04d}"
            )
            output_rows.append(
                {
                    "task_id": f"{split}-{task_suffix}",
                    "split": split,
                    "task_type": "answerable",
                    "queue_role": "reserve" if reserve else "primary",
                    "source_chunk": row,
                    "authoring_status": "pending",
                    "instructions": (
                        "Write one Chinese question answerable only from source_chunk.text, "
                        "a concise reference answer, and cite source_chunk.chunk_id."
                    ),
                }
            )
    for split, count in insufficient_requested.items():
        for ordinal in range(1, count + 1):
            output_rows.append(
                {
                    "task_id": f"{split}-insufficient-evidence-{ordinal:04d}",
                    "split": split,
                    "task_type": "insufficient_evidence",
                    "queue_role": "primary",
                    "source_chunk": None,
                    "authoring_status": "pending",
                    "instructions": (
                        "Write a realistic Chinese Kubernetes question whose answer is not in the "
                        "locked corpus. The expected behavior is an "
                        "evidence-insufficient abstention."
                    ),
                }
            )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = AuthoringQueueManifest(
        source_split_sha256=checksums,
        seed=seed,
        answerable_task_counts=requested,
        reserve_task_counts=reserve_requested,
        insufficient_evidence_task_counts=insufficient_requested,
    )
    manifest_file = Path(manifest_path)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(
        json.dumps(manifest.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _submission_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on line {line_number} of {path}") from error
        if not isinstance(row, dict):
            raise ValueError(f"line {line_number} of {path} must contain a JSON object")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no authoring rows")
    return rows


def _string_list(value: Any, field: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{field} must be a list of non-empty strings")
    if not allow_empty and not value:
        raise ValueError(f"{field} must not be empty")
    return value


def validate_authoring_submissions(
    queue_path: str | Path,
    submissions_path: str | Path,
    *,
    require_complete: bool = False,
) -> AuthoringValidationReport:
    """Validate reviewed content without treating model-written text as ground truth.

    Each submitted task must exist in the immutable queue. Only an approved
    answerable task with a reviewer confirmation can later become a training or
    evaluation example. This function deliberately performs no LLM judging.
    """
    queue_rows = _submission_rows(Path(queue_path))
    queue: dict[str, dict[str, Any]] = {}
    for queue_row in queue_rows:
        task_id = _required_text(queue_row.get("task_id"), "queue.task_id")
        if task_id in queue:
            raise ValueError(f"duplicate queue task ID: {task_id}")
        queue[task_id] = queue_row

    submissions = _submission_rows(Path(submissions_path))
    seen: set[str] = set()
    approved = 0
    for row in submissions:
        task_id = _required_text(row.get("task_id"), "submission.task_id")
        if task_id in seen:
            raise ValueError(f"duplicate submission task ID: {task_id}")
        seen.add(task_id)
        queue_task = queue.get(task_id)
        if queue_task is None:
            raise ValueError(f"submission references unknown task ID: {task_id}")
        if queue_task.get("split") not in {"train", "validation", "test"}:
            raise ValueError(f"queue task {task_id} has an invalid split")

        _required_text(row.get("author_id"), f"{task_id}.author_id")
        review = row.get("review")
        if not isinstance(review, dict):
            raise ValueError(f"{task_id}.review must be an object")
        status = _required_text(review.get("status"), f"{task_id}.review.status")
        if status not in {"approved", "needs_revision", "rejected"}:
            raise ValueError(f"{task_id}.review.status is invalid")
        if status == "rejected":
            _required_text(review.get("notes"), f"{task_id}.review.notes")
            continue
        _required_text(row.get("question"), f"{task_id}.question")
        _required_text(row.get("reference_answer"), f"{task_id}.reference_answer")

        citation_ids = _string_list(
            row.get("reference_citation_ids"),
            f"{task_id}.reference_citation_ids",
            allow_empty=queue_task.get("task_type") == "insufficient_evidence",
        )
        if queue_task.get("task_type") == "answerable":
            source = queue_task.get("source_chunk")
            if not isinstance(source, dict) or not isinstance(source.get("chunk_id"), str):
                raise ValueError(f"queue task {task_id} has no valid source chunk")
            for field in ("document_id", "text", "source_path", "source_url"):
                _required_text(source.get(field), f"queue task {task_id}.source_chunk.{field}")
            if citation_ids != [source["chunk_id"]]:
                raise ValueError(f"{task_id} must cite only its assigned source_chunk.chunk_id")
        elif queue_task.get("task_type") == "insufficient_evidence":
            if citation_ids:
                raise ValueError(f"{task_id} is an abstention task and must not invent citations")
            if "证据不足" not in row["reference_answer"]:
                raise ValueError(
                    f"{task_id} reference_answer must explicitly state evidence insufficiency"
                )
        else:
            raise ValueError(f"queue task {task_id} has an invalid task_type")
        if status == "approved":
            reviewer_id = _required_text(review.get("reviewer_id"), f"{task_id}.review.reviewer_id")
            reviewer_type = _required_text(
                review.get("reviewer_type"), f"{task_id}.review.reviewer_type"
            )
            if reviewer_type not in {"human", "model"}:
                raise ValueError(f"{task_id}.review.reviewer_type must be human or model")
            if reviewer_id == row["author_id"]:
                raise ValueError(f"{task_id} author and reviewer must be different")
            if queue_task["split"] in {"validation", "test"} and reviewer_type != "human":
                raise ValueError(f"{task_id} requires a human reviewer for held-out data")
            if review.get("evidence_supported") is not True:
                raise ValueError(
                    f"{task_id} must have review.evidence_supported=true before approval"
                )
            approved += 1

    if require_complete and seen != set(queue):
        missing = sorted(set(queue) - seen)
        raise ValueError(f"missing submissions for {len(missing)} queue tasks")
    return AuthoringValidationReport(
        queue_task_count=len(queue),
        submitted_task_count=len(submissions),
        approved_task_count=approved,
        pending_task_count=len(queue) - approved,
    )


def compile_reviewed_dataset(
    queue_path: str | Path,
    submissions_path: str | Path,
    output_path: str | Path,
    *,
    require_complete: bool = False,
) -> AuthoringValidationReport:
    """Compile only approved, review-validated rows into the evaluation schema."""
    report = validate_authoring_submissions(
        queue_path, submissions_path, require_complete=require_complete
    )
    queue = {row["task_id"]: row for row in _submission_rows(Path(queue_path))}
    compiled: list[dict[str, object]] = []
    for submission in _submission_rows(Path(submissions_path)):
        review = submission["review"]
        if review["status"] != "approved":
            continue
        task = queue[submission["task_id"]]
        if task["task_type"] == "answerable":
            source = task["source_chunk"]
            assert isinstance(source, dict)  # checked by validate_authoring_submissions
            compiled.append(
                {
                    "example_id": task["task_id"],
                    "split": task["split"],
                    "question": submission["question"],
                    "reference_answer": submission["reference_answer"],
                    "reference_citation_ids": submission["reference_citation_ids"],
                    "evidence": [
                        {
                            "evidence_id": source["chunk_id"],
                            "document_id": source["document_id"],
                            "text": source["text"],
                            "source_title": source["source_path"],
                            "source_url": source["source_url"],
                        }
                    ],
                    "expected_behavior": "answer",
                }
            )
        else:
            compiled.append(
                {
                    "example_id": task["task_id"],
                    "split": task["split"],
                    "question": submission["question"],
                    "reference_answer": submission["reference_answer"],
                    "reference_citation_ids": [],
                    "evidence": [],
                    "expected_behavior": "abstain_evidence_insufficient",
                }
            )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in compiled:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return report


def render_review_packet(
    queue_path: str | Path,
    submissions_path: str | Path,
    output_path: str | Path,
) -> AuthoringValidationReport:
    """Render candidate QA and assigned evidence into a human-readable packet."""
    report = validate_authoring_submissions(queue_path, submissions_path)
    queue = {row["task_id"]: row for row in _submission_rows(Path(queue_path))}
    sections = [
        "# CiteTune-CN QA 人工审核包",
        "",
        "> 这份文件用于人工核对，不是已批准的数据集。请同时检查问题、答案和原文证据。",
        "",
    ]
    for submission in _submission_rows(Path(submissions_path)):
        task = queue[submission["task_id"]]
        sections.extend(
            [
                f"## {task['task_id']}",
                "",
                f"- 数据切分：`{task['split']}`",
                f"- 队列角色：`{task.get('queue_role', 'primary')}`",
                f"- 当前状态：`{submission['review']['status']}`",
                "",
            ]
        )
        if submission["review"]["status"] == "rejected":
            sections.extend([f"**拒绝原因：** {submission['review']['notes']}", "", "---", ""])
            continue
        source = task["source_chunk"]
        assert isinstance(source, dict)
        evidence_lines = [line.rstrip() for line in source["text"].splitlines()]
        evidence = "\n".join(f"    {line}" if line else "" for line in evidence_lines)
        sections.extend(
            [
                f"**问题：** {submission['question']}",
                "",
                f"**参考答案：** {submission['reference_answer']}",
                "",
                f"**引用 ID：** `{source['chunk_id']}`",
                "",
                f"**来源：** [{source['source_path']}]({source['source_url']})",
                "",
                "**原文证据：**",
                "",
                evidence,
                "",
                "- [ ] 批准：问题自然，答案完全由证据支持",
                "- [ ] 需修改：在下面写明问题",
                "- [ ] 拒绝：来源本身不适合出题",
                "- 审核备注：",
                "- 审核人：",
                "",
                "---",
                "",
            ]
        )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(sections), encoding="utf-8")
    return report
