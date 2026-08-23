"""Deterministic risk screening for unreviewed QA candidate drafts.

The report deliberately identifies review priorities only. It never changes a
candidate's review status and cannot approve a model-generated answer.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .authoring import validate_authoring_submissions


@dataclass(frozen=True, slots=True)
class CandidateAuditReport:
    queue_sha256: str
    submissions_sha256: str
    screened_candidate_count: int
    status_counts: dict[str, int]
    split_counts: dict[str, int]
    issue_counts: dict[str, int]
    flagged_task_count: int
    flagged_task_ids_sha256: str
    review_packet_task_count: int
    seed: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _ScreenedCandidate:
    row: dict[str, Any]
    task: dict[str, Any]
    issues: list[str]
    overlap: float


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _content_characters(text: str) -> set[str]:
    return set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", text.lower()))


def _risk_issues(
    question: str,
    answer: str,
    evidence: str,
    *,
    duplicate_question: bool,
) -> tuple[list[str], float]:
    issues: list[str] = []
    if len(question.strip()) < 8:
        issues.append("question_too_short")
    if len(answer.strip()) < 12:
        issues.append("answer_too_short")
    if duplicate_question:
        issues.append("duplicate_question")
    answer_characters = _content_characters(answer)
    evidence_characters = _content_characters(evidence)
    overlap = len(answer_characters & evidence_characters) / max(1, len(answer_characters))
    if overlap < 0.15:
        issues.append("low_answer_evidence_character_overlap")
    if any(marker in answer for marker in ("无法回答", "不知道", "没有提供", "抱歉")):
        issues.append("generic_or_abstaining_answer")
    return issues, overlap


def _review_section(
    row: dict[str, Any], task: dict[str, Any], issues: list[str], overlap: float
) -> str:
    source = task["source_chunk"]
    risk_text = "、".join(issues) if issues else "无规则风险标记（仍需人工核对）"
    return "\n".join(
        [
            f"## {row['task_id']}",
            "",
            f"- 数据切分：`{task['split']}`",
            f"- 队列角色：`{task.get('queue_role', 'primary')}`",
            f"- 规则风险：{risk_text}",
            f"- 答案-证据字符重合：`{overlap:.2f}`（仅作筛查信号，不代表事实正确）",
            "",
            f"**问题：** {row['question']}",
            "",
            f"**参考答案：** {row['reference_answer']}",
            "",
            f"**引用 ID：** `{source['chunk_id']}`",
            "",
            "**原文证据：**",
            "```markdown",
            source["text"],
            "```",
            "",
            "- [ ] 批准：问题自然，答案完全由证据支持",
            "- [ ] 需修改：在下面写明问题",
            "- [ ] 拒绝：来源或草稿不适合出题",
            "- 审核备注：",
            "- 审核人：",
            "",
            "---",
            "",
        ]
    )


def audit_candidate_drafts(
    queue_path: str | Path,
    submissions_path: str | Path,
    report_path: str | Path,
    review_packet_path: str | Path,
    *,
    review_limit: int = 60,
    seed: int = 42,
) -> CandidateAuditReport:
    """Screen candidate drafts and render a deterministic, bounded human batch."""
    if review_limit <= 0:
        raise ValueError("review_limit must be positive")
    queue_file = Path(queue_path)
    submissions_file = Path(submissions_path)
    validate_authoring_submissions(queue_file, submissions_file)
    queue = {row["task_id"]: row for row in _rows(queue_file)}
    submissions = _rows(submissions_file)
    status_counts = Counter(
        str(row.get("review", {}).get("status", "missing")) for row in submissions
    )
    candidates = [
        row
        for row in submissions
        if row.get("review", {}).get("status") == "needs_revision"
        and queue[row["task_id"]].get("task_type") == "answerable"
    ]
    question_counts = Counter(" ".join(row["question"].split()).lower() for row in candidates)
    screened: list[_ScreenedCandidate] = []
    issue_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    for row in candidates:
        task = queue[row["task_id"]]
        source = task["source_chunk"]
        assert isinstance(source, dict)
        issues, overlap = _risk_issues(
            row["question"],
            row["reference_answer"],
            source["text"],
            duplicate_question=question_counts[" ".join(row["question"].split()).lower()] > 1,
        )
        issue_counts.update(issues)
        split_counts.update([task["split"]])
        screened.append(_ScreenedCandidate(row=row, task=task, issues=issues, overlap=overlap))
    flagged = [item for item in screened if item.issues]
    ordered = sorted(
        screened,
        key=lambda item: (
            -len(item.issues),
            hashlib.sha256(f"{seed}:{item.row['task_id']}".encode()).hexdigest(),
        ),
    )
    selected = ordered[:review_limit]
    packet = Path(review_packet_path)
    packet.parent.mkdir(parents=True, exist_ok=True)
    packet.write_text(
        "# CiteTune-CN 候选 QA 审核批次\n\n"
        "> 本批次由确定性风险筛查排序，不含自动批准。字符重合仅用于定位风险，不能证明答案正确。\n\n"
        + "".join(
            _review_section(item.row, item.task, item.issues, item.overlap) for item in selected
        ),
        encoding="utf-8",
    )
    flagged_ids = "\n".join(sorted(item.row["task_id"] for item in flagged)).encode()
    report = CandidateAuditReport(
        queue_sha256=hashlib.sha256(queue_file.read_bytes()).hexdigest(),
        submissions_sha256=hashlib.sha256(submissions_file.read_bytes()).hexdigest(),
        screened_candidate_count=len(screened),
        status_counts=dict(sorted(status_counts.items())),
        split_counts=dict(sorted(split_counts.items())),
        issue_counts=dict(sorted(issue_counts.items())),
        flagged_task_count=len(flagged),
        flagged_task_ids_sha256=hashlib.sha256(flagged_ids).hexdigest(),
        review_packet_task_count=len(selected),
        seed=seed,
    )
    output = Path(report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
