"""Versioned data contracts for grounded-answer experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _text_list(value: Any, field: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{field} must be a list of non-empty strings")
    if not allow_empty and not value:
        raise ValueError(f"{field} must not be empty")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class Evidence:
    evidence_id: str
    document_id: str
    text: str
    source_title: str
    source_url: str | None = None

    @classmethod
    def from_dict(cls, raw: Any) -> Evidence:
        if not isinstance(raw, dict):
            raise ValueError("evidence must be an object")
        source_url = raw.get("source_url")
        if source_url is not None and not isinstance(source_url, str):
            raise ValueError("evidence.source_url must be a string or null")
        return cls(
            evidence_id=_required_text(raw.get("evidence_id"), "evidence.evidence_id"),
            document_id=_required_text(raw.get("document_id"), "evidence.document_id"),
            text=_required_text(raw.get("text"), "evidence.text"),
            source_title=_required_text(raw.get("source_title"), "evidence.source_title"),
            source_url=source_url,
        )


@dataclass(frozen=True, slots=True)
class GroundedExample:
    """One question with auditable evidence and a reference answer."""

    example_id: str
    split: str
    question: str
    reference_answer: str
    reference_citation_ids: tuple[str, ...]
    evidence: tuple[Evidence, ...]
    expected_behavior: str = "answer"

    @classmethod
    def from_dict(cls, raw: Any) -> GroundedExample:
        if not isinstance(raw, dict):
            raise ValueError("dataset row must be an object")
        expected_behavior = raw.get("expected_behavior", "answer")
        if expected_behavior not in {"answer", "abstain_evidence_insufficient"}:
            raise ValueError("expected_behavior must be answer or abstain_evidence_insufficient")
        evidence_raw = raw.get("evidence")
        if not isinstance(evidence_raw, list):
            raise ValueError("evidence must be a list")
        if expected_behavior == "answer" and not evidence_raw:
            raise ValueError("answer examples must include evidence")
        example = cls(
            example_id=_required_text(raw.get("example_id"), "example_id"),
            split=_required_text(raw.get("split"), "split"),
            question=_required_text(raw.get("question"), "question"),
            reference_answer=_required_text(raw.get("reference_answer"), "reference_answer"),
            reference_citation_ids=_text_list(
                raw.get("reference_citation_ids"),
                "reference_citation_ids",
                allow_empty=expected_behavior == "abstain_evidence_insufficient",
            ),
            evidence=tuple(Evidence.from_dict(item) for item in evidence_raw),
            expected_behavior=expected_behavior,
        )
        if example.split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        evidence_ids = [item.evidence_id for item in example.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError(f"duplicate evidence_id in {example.example_id}")
        unknown = set(example.reference_citation_ids) - set(evidence_ids)
        if unknown:
            raise ValueError(f"reference has unknown citation IDs: {sorted(unknown)}")
        if example.expected_behavior == "abstain_evidence_insufficient":
            if example.evidence or example.reference_citation_ids:
                raise ValueError("abstention examples must not provide evidence or citations")
            if "证据不足" not in example.reference_answer:
                raise ValueError(
                    "abstention reference_answer must explicitly state evidence insufficiency"
                )
        return example

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class HumanAnnotation:
    """Optional human labels; scores are never inferred by an LLM in V1."""

    factuality: int
    unsupported_claims: int
    notes: str = ""

    @classmethod
    def from_dict(cls, raw: Any) -> HumanAnnotation:
        if not isinstance(raw, dict):
            raise ValueError("annotation must be an object")
        factuality = raw.get("factuality")
        unsupported_claims = raw.get("unsupported_claims")
        notes = raw.get("notes", "")
        if factuality not in {0, 1, 2}:
            raise ValueError("annotation.factuality must be 0, 1, or 2")
        if not isinstance(unsupported_claims, int) or unsupported_claims < 0:
            raise ValueError("annotation.unsupported_claims must be a non-negative integer")
        if not isinstance(notes, str):
            raise ValueError("annotation.notes must be a string")
        return cls(factuality=factuality, unsupported_claims=unsupported_claims, notes=notes)


@dataclass(frozen=True, slots=True)
class ModelPrediction:
    example_id: str
    system_name: str
    answer: str
    citation_ids: tuple[str, ...]
    latency_ms: float | None = None
    annotation: HumanAnnotation | None = None

    @classmethod
    def from_dict(cls, raw: Any) -> ModelPrediction:
        if not isinstance(raw, dict):
            raise ValueError("prediction row must be an object")
        latency_ms = raw.get("latency_ms")
        if latency_ms is not None and (
            not isinstance(latency_ms, (int, float))
            or isinstance(latency_ms, bool)
            or latency_ms < 0
        ):
            raise ValueError("latency_ms must be a non-negative number or null")
        annotation_raw = raw.get("annotation")
        return cls(
            example_id=_required_text(raw.get("example_id"), "prediction.example_id"),
            system_name=_required_text(raw.get("system_name"), "prediction.system_name"),
            answer=_required_text(raw.get("answer"), "prediction.answer"),
            citation_ids=_text_list(
                raw.get("citation_ids"), "prediction.citation_ids", allow_empty=True
            ),
            latency_ms=float(latency_ms) if latency_ms is not None else None,
            annotation=(
                HumanAnnotation.from_dict(annotation_raw) if annotation_raw is not None else None
            ),
        )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)
