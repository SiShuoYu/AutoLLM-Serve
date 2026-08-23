"""Per-example and aggregate evaluation for grounded answers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean

from .metrics import character_f1, exact_match, set_precision, set_recall
from .schemas import GroundedExample, ModelPrediction


@dataclass(frozen=True, slots=True)
class ExampleEvaluation:
    example_id: str
    system_name: str
    exact_match: float
    character_f1: float
    citation_precision: float | None
    citation_recall: float
    invalid_citation_ids: tuple[str, ...]
    abstention_correct: float | None
    latency_ms: float | None
    factuality: int | None
    unsupported_claims: int | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    system_name: str
    example_count: int
    exact_match: float
    character_f1: float
    citation_precision: float | None
    citation_recall: float
    citation_valid_rate: float
    mean_latency_ms: float | None
    human_annotation_count: int
    fully_correct_rate: float | None
    evidence_supported_rate: float | None
    abstention_accuracy: float | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_example(example: GroundedExample, prediction: ModelPrediction) -> ExampleEvaluation:
    if example.example_id != prediction.example_id:
        raise ValueError("example and prediction IDs must match")
    known_citations = {evidence.evidence_id for evidence in example.evidence}
    expected_citations = set(example.reference_citation_ids)
    observed_citations = set(prediction.citation_ids)
    annotation = prediction.annotation
    return ExampleEvaluation(
        example_id=example.example_id,
        system_name=prediction.system_name,
        exact_match=exact_match(example.reference_answer, prediction.answer),
        character_f1=character_f1(example.reference_answer, prediction.answer),
        citation_precision=set_precision(expected_citations, observed_citations),
        citation_recall=set_recall(expected_citations, observed_citations),
        invalid_citation_ids=tuple(sorted(observed_citations - known_citations)),
        abstention_correct=(
            float("证据不足" in prediction.answer)
            if example.expected_behavior == "abstain_evidence_insufficient"
            else None
        ),
        latency_ms=prediction.latency_ms,
        factuality=annotation.factuality if annotation else None,
        unsupported_claims=annotation.unsupported_claims if annotation else None,
    )


def evaluate_dataset(
    examples: list[GroundedExample], predictions: list[ModelPrediction]
) -> tuple[list[ExampleEvaluation], EvaluationSummary]:
    example_by_id = {example.example_id: example for example in examples}
    prediction_by_id = {prediction.example_id: prediction for prediction in predictions}
    missing = sorted(set(example_by_id) - set(prediction_by_id))
    unexpected = sorted(set(prediction_by_id) - set(example_by_id))
    if missing or unexpected:
        raise ValueError(
            f"prediction IDs do not match dataset; missing={missing}, unexpected={unexpected}"
        )
    systems = {prediction.system_name for prediction in predictions}
    if len(systems) != 1:
        raise ValueError("one prediction file must contain exactly one system_name")
    evaluations = [
        evaluate_example(example, prediction_by_id[example.example_id]) for example in examples
    ]
    annotations = [evaluation for evaluation in evaluations if evaluation.factuality is not None]
    precision_values = [
        evaluation.citation_precision
        for evaluation in evaluations
        if evaluation.citation_precision is not None
    ]
    latencies = [
        evaluation.latency_ms for evaluation in evaluations if evaluation.latency_ms is not None
    ]
    abstentions = [
        evaluation.abstention_correct
        for evaluation in evaluations
        if evaluation.abstention_correct is not None
    ]
    summary = EvaluationSummary(
        system_name=next(iter(systems)),
        example_count=len(evaluations),
        exact_match=mean(evaluation.exact_match for evaluation in evaluations),
        character_f1=mean(evaluation.character_f1 for evaluation in evaluations),
        citation_precision=mean(precision_values) if precision_values else None,
        citation_recall=mean(evaluation.citation_recall for evaluation in evaluations),
        citation_valid_rate=mean(
            float(not evaluation.invalid_citation_ids) for evaluation in evaluations
        ),
        mean_latency_ms=mean(latencies) if latencies else None,
        human_annotation_count=len(annotations),
        fully_correct_rate=(
            mean(float(evaluation.factuality == 2) for evaluation in annotations)
            if annotations
            else None
        ),
        evidence_supported_rate=(
            mean(float(evaluation.unsupported_claims == 0) for evaluation in annotations)
            if annotations
            else None
        ),
        abstention_accuracy=mean(abstentions) if abstentions else None,
    )
    return evaluations, summary
