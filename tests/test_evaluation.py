from citetune.dataset import load_dataset, load_predictions
from citetune.evaluation import evaluate_dataset
from citetune.schemas import GroundedExample, ModelPrediction


def test_evaluation_reports_quality_and_human_labels() -> None:
    evaluations, summary = evaluate_dataset(
        load_dataset("data/samples/grounded_qa.jsonl"),
        load_predictions("data/samples/baseline_predictions.jsonl"),
    )
    assert len(evaluations) == 3
    assert summary.system_name == "sample-baseline"
    assert summary.example_count == 3
    assert summary.citation_valid_rate == 1.0
    assert summary.human_annotation_count == 3
    assert summary.fully_correct_rate == 2 / 3
    assert summary.evidence_supported_rate == 2 / 3


def test_evaluation_reports_abstention_accuracy() -> None:
    example = GroundedExample.from_dict(
        {
            "example_id": "abstain-1",
            "split": "test",
            "question": "问题",
            "reference_answer": "提供的证据不足以回答此问题。",
            "reference_citation_ids": [],
            "evidence": [],
            "expected_behavior": "abstain_evidence_insufficient",
        }
    )
    prediction = ModelPrediction.from_dict(
        {
            "example_id": "abstain-1",
            "system_name": "baseline",
            "answer": "当前证据不足，无法回答。",
            "citation_ids": [],
        }
    )
    evaluations, summary = evaluate_dataset([example], [prediction])
    assert evaluations[0].abstention_correct == 1.0
    assert summary.abstention_accuracy == 1.0
