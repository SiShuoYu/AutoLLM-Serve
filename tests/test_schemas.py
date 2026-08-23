import pytest

from citetune.schemas import GroundedExample, ModelPrediction


def test_rejects_reference_citation_missing_from_evidence() -> None:
    raw = {
        "example_id": "bad-1",
        "split": "test",
        "question": "问题",
        "reference_answer": "答案",
        "reference_citation_ids": ["unknown"],
        "evidence": [
            {
                "evidence_id": "known",
                "document_id": "doc-1",
                "text": "证据",
                "source_title": "来源",
            }
        ],
    }
    with pytest.raises(ValueError, match="unknown citation"):
        GroundedExample.from_dict(raw)


def test_abstention_examples_and_predictions_allow_no_citations() -> None:
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
            "system_name": "test",
            "answer": "证据不足，无法回答。",
            "citation_ids": [],
        }
    )
    assert example.expected_behavior == "abstain_evidence_insufficient"
    assert prediction.citation_ids == ()
