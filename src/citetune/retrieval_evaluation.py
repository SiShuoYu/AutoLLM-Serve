"""Auditable retrieval evaluation over reviewed grounded QA."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

from .dataset import load_dataset
from .retrieval import BM25Retriever
from .schemas import GroundedExample


@dataclass(frozen=True, slots=True)
class RetrievalExampleEvaluation:
    example_id: str
    relevant_rank: int | None
    retrieved_chunk_ids: tuple[str, ...]
    recall_at_k: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationSummary:
    retriever: str
    evaluated_answerable_count: int
    excluded_abstention_count: int
    recall_at_k: dict[str, float]
    mean_reciprocal_rank: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_retrieval(
    retriever: BM25Retriever,
    examples: list[GroundedExample],
    *,
    k_values: tuple[int, ...] = (1, 5, 10),
) -> tuple[list[RetrievalExampleEvaluation], RetrievalEvaluationSummary]:
    normalized_k = tuple(sorted(set(k_values)))
    if not normalized_k or any(value <= 0 for value in normalized_k):
        raise ValueError("k_values must contain positive integers")
    answerable = [example for example in examples if example.expected_behavior == "answer"]
    if not answerable:
        raise ValueError("retrieval evaluation needs at least one answerable example")
    evaluations: list[RetrievalExampleEvaluation] = []
    for example in answerable:
        expected = set(example.reference_citation_ids)
        results = retriever.search(example.question, top_k=max(normalized_k))
        retrieved = tuple(result.chunk_id for result in results)
        relevant_rank = next(
            (rank for rank, chunk_id in enumerate(retrieved, start=1) if chunk_id in expected),
            None,
        )
        evaluations.append(
            RetrievalExampleEvaluation(
                example_id=example.example_id,
                relevant_rank=relevant_rank,
                retrieved_chunk_ids=retrieved,
                recall_at_k={
                    str(k): len(expected & set(retrieved[:k])) / len(expected) for k in normalized_k
                },
            )
        )
    summary = RetrievalEvaluationSummary(
        retriever="bm25",
        evaluated_answerable_count=len(evaluations),
        excluded_abstention_count=len(examples) - len(answerable),
        recall_at_k={
            str(k): mean(evaluation.recall_at_k[str(k)] for evaluation in evaluations)
            for k in normalized_k
        },
        mean_reciprocal_rank=mean(
            1 / evaluation.relevant_rank if evaluation.relevant_rank is not None else 0.0
            for evaluation in evaluations
        ),
    )
    return evaluations, summary


def run_retrieval_evaluation(
    corpus_path: str | Path,
    dataset_path: str | Path,
    output_directory: str | Path,
    *,
    k_values: tuple[int, ...] = (1, 5, 10),
) -> RetrievalEvaluationSummary:
    corpus = Path(corpus_path)
    dataset = Path(dataset_path)
    output = Path(output_directory)
    evaluations, summary = evaluate_retrieval(
        BM25Retriever.from_jsonl(corpus), load_dataset(dataset), k_values=k_values
    )
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "retriever": "bm25",
        "created_at": datetime.now(UTC).isoformat(),
        "k_values": list(sorted(set(k_values))),
        "corpus_path": str(corpus),
        "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
        "dataset_path": str(dataset),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
    }
    (output / "run.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "summary.json").write_text(
        json.dumps(summary.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output / "per_example.jsonl").open("w", encoding="utf-8") as handle:
        for evaluation in evaluations:
            handle.write(
                json.dumps(evaluation.as_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            )
    return summary
