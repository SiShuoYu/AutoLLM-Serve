import json
from pathlib import Path

from citetune.retrieval import BM25Retriever, tokenize
from citetune.retrieval_evaluation import evaluate_retrieval, run_retrieval_evaluation
from citetune.schemas import GroundedExample


def test_tokenizer_keeps_chinese_characters_and_technical_terms() -> None:
    assert tokenize("Pod 在 Kubernetes 中运行") == ["pod", "在", "kubernetes", "中", "运", "行"]


def test_bm25_returns_traceable_ranked_source_chunks(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in (
                {
                    "chunk_id": "pods",
                    "text": "Pod 是 Kubernetes 中最小的可部署单元。",
                    "source_url": "https://example.test/pods",
                },
                {
                    "chunk_id": "services",
                    "text": "Service 为一组 Pod 提供稳定的网络访问。",
                    "source_url": "https://example.test/services",
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    results = BM25Retriever.from_jsonl(corpus).search("Pod 最小部署单元", top_k=1)
    assert results[0].chunk_id == "pods"
    assert results[0].source_url == "https://example.test/pods"


def test_retrieval_evaluation_reports_recall_mrr_and_excludes_abstentions(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in (
                {
                    "chunk_id": "pods",
                    "text": "Pod 是 Kubernetes 中最小的可部署单元。",
                    "source_url": "https://example.test/pods",
                },
                {
                    "chunk_id": "services",
                    "text": "Service 提供稳定的网络访问。",
                    "source_url": "https://example.test/services",
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    answerable_raw = {
        "example_id": "qa-1",
        "split": "test",
        "question": "Kubernetes 最小的可部署单元是什么？",
        "reference_answer": "Pod。",
        "reference_citation_ids": ["pods"],
        "evidence": [
            {
                "evidence_id": "pods",
                "document_id": "doc-pods",
                "text": "Pod 是 Kubernetes 中最小的可部署单元。",
                "source_title": "pods",
            }
        ],
    }
    abstention_raw = {
        "example_id": "qa-2",
        "split": "test",
        "question": "没有证据的问题",
        "reference_answer": "提供的证据不足以回答此问题。",
        "reference_citation_ids": [],
        "evidence": [],
        "expected_behavior": "abstain_evidence_insufficient",
    }
    examples = [
        GroundedExample.from_dict(answerable_raw),
        GroundedExample.from_dict(abstention_raw),
    ]
    evaluations, summary = evaluate_retrieval(
        BM25Retriever.from_jsonl(corpus), examples, k_values=(1, 2)
    )
    assert evaluations[0].relevant_rank == 1
    assert summary.recall_at_k == {"1": 1.0, "2": 1.0}
    assert summary.mean_reciprocal_rank == 1.0
    assert summary.excluded_abstention_count == 1

    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in (answerable_raw, abstention_raw))
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "retrieval-run"
    persisted = run_retrieval_evaluation(corpus, dataset, output, k_values=(1, 2))
    assert persisted.recall_at_k["1"] == 1.0
    assert (output / "run.json").is_file()
    assert (output / "summary.json").is_file()
    assert (output / "per_example.jsonl").is_file()
