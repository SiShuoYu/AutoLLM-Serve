# Architecture

```text
Versioned JSONL dataset ──► schema validation ──► dataset manifest (SHA-256)
          │
          └────────────────────────► model predictions + optional human labels
                                             │
                                             ▼
                                  deterministic evaluator
                                             │
                         ┌───────────────────┼───────────────────┐
                         ▼                   ▼                   ▼
                   per-example JSONL    summary JSON         run metadata
```

V1 measures lexical answer agreement, citation-ID agreement, known-invalid
citations, and optional human factuality/evidence-support labels. It does not
claim that lexical similarity proves factuality, and it does not ask an LLM to
grade itself. Later versions will add an independently validated judge model and
human-adjudicated benchmark subsets.

## Retrieval baseline

The first retrieval baseline is a deterministic, dependency-free BM25 index
over the locked JSONL corpus. Its purpose is transparency: every returned
result includes its `chunk_id`, score, source text, and immutable source URL.
It is not presented as a semantic-retrieval result, and it is evaluated only
after reviewed QA data exists.

```text
locked corpus JSONL ──► BM25 ──► ranked chunk IDs + source text
                                        │
                                        └──► future RAG prompt / retrieval recall evaluation
```

Once reviewed QA exists, `citetune evaluate-retrieval` records Recall@1/5/10,
MRR, per-question ranks, and SHA-256 hashes for both inputs. Evidence-insufficient
questions are excluded from retrieval recall and counted separately.
