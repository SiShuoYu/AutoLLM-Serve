# Question Authoring and Review Guide

## Answerable task

Question tasks are selected only from a corpus-quality-filtered source file.
The filter removes very short chunks, chunks with little Chinese explanatory
text, and chunks dominated by Hugo/HTML template markers. This makes source
selection reproducible; it does not replace the review below.

For each assigned source chunk:

1. Write one natural Chinese Kubernetes question whose complete answer appears
   in the chunk. Do not require external knowledge.
2. Write a short reference answer using only the chunk's information.
3. Cite the supplied `chunk_id` and do not add another source.
4. Reject questions that merely copy a heading, contain an ambiguous pronoun, or
   can be answered correctly without reading the evidence.

## Insufficient-evidence test task

Write a realistic Kubernetes question that cannot be answered from the locked
Chinese-document corpus. Its reference behavior is an explicit abstention such
as “提供的证据不足以回答此问题”, not a guessed answer. Do not use sensitive,
operational, or security-exploitation prompts.

## Review labels

- `factuality`: 0 = incorrect, 1 = partly correct, 2 = fully correct.
- `unsupported_claims`: count distinct factual claims not justified by evidence.
- A valid citation ID is necessary but not sufficient: review whether the cited
  text actually supports the answer.

Reject the source task itself when its chunk starts mid-sentence, depends on an
undefined pronoun, or omits a name needed to understand the question. A
rejected task needs `review.status: rejected` and a non-empty `review.notes`;
it must not contain an invented placeholder question or answer.

The queue contains a deterministic reserve pool (20% of the answerable target
for each split). Replace rejected primary tasks only with reserve tasks from the
same split. Never move a validation or test reserve into training.

For a non-technical reviewer, render candidates and their assigned evidence as
a readable Markdown packet:

```bash
citetune render-review-packet --queue data/processed/kubernetes_zh_authoring_queue.jsonl --submissions data/pilot/qa_candidates.jsonl --output data/pilot/qa_review_packet.md
```

Checking boxes in the packet documents the decision, but the corresponding
JSONL review fields must still be updated before compilation.

## Submission format and quality gate

Submit one JSON object per line. Do not change `task_id`; it ties the item to
the split chosen before any question was written.

```json
{
  "task_id": "train-answerable-0001",
  "question": "...",
  "reference_answer": "...",
  "reference_citation_ids": ["assigned-chunk-id"],
  "author_id": "annotator-01",
  "review": {
    "status": "approved",
    "reviewer_id": "reviewer-01",
    "evidence_supported": true,
    "notes": ""
  }
}
```

Before an item can enter a dataset, run:

```bash
citetune validate-authoring --queue data/processed/kubernetes_zh_authoring_queue.jsonl --submissions path/to/reviewed.jsonl
```

The command rejects unknown or duplicate tasks, unreviewed approvals, and
citations that do not exactly match the queue's assigned source chunk. A
`needs_revision` or `rejected` row remains visible in the audit trail but is
not counted as approved. Drafts marked `needs_revision` must already have a
valid assigned citation; only the independent reviewer decision remains open.
For an `insufficient_evidence` task, provide an empty
reference citation list and make the reference answer explicitly say “证据不足”.

An approval must record `reviewer_type` as `human` or `model`, and the reviewer
ID must differ from `author_id`. Validation and test approvals require
`reviewer_type: human`; model pre-review can assist but cannot unlock held-out
data or satisfy the project charter's human-review claim.

After validation, compile only approved rows with:

```bash
citetune compile-reviewed-dataset --queue data/processed/kubernetes_zh_authoring_queue.jsonl --submissions path/to/reviewed.jsonl --output data/processed/reviewed_qa.jsonl
```

The compiled dataset retains the assigned document text and source URL for an
answerable item. An abstention item has an explicit
`expected_behavior: abstain_evidence_insufficient` and no invented evidence.
