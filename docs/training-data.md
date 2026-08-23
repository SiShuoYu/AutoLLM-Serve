# Training Data Boundary

The SFT artifact is generated only through `citetune export-sft`. The exporter
loads the reviewed dataset, selects rows whose split is exactly `train`, writes
Qwen-style conversational `prompt` and `completion` messages, and records hashes for the
input, output, and fixed system prompt.

The prompt contains the system instruction, user question, and cited evidence;
the completion contains only the assistant answer. TRL therefore trains on the
completion rather than treating the question and evidence as answer targets.

Validation and test rows are never copied into the SFT output. Their excluded
counts remain in the manifest so that a missing or unexpected split is visible.
Every exported row also retains `metadata.split: train`; `citetune verify-sft`
fails if a non-training row is inserted later.

The authoring queue can contain prelocked reserve source chunks from the same
split. A reserve may fill a rejected primary task only after ordinary review
approval; it never bypasses the split boundary or becomes approved automatically.

```bash
citetune export-sft \
  --dataset data/processed/reviewed_qa.jsonl \
  --output data/processed/sft_train.jsonl \
  --manifest data/processed/sft_train_manifest.json

citetune verify-sft --dataset data/processed/sft_train.jsonl
```

This export is model-ready data formatting only. It does not download model
weights, install CUDA packages, or start training.

## GPU pipeline smoke test (not a quality experiment)

Before reviewed data exists, a small deterministic subset of `needs_revision`
candidate drafts can test CUDA, QLoRA, checkpoints, and adapter loading. This
is deliberately exported through a separate command and carries
`dataset_status: candidate_smoke_unreviewed` plus
`purpose: pipeline_smoke_only` in every row and manifest. It must never be
used for validation/test metrics, model selection, or a portfolio quality
claim.

```bash
citetune export-candidate-smoke-sft \
  --queue data/processed/kubernetes_zh_clean_v2_authoring_queue.jsonl \
  --submissions data/pilot/qa_candidates_clean_v2.jsonl \
  --output data/pilot/smoke/sft_train_unreviewed.jsonl \
  --manifest data/pilot/smoke/sft_train_unreviewed_manifest.json \
  --limit 100
```
