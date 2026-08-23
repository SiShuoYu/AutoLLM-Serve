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

### Precision setting

The QLoRA configuration records its `training.mixed_precision` mode explicitly.
For the current RTX 3060 / CUDA environment it is `none`: the 4-bit model
still performs its quantized compute with FP16, while PyTorch's outer FP16
gradient scaler is disabled for compatibility with float32 LoRA gradients.
This is a reproducibility setting, not a relaxation of the training-data gate.

## Independent model pre-review

`citetune model-pre-review-train` uses a separately pinned reviewer model to
inspect only `train` candidate drafts against their locked source chunks. Its
manifest records queue, input, and output hashes. An approval is explicitly
labelled `reviewer_type: model`; it is useful for training-data triage but is
not human ground truth. Validation and test rows are never changed by this
command and still require human approval before benchmark reporting.

## Source-grounded synthetic training data

After deterministic corpus filtering, duplicate/generic-draft curation, and
candidate risk screening, `citetune export-synthetic-train-sft` may export the
remaining train-only candidates for a formal adaptation run. Every record and
manifest marks this dataset as synthetic and `train_only_not_benchmark`.
It is legitimate to report the training configuration and GPU cost, but not to
claim model quality until a separately human-approved held-out benchmark exists.

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
