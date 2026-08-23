# CiteTune-CN Project Charter

## Objective

Build and evaluate a Chinese Kubernetes documentation assistant that answers
only from cited official documentation. The portfolio contribution is not a UI;
it is a reproducible study of how retrieval and parameter-efficient adaptation
change grounded-answer quality.

## Fixed scope

- **Corpus:** the Chinese documentation under `content/zh-cn/docs/` from a
  pinned revision of `kubernetes/website`, with source URLs, document hashes,
  and CC BY 4.0 attribution retained.
- **Base model:** `Qwen/Qwen2.5-1.5B-Instruct` for the RTX 3060 track. A larger
  model may be added on a school GPU, but never replaces the small-model result.
- **Task:** answer a Chinese Kubernetes question with one or more source-chunk
  citations, or explicitly abstain when the supplied evidence cannot answer it.
- **Systems compared:** base model, base + RAG, QLoRA/SFT, and RAG + QLoRA.

## Dataset and evaluation gates

Before any result is described as portfolio-ready, the repository must contain:

1. A pinned corpus manifest with repository revision, file paths, source URLs,
   document hashes, and license attribution.
2. At least 1,200 training instances and a locked, human-reviewed held-out set
   of at least 200 questions. The held-out set must include at least 40
   unanswerable or insufficient-evidence questions.
   Training and validation also include insufficient-evidence examples so that
   abstention is learned and tuned without exposing held-out test questions.
3. Annotation guidelines and a second-pass review of at least 100 held-out
   items. Split assignment happens before training and is recorded in Git.
4. For every system: three fixed-seed runs, raw predictions, run metadata,
   model revision, tokenizer revision, hardware, package versions, and exact
   configuration.
5. A blind human audit of at least 100 answers per final system, recording
   factuality, evidence support, citation validity, and abstention correctness.

## Success target and reporting rule

The optimization target is to improve the human evidence-supported-answer rate
by **at least 15 percentage points** over the base model on the locked test set,
while maintaining at least 95% citation-ID validity. Retrieval Recall@5, answer
quality, latency, and cost are reported together.

If the target is not achieved, the work is still reported honestly as a negative
or inconclusive experiment. The project must never claim an improvement from a
single run, a changing test set, an LLM self-judge alone, or mock GPU values.

## Engineering release gates

- Unit, integration, and end-to-end tests; at least 80% coverage for project
  code, with CI enforcing format, lint, types, and tests.
- Deterministic data builds with hash manifests and leakage checks.
- Docker/WSL instructions for the RTX 3060 and a separate documented school-GPU
  profile.
- Data card, model card, experiment report, error taxonomy, and limitations.
- A reproducibility command that rebuilds an evaluation run from a pinned data
  manifest and experiment config.

## Explicit non-goals

- Training a foundation model from scratch.
- Claiming production reliability from a student-scale benchmark.
- Building a generic chat UI before the evaluation and adaptation evidence.
- Replacing vLLM, Kubernetes, or any source project.
