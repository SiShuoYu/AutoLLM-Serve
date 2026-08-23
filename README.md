# CiteTune-CN

**CiteTune-CN** is a reproducible experiment platform for improving the
reliability of **Chinese Kubernetes documentation question answering**. It is
an algorithm-focused portfolio project: the final system will compare a base
LLM, RAG, QLoRA/SFT, and RAG + QLoRA under the same versioned dataset and
evaluation protocol.

It is deliberately not a generic chatbot demo. The central question is:

> Which adaptation strategy improves answer quality and evidence support for
> Chinese Kubernetes questions, and what latency/cost trade-off does it
> introduce?

The precise objective, hard release gates, and non-goals are in the
[project charter](docs/project-charter.md).

## V1: trustworthy evaluation foundation

V1 provides:

- strict JSONL contracts for evidence, references, predictions, and human labels;
- dataset validation plus SHA-256 manifests to prevent silent data drift;
- deterministic exact-match, character-F1, citation precision/recall, and invalid-citation checks;
- optional human factuality and unsupported-claim labels;
- reproducible YAML experiments with `run.json`, `summary.json`, and per-example results.

No hallucination rate is fabricated. In V1, evidence-support and factuality
rates are reported only when a human annotation is present. Citation overlap is
reported separately because citing the expected document does not by itself
prove that an answer is true.

The repository also includes the first transparent retrieval baseline: BM25 on
the locked corpus. It returns source chunk IDs and URLs rather than claiming
semantic quality before a reviewed benchmark exists:

```bash
citetune retrieve --corpus data/processed/kubernetes_zh_chunks.jsonl --query "Pod 如何被删除" --top-k 3
```

## Requirements

- Python 3.11+

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
citetune validate-dataset --dataset data/samples/grounded_qa.jsonl
citetune run --config configs/sample-baseline.yaml
```

The final command writes reproducible artifacts to `artifacts/sample-baseline/`.
They are intentionally ignored by Git; a published experiment will commit a
curated result bundle and its exact configuration separately.

Local quality gates match CI:

```bash
ruff format --check .
ruff check .
mypy
pytest --cov=citetune --cov-report=term-missing --cov-fail-under=80
```

## Metrics

| Metric | Meaning | Limitation |
|---|---|---|
| Exact match / character F1 | Agreement with the reference answer | Does not prove factual correctness |
| Citation precision / recall | Agreement with reference evidence IDs | Does not prove a citation supports the answer |
| Citation valid rate | Citation IDs exist in that example's evidence | Does not measure semantic support |
| Fully correct / evidence-supported rate | Human labels | Requires annotation coverage |

## Project roadmap and release gates

1. **V1 — evaluation contracts**: complete. Validate datasets, preserve run metadata, and make metrics auditable.
2. **V2 — data pipeline**: build a licensed Chinese technical-document dataset, version train/validation/test splits, and define annotation guidelines with an inter-annotator agreement check.
3. **V3 — adaptation experiments**: run base, RAG, QLoRA/SFT, and combined baselines on an RTX 3060 or school GPU; publish repeated runs and error taxonomy.
4. **V4 — algorithm optimization**: improve data selection, retrieval, or training strategy against a locked test set; quantify quality, latency, and cost trade-offs.
5. **V5 — portfolio release**: Docker, CI, model/data cards, reproducible report, and a concise technical write-up.

The finished project is considered portfolio-ready only when every published
claim links to raw result artifacts, hardware/model/configuration metadata, and
at least three repeated GPU runs. It will never substitute mock numbers for
model-quality or GPU measurements.

## Layout

```text
src/citetune/       data contracts, validation, evaluator, experiment runner
data/samples/       tiny tracked demonstration dataset and predictions
data/raw/           untracked source data
data/processed/     untracked derived data
configs/            reproducible experiment definitions
artifacts/          ignored local run outputs
tests/              contracts, metrics, and artifact persistence tests
docs/               architecture and evaluation boundaries
```

See [architecture](docs/architecture.md) for the V1 data flow.
