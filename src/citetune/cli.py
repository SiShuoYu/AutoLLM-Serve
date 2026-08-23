"""Command line interface for CiteTune-CN."""

from __future__ import annotations

import argparse
import json

from .authoring import (
    build_authoring_queue,
    compile_reviewed_dataset,
    render_review_packet,
    validate_authoring_submissions,
)
from .candidate_generation import (
    QwenCandidateGenerator,
    generate_answerable_candidates,
    generation_preflight,
    load_candidate_generation_config,
)
from .corpus import build_kubernetes_corpus, fetch_kubernetes_docs
from .dataset import dataset_manifest, load_dataset
from .experiment import load_experiment_config, run_experiment
from .quality import filter_corpus_for_authoring
from .readiness import assess_data_readiness
from .retrieval import BM25Retriever
from .retrieval_evaluation import run_retrieval_evaluation
from .sft import export_sft_dataset, verify_sft_jsonl
from .splits import split_corpus_by_document
from .training import load_qlora_config, run_qlora_training, training_preflight


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run reproducible grounded-answer evaluations.")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-dataset", help="validate a JSONL grounded dataset")
    validate.add_argument("--dataset", required=True)
    run = commands.add_parser("run", help="evaluate a configured prediction set")
    run.add_argument("--config", required=True)
    fetch = commands.add_parser(
        "fetch-kubernetes-docs", help="sparse-clone Chinese Kubernetes docs"
    )
    fetch.add_argument("--destination", default="data/raw/kubernetes-website")
    fetch.add_argument("--ref", default="main")
    build = commands.add_parser("build-kubernetes-corpus", help="build corpus chunks and manifest")
    build.add_argument("--source-root", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--manifest", required=True)
    build.add_argument("--chunk-size", type=int, default=1200)
    split = commands.add_parser("split-corpus", help="split corpus by document without leakage")
    split.add_argument("--corpus", required=True)
    split.add_argument("--output-dir", required=True)
    split.add_argument("--manifest", required=True)
    split.add_argument("--seed", type=int, default=42)
    authoring = commands.add_parser(
        "build-authoring-queue", help="select source chunks for reviewed QA"
    )
    authoring.add_argument("--splits", required=True)
    authoring.add_argument("--output", required=True)
    authoring.add_argument("--manifest", required=True)
    authoring.add_argument("--seed", type=int, default=42)
    validate_authoring = commands.add_parser(
        "validate-authoring", help="validate human-reviewed QA against the locked task queue"
    )
    validate_authoring.add_argument("--queue", required=True)
    validate_authoring.add_argument("--submissions", required=True)
    validate_authoring.add_argument("--require-complete", action="store_true")
    compile_authoring = commands.add_parser(
        "compile-reviewed-dataset", help="compile approved reviewed QA into an evaluation dataset"
    )
    compile_authoring.add_argument("--queue", required=True)
    compile_authoring.add_argument("--submissions", required=True)
    compile_authoring.add_argument("--output", required=True)
    compile_authoring.add_argument("--require-complete", action="store_true")
    retrieve = commands.add_parser("retrieve", help="retrieve locked corpus evidence with BM25")
    retrieve.add_argument("--corpus", required=True)
    retrieve.add_argument("--query", required=True)
    retrieve.add_argument("--top-k", type=int, default=5)
    quality = commands.add_parser(
        "filter-corpus", help="filter low-signal source chunks before QA authoring"
    )
    quality.add_argument("--corpus", required=True)
    quality.add_argument("--output", required=True)
    quality.add_argument("--manifest", required=True)
    quality.add_argument("--minimum-characters", type=int, default=240)
    quality.add_argument("--minimum-chinese-ratio", type=float, default=0.12)
    quality.add_argument("--maximum-template-markers", type=int, default=3)
    packet = commands.add_parser(
        "render-review-packet", help="render candidate QA and evidence for human review"
    )
    packet.add_argument("--queue", required=True)
    packet.add_argument("--submissions", required=True)
    packet.add_argument("--output", required=True)
    retrieval_eval = commands.add_parser(
        "evaluate-retrieval", help="evaluate BM25 recall and MRR on reviewed QA"
    )
    retrieval_eval.add_argument("--corpus", required=True)
    retrieval_eval.add_argument("--dataset", required=True)
    retrieval_eval.add_argument("--output-dir", required=True)
    retrieval_eval.add_argument("--k", type=int, nargs="+", default=[1, 5, 10])
    export_sft = commands.add_parser(
        "export-sft", help="export only approved train rows to Qwen chat JSONL"
    )
    export_sft.add_argument("--dataset", required=True)
    export_sft.add_argument("--output", required=True)
    export_sft.add_argument("--manifest", required=True)
    verify_sft = commands.add_parser(
        "verify-sft", help="verify that an SFT JSONL contains train rows only"
    )
    verify_sft.add_argument("--dataset", required=True)
    preflight = commands.add_parser(
        "training-preflight", help="validate pinned QLoRA config and SFT data without GPU work"
    )
    preflight.add_argument("--config", required=True)
    train_qlora = commands.add_parser("train-qlora", help="execute QLoRA on a CUDA GPU")
    train_qlora.add_argument("--config", required=True)
    readiness = commands.add_parser(
        "data-readiness", help="check whether reviewed data can enter GPU experiments"
    )
    readiness.add_argument("--queue", required=True)
    readiness.add_argument("--submissions", required=True)
    generation_check = commands.add_parser(
        "candidate-generation-preflight",
        help="check the resumable GPU candidate generation stage without loading a model",
    )
    generation_check.add_argument("--config", required=True)
    generation_check.add_argument("--queue", required=True)
    generation_check.add_argument("--output", required=True)
    generation_check.add_argument(
        "--split", choices=["train", "validation", "test"], default="train"
    )
    generate = commands.add_parser(
        "generate-candidates", help="generate source-grounded QA drafts on a CUDA GPU"
    )
    generate.add_argument("--config", required=True)
    generate.add_argument("--queue", required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument("--split", choices=["train", "validation", "test"], default="train")
    generate.add_argument("--limit", type=int, default=25)
    args = parser.parse_args(argv)
    if args.command == "validate-dataset":
        examples = load_dataset(args.dataset)
        print(
            json.dumps(
                dataset_manifest(args.dataset, examples).as_dict(), ensure_ascii=False, indent=2
            )
        )
    if args.command == "run":
        summary = run_experiment(load_experiment_config(args.config))
        print(json.dumps(summary.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    if args.command == "fetch-kubernetes-docs":
        revision = fetch_kubernetes_docs(args.destination, args.ref)
        print(json.dumps({"destination": args.destination, "revision": revision}, indent=2))
    if args.command == "build-kubernetes-corpus":
        corpus_manifest = build_kubernetes_corpus(
            args.source_root, args.output, args.manifest, chunk_size_characters=args.chunk_size
        )
        print(json.dumps(corpus_manifest.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    if args.command == "split-corpus":
        split_manifest = split_corpus_by_document(
            args.corpus, args.output_dir, args.manifest, seed=args.seed
        )
        print(json.dumps(split_manifest.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    if args.command == "build-authoring-queue":
        queue_manifest = build_authoring_queue(
            args.splits, args.output, args.manifest, seed=args.seed
        )
        print(json.dumps(queue_manifest.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    if args.command == "validate-authoring":
        report = validate_authoring_submissions(
            args.queue, args.submissions, require_complete=args.require_complete
        )
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    if args.command == "compile-reviewed-dataset":
        report = compile_reviewed_dataset(
            args.queue,
            args.submissions,
            args.output,
            require_complete=args.require_complete,
        )
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    if args.command == "retrieve":
        results = BM25Retriever.from_jsonl(args.corpus).search(args.query, top_k=args.top_k)
        print(json.dumps([result.as_dict() for result in results], ensure_ascii=False, indent=2))
    if args.command == "filter-corpus":
        quality_manifest = filter_corpus_for_authoring(
            args.corpus,
            args.output,
            args.manifest,
            minimum_characters=args.minimum_characters,
            minimum_chinese_ratio=args.minimum_chinese_ratio,
            maximum_template_markers=args.maximum_template_markers,
        )
        print(json.dumps(quality_manifest.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    if args.command == "render-review-packet":
        report = render_review_packet(args.queue, args.submissions, args.output)
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    if args.command == "evaluate-retrieval":
        retrieval_summary = run_retrieval_evaluation(
            args.corpus, args.dataset, args.output_dir, k_values=tuple(args.k)
        )
        print(json.dumps(retrieval_summary.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    if args.command == "export-sft":
        sft_manifest = export_sft_dataset(args.dataset, args.output, args.manifest)
        print(json.dumps(sft_manifest.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    if args.command == "verify-sft":
        count = verify_sft_jsonl(args.dataset)
        print(json.dumps({"verified_train_rows": count}, ensure_ascii=False, indent=2))
    if args.command == "training-preflight":
        preflight_report = training_preflight(load_qlora_config(args.config))
        print(json.dumps(preflight_report.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    if args.command == "train-qlora":
        run_qlora_training(load_qlora_config(args.config))
    if args.command == "data-readiness":
        readiness_report = assess_data_readiness(args.queue, args.submissions)
        print(json.dumps(readiness_report.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    if args.command == "candidate-generation-preflight":
        generation_config = load_candidate_generation_config(args.config)
        generation_report = generation_preflight(
            generation_config, args.queue, args.output, split=args.split
        )
        print(json.dumps(generation_report.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    if args.command == "generate-candidates":
        generation_config = load_candidate_generation_config(args.config)
        generated_count = generate_answerable_candidates(
            args.queue,
            args.output,
            QwenCandidateGenerator(generation_config),
            split=args.split,
            limit=args.limit,
        )
        print(json.dumps({"generated_candidates": generated_count}, ensure_ascii=False, indent=2))
