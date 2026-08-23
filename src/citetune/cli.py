"""Command line interface for CiteTune-CN."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .authoring import (
    build_authoring_queue,
    compile_reviewed_dataset,
    render_review_packet,
    validate_authoring_submissions,
)
from .candidate_audit import audit_candidate_drafts
from .candidate_curation import curate_heldout_candidate_drafts, curate_train_candidate_drafts
from .candidate_generation import (
    QwenCandidateGenerator,
    generate_answerable_candidates,
    generate_heldout_candidates,
    generate_reserve_replacements,
    generation_preflight,
    load_candidate_generation_config,
)
from .corpus import build_kubernetes_corpus, fetch_kubernetes_docs
from .dataset import dataset_manifest, load_dataset
from .experiment import load_experiment_config, run_experiment
from .inference import QwenAnswerGenerator, generate_predictions, load_inference_config
from .model_review import QwenEvidenceReviewer, load_model_review_config, review_train_candidates
from .quality import filter_corpus_for_authoring
from .readiness import assess_data_readiness
from .retrieval import BM25Retriever
from .retrieval_evaluation import run_retrieval_evaluation
from .sft import (
    export_candidate_smoke_sft,
    export_sft_dataset,
    export_synthetic_train_sft,
    verify_sft_jsonl,
)
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
    quality.add_argument("--maximum-template-markers", type=int, default=0)
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
    export_smoke = commands.add_parser(
        "export-candidate-smoke-sft",
        help="export unreviewed train drafts only for a non-reportable GPU pipeline smoke test",
    )
    export_smoke.add_argument("--queue", required=True)
    export_smoke.add_argument("--submissions", required=True)
    export_smoke.add_argument("--output", required=True)
    export_smoke.add_argument("--manifest", required=True)
    export_smoke.add_argument("--limit", type=int, default=100)
    export_smoke.add_argument("--seed", type=int, default=42)
    export_synthetic = commands.add_parser(
        "export-synthetic-train-sft",
        help="export screened source-grounded train drafts for an adaptation run, not benchmarking",
    )
    export_synthetic.add_argument("--queue", required=True)
    export_synthetic.add_argument("--submissions", required=True)
    export_synthetic.add_argument("--output", required=True)
    export_synthetic.add_argument("--manifest", required=True)
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
    generate.add_argument(
        "--include-reserves", action="store_true", help="allow reserve source tasks after primaries"
    )
    generate_heldout = commands.add_parser(
        "generate-heldout-candidates",
        help="generate validation and test drafts in one CUDA model session",
    )
    generate_heldout.add_argument("--config", required=True)
    generate_heldout.add_argument("--queue", required=True)
    generate_heldout.add_argument("--output", required=True)
    generate_heldout.add_argument("--validation-limit", type=int, default=100)
    generate_heldout.add_argument("--test-limit", type=int, default=160)
    generate_heldout.add_argument(
        "--include-reserves",
        action="store_true",
        help="use same-split reserves after rejected drafts",
    )
    audit_candidates = commands.add_parser(
        "audit-candidate-drafts",
        help="screen unreviewed drafts and render a bounded human review batch",
    )
    audit_candidates.add_argument("--queue", required=True)
    audit_candidates.add_argument("--submissions", required=True)
    audit_candidates.add_argument("--report", required=True)
    audit_candidates.add_argument("--review-packet", required=True)
    audit_candidates.add_argument("--review-limit", type=int, default=60)
    audit_candidates.add_argument("--seed", type=int, default=42)
    curate_candidates = commands.add_parser(
        "curate-and-replace-train-drafts",
        help="conservatively reject duplicate/generic train drafts and replace them from reserves",
    )
    curate_candidates.add_argument("--config", required=True)
    curate_candidates.add_argument("--queue", required=True)
    curate_candidates.add_argument("--submissions", required=True)
    curate_candidates.add_argument("--output", required=True)
    curate_candidates.add_argument("--manifest", required=True)
    curate_heldout = commands.add_parser(
        "curate-and-replace-heldout-drafts",
        help="conservatively clean held-out drafts and replace from same-split reserves",
    )
    curate_heldout.add_argument("--config", required=True)
    curate_heldout.add_argument("--queue", required=True)
    curate_heldout.add_argument("--submissions", required=True)
    curate_heldout.add_argument("--output", required=True)
    curate_heldout.add_argument("--manifest", required=True)
    top_up_candidates = commands.add_parser(
        "top-up-reserve-candidates",
        help="fill a known train candidate shortfall from reserves with bounded retries",
    )
    top_up_candidates.add_argument("--config", required=True)
    top_up_candidates.add_argument("--queue", required=True)
    top_up_candidates.add_argument("--output", required=True)
    top_up_candidates.add_argument("--requested-count", required=True, type=int)
    model_review = commands.add_parser(
        "model-pre-review-train",
        help="run a pinned independent model reviewer on train drafts only",
    )
    model_review.add_argument("--config", required=True)
    model_review.add_argument("--queue", required=True)
    model_review.add_argument("--submissions", required=True)
    model_review.add_argument("--output", required=True)
    model_review.add_argument("--manifest", required=True)
    generate_predictions_command = commands.add_parser(
        "generate-predictions",
        help="generate Base/RAG/QLoRA predictions with real timing, without scoring",
    )
    generate_predictions_command.add_argument("--config", required=True)
    generate_predictions_command.add_argument("--dataset", required=True)
    generate_predictions_command.add_argument("--output", required=True)
    generate_predictions_command.add_argument("--manifest", required=True)
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
    if args.command == "export-candidate-smoke-sft":
        smoke_manifest = export_candidate_smoke_sft(
            args.queue,
            args.submissions,
            args.output,
            args.manifest,
            limit=args.limit,
            seed=args.seed,
        )
        print(json.dumps(smoke_manifest.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    if args.command == "export-synthetic-train-sft":
        synthetic_manifest = export_synthetic_train_sft(
            args.queue, args.submissions, args.output, args.manifest
        )
        print(
            json.dumps(synthetic_manifest.as_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        )
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
            include_reserves=args.include_reserves,
        )
        print(json.dumps({"generated_candidates": generated_count}, ensure_ascii=False, indent=2))
    if args.command == "generate-heldout-candidates":
        generation_config = load_candidate_generation_config(args.config)
        heldout_counts = generate_heldout_candidates(
            args.queue,
            args.output,
            QwenCandidateGenerator(generation_config),
            validation_limit=args.validation_limit,
            test_limit=args.test_limit,
            include_reserves=args.include_reserves,
        )
        print(json.dumps({"generated_candidates": heldout_counts}, ensure_ascii=False, indent=2))
    if args.command == "audit-candidate-drafts":
        audit_report = audit_candidate_drafts(
            args.queue,
            args.submissions,
            args.report,
            args.review_packet,
            review_limit=args.review_limit,
            seed=args.seed,
        )
        print(json.dumps(audit_report.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    if args.command == "curate-and-replace-train-drafts":
        curation_plan = curate_train_candidate_drafts(args.queue, args.submissions, args.output)
        generation_config = load_candidate_generation_config(args.config)
        replacement_result = generate_reserve_replacements(
            args.queue,
            args.output,
            QwenCandidateGenerator(generation_config),
            requested_count=curation_plan.replacement_requested_count,
        )
        output_path = Path(args.output)
        curation_report = {
            **curation_plan.as_dict(),
            "generated_replacement_count": replacement_result.generated_count,
            "generation_pass_count": replacement_result.generation_pass_count,
            "unfilled_replacement_count": replacement_result.unfilled_count,
            "final_output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        }
        manifest_path = Path(args.manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(curation_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(curation_report, ensure_ascii=False, indent=2, sort_keys=True))
    if args.command == "curate-and-replace-heldout-drafts":
        heldout_plan = curate_heldout_candidate_drafts(args.queue, args.submissions, args.output)
        generation_config = load_candidate_generation_config(args.config)
        generator = QwenCandidateGenerator(generation_config)
        replacement_report: dict[str, dict[str, int]] = {}
        for data_split, requested_count in heldout_plan.replacement_requested_counts.items():
            result = generate_reserve_replacements(
                args.queue,
                args.output,
                generator,
                requested_count=requested_count,
                split=data_split,
            )
            replacement_report[data_split] = {
                "generated_count": result.generated_count,
                "generation_pass_count": result.generation_pass_count,
                "unfilled_count": result.unfilled_count,
            }
        output_path = Path(args.output)
        heldout_report = {
            **heldout_plan.as_dict(),
            "replacement_results": replacement_report,
            "final_output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        }
        manifest_path = Path(args.manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(heldout_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(heldout_report, ensure_ascii=False, indent=2, sort_keys=True))
    if args.command == "top-up-reserve-candidates":
        generation_config = load_candidate_generation_config(args.config)
        top_up_result = generate_reserve_replacements(
            args.queue,
            args.output,
            QwenCandidateGenerator(generation_config),
            requested_count=args.requested_count,
        )
        print(
            json.dumps(
                {
                    "generated_replacement_count": top_up_result.generated_count,
                    "generation_pass_count": top_up_result.generation_pass_count,
                    "unfilled_replacement_count": top_up_result.unfilled_count,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    if args.command == "model-pre-review-train":
        review_config = load_model_review_config(args.config)
        review_report = review_train_candidates(
            args.queue,
            args.submissions,
            args.output,
            QwenEvidenceReviewer(review_config),
        )
        manifest_path = Path(args.manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(review_report.as_dict(), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps(review_report.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    if args.command == "generate-predictions":
        inference_config = load_inference_config(args.config)
        inference_report = generate_predictions(
            inference_config,
            args.dataset,
            args.output,
            QwenAnswerGenerator(inference_config),
        )
        manifest_path = Path(args.manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(inference_report.as_dict(), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps(inference_report.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
