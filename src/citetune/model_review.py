"""Pinned-model, evidence-aware pre-review for training candidate drafts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import yaml

from .authoring import validate_authoring_submissions

ReviewVerdict = Literal["approved", "needs_revision", "rejected"]


@dataclass(frozen=True, slots=True)
class ModelReviewConfig:
    model_id: str
    model_revision: str
    max_new_tokens: int
    temperature: float
    seed: int


@dataclass(frozen=True, slots=True)
class ReviewDecision:
    verdict: ReviewVerdict
    evidence_supported: bool
    notes: str


@dataclass(frozen=True, slots=True)
class ModelReviewReport:
    model_id: str
    model_revision: str
    queue_sha256: str
    submissions_sha256: str
    output_sha256: str
    reviewed_train_candidate_count: int
    verdict_counts: dict[str, int]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class EvidenceReviewer(Protocol):
    reviewer_id: str

    def review(self, task: dict[str, Any], candidate: dict[str, Any]) -> ReviewDecision: ...


def load_model_review_config(path: str | Path) -> ModelReviewConfig:
    raw: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("review"), dict):
        raise ValueError("config must contain a review mapping")
    review = raw["review"]
    model_id = review.get("model_id")
    revision = review.get("model_revision")
    max_new_tokens = review.get("max_new_tokens")
    temperature = review.get("temperature")
    seed = review.get("seed")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("review.model_id must be a non-empty string")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise ValueError("review.model_revision must be a full commit hash")
    if (
        not isinstance(max_new_tokens, int)
        or isinstance(max_new_tokens, bool)
        or max_new_tokens <= 0
    ):
        raise ValueError("review.max_new_tokens must be a positive integer")
    if (
        not isinstance(temperature, (int, float))
        or isinstance(temperature, bool)
        or temperature < 0
    ):
        raise ValueError("review.temperature must be non-negative")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("review.seed must be an integer")
    return ModelReviewConfig(
        model_id=model_id.strip(),
        model_revision=revision,
        max_new_tokens=max_new_tokens,
        temperature=float(temperature),
        seed=seed,
    )


def review_train_candidates(
    queue_path: str | Path,
    submissions_path: str | Path,
    output_path: str | Path,
    reviewer: EvidenceReviewer,
) -> ModelReviewReport:
    """Pre-review train-only drafts, never processing held-out examples."""
    queue_file = Path(queue_path)
    submissions_file = Path(submissions_path)
    output = Path(output_path)
    if output.exists():
        raise ValueError(f"refusing to overwrite existing review output: {output}")
    validate_authoring_submissions(queue_file, submissions_file)
    queue = {
        row["task_id"]: row
        for row in (
            json.loads(line) for line in queue_file.read_text(encoding="utf-8").splitlines() if line
        )
    }
    submissions = [
        json.loads(line)
        for line in submissions_file.read_text(encoding="utf-8").splitlines()
        if line
    ]
    verdicts: Counter[str] = Counter()
    reviewed = 0
    reviewed_rows: list[dict[str, Any]] = []
    for row in submissions:
        task = queue[row["task_id"]]
        if (
            task.get("split") != "train"
            or task.get("task_type") != "answerable"
            or row.get("review", {}).get("status") != "needs_revision"
        ):
            reviewed_rows.append(row)
            continue
        try:
            decision = reviewer.review(task, row)
        except ValueError as error:
            decision = ReviewDecision(
                verdict="needs_revision",
                evidence_supported=False,
                notes=f"模型预审输出无效：{error}",
            )
        reviewed += 1
        verdicts.update([decision.verdict])
        review: dict[str, object] = {"status": decision.verdict, "notes": decision.notes}
        if decision.verdict == "approved":
            review.update(
                {
                    "reviewer_id": reviewer.reviewer_id,
                    "reviewer_type": "model",
                    "evidence_supported": decision.evidence_supported,
                }
            )
        reviewed_rows.append({**row, "review": review})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in reviewed_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    validate_authoring_submissions(queue_file, output)
    reviewer_name, _, reviewer_revision = reviewer.reviewer_id.partition("@")
    return ModelReviewReport(
        model_id=reviewer_name.removeprefix("model:"),
        model_revision=reviewer_revision,
        queue_sha256=hashlib.sha256(queue_file.read_bytes()).hexdigest(),
        submissions_sha256=hashlib.sha256(submissions_file.read_bytes()).hexdigest(),
        output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        reviewed_train_candidate_count=reviewed,
        verdict_counts=dict(sorted(verdicts.items())),
    )


class QwenEvidenceReviewer:
    """CUDA-only independent reviewer model, separate from the draft generator."""

    def __init__(self, config: ModelReviewConfig) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as error:
            raise RuntimeError("install the project training dependencies first") from error
        if not torch.cuda.is_available():
            raise RuntimeError("model pre-review requires a CUDA GPU")
        torch.manual_seed(config.seed)
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        self._torch = torch
        self._config = config
        self._tokenizer = AutoTokenizer.from_pretrained(
            config.model_id, revision=config.model_revision
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            revision=config.model_revision,
            quantization_config=quantization,
            device_map={"": 0},
        )
        self.reviewer_id = f"model:{config.model_id}@{config.model_revision}"

    def review(self, task: dict[str, Any], candidate: dict[str, Any]) -> ReviewDecision:
        source = task["source_chunk"]
        messages = [
            {
                "role": "system",
                "content": (
                    "你是独立的中文 Kubernetes 数据审核员。只依据原文证据判断候选问题和答案。"
                    "仅输出 JSON：verdict 为 approved、needs_revision 或 rejected；"
                    "evidence_supported 为 true 或 false；notes 是不超过 40 字的中文理由。"
                    "只有问题自然、答案完全受证据支持时才可 approved。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"原文证据：\n{source['text']}\n\n候选问题：{candidate['question']}"
                    f"\n候选答案：{candidate['reference_answer']}"
                ),
            },
        ]
        inputs = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)
        generation_args: dict[str, Any] = {
            "max_new_tokens": self._config.max_new_tokens,
            "do_sample": self._config.temperature > 0,
        }
        if self._config.temperature > 0:
            generation_args["temperature"] = self._config.temperature
        with self._torch.no_grad():
            generated = self._model.generate(**inputs, **generation_args)
        text = self._tokenizer.decode(
            generated[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        )
        return parse_review_output(text)


def parse_review_output(text: str) -> ReviewDecision:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            raw, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(raw, dict):
            continue
        verdict = raw.get("verdict")
        evidence_supported = raw.get("evidence_supported")
        notes = raw.get("notes")
        if verdict not in {"approved", "needs_revision", "rejected"}:
            raise ValueError("review output has invalid verdict")
        if not isinstance(evidence_supported, bool):
            raise ValueError("review output has no boolean evidence_supported")
        if not isinstance(notes, str) or not notes.strip():
            raise ValueError("review output has no notes")
        if verdict == "approved" and not evidence_supported:
            raise ValueError("approved review must be evidence supported")
        return ReviewDecision(
            verdict=verdict, evidence_supported=evidence_supported, notes=notes.strip()
        )
    raise ValueError("review output did not contain a JSON object")


def model_review_dependencies_present() -> dict[str, bool]:
    return {
        package: importlib.util.find_spec(package) is not None
        for package in ("torch", "transformers", "bitsandbytes")
    }
