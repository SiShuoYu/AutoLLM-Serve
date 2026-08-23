"""Resumable, source-grounded QA draft generation for the GPU data stage."""

from __future__ import annotations

import importlib.util
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml

from .authoring import validate_authoring_submissions


@dataclass(frozen=True, slots=True)
class CandidateGenerationConfig:
    model_id: str
    model_revision: str
    max_new_tokens: int
    temperature: float
    seed: int


@dataclass(frozen=True, slots=True)
class CandidateGenerationPreflight:
    model_id: str
    model_revision: str
    eligible_answerable_tasks: int
    existing_submission_count: int
    remaining_answerable_tasks: int
    gpu_required: bool
    gpu_dependencies_present: dict[str, bool]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class CandidateGenerator(Protocol):
    author_id: str

    def generate(self, task: dict[str, Any]) -> tuple[str, str]: ...


def _read_jsonl(path: Path, *, allow_missing: bool = False) -> list[dict[str, Any]]:
    if allow_missing and not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row: Any = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on line {line_number} of {path}") from error
        if not isinstance(row, dict):
            raise ValueError(f"line {line_number} of {path} must contain an object")
        rows.append(row)
    return rows


def load_candidate_generation_config(path: str | Path) -> CandidateGenerationConfig:
    raw: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("generation"), dict):
        raise ValueError("config must contain a generation mapping")
    generation = raw["generation"]
    model_id = generation.get("model_id")
    revision = generation.get("model_revision")
    max_new_tokens = generation.get("max_new_tokens")
    temperature = generation.get("temperature")
    seed = generation.get("seed")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("generation.model_id must be a non-empty string")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise ValueError("generation.model_revision must be a full commit hash")
    if (
        not isinstance(max_new_tokens, int)
        or isinstance(max_new_tokens, bool)
        or max_new_tokens <= 0
    ):
        raise ValueError("generation.max_new_tokens must be a positive integer")
    if (
        not isinstance(temperature, (int, float))
        or isinstance(temperature, bool)
        or temperature < 0
    ):
        raise ValueError("generation.temperature must be non-negative")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("generation.seed must be an integer")
    return CandidateGenerationConfig(
        model_id=model_id.strip(),
        model_revision=revision,
        max_new_tokens=max_new_tokens,
        temperature=float(temperature),
        seed=seed,
    )


def generation_preflight(
    config: CandidateGenerationConfig,
    queue_path: str | Path,
    output_path: str | Path,
    *,
    split: str = "train",
) -> CandidateGenerationPreflight:
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    queue = _read_jsonl(Path(queue_path))
    existing = _read_jsonl(Path(output_path), allow_missing=True)
    existing_ids = {row.get("task_id") for row in existing}
    eligible = [
        row
        for row in queue
        if row.get("split") == split
        and row.get("task_type") == "answerable"
        and row.get("queue_role", "primary") == "primary"
    ]
    dependencies = {
        package: importlib.util.find_spec(package) is not None
        for package in ("torch", "transformers", "bitsandbytes")
    }
    return CandidateGenerationPreflight(
        model_id=config.model_id,
        model_revision=config.model_revision,
        eligible_answerable_tasks=len(eligible),
        existing_submission_count=len(existing),
        remaining_answerable_tasks=sum(row.get("task_id") not in existing_ids for row in eligible),
        gpu_required=True,
        gpu_dependencies_present=dependencies,
    )


def generate_answerable_candidates(
    queue_path: str | Path,
    output_path: str | Path,
    generator: CandidateGenerator,
    *,
    split: str = "train",
    limit: int = 25,
) -> int:
    """Append valid drafts, record unparsable outputs, and skip completed tasks."""
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    if limit <= 0:
        raise ValueError("limit must be positive")
    queue_file = Path(queue_path)
    output = Path(output_path)
    queue = _read_jsonl(queue_file)
    existing = _read_jsonl(output, allow_missing=True)
    existing_ids = {row.get("task_id") for row in existing}
    selected = [
        row
        for row in queue
        if row.get("split") == split
        and row.get("task_type") == "answerable"
        and row.get("queue_role", "primary") == "primary"
        and row.get("task_id") not in existing_ids
    ][:limit]
    if not selected:
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    generated_count = 0
    with output.open("a", encoding="utf-8") as handle:
        for task in selected:
            source = task.get("source_chunk")
            if not isinstance(source, dict) or not isinstance(source.get("chunk_id"), str):
                raise ValueError(f"task {task.get('task_id')} has no assigned source")
            try:
                question, answer = generator.generate(task)
                if not question.strip() or not answer.strip():
                    raise ValueError("generator returned empty content")
            except ValueError:
                row = {
                    "task_id": task["task_id"],
                    "author_id": generator.author_id,
                    "review": {
                        "status": "rejected",
                        "notes": "GPU 模型未返回完整问答，已跳过并保留任务记录。",
                    },
                }
            else:
                row = {
                    "task_id": task["task_id"],
                    "question": question.strip(),
                    "reference_answer": answer.strip(),
                    "reference_citation_ids": [source["chunk_id"]],
                    "author_id": generator.author_id,
                    "review": {
                        "status": "needs_revision",
                        "notes": "GPU 模型生成草稿，等待独立审核。",
                    },
                }
                generated_count += 1
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
    validate_authoring_submissions(queue_file, output)
    return generated_count


class QwenCandidateGenerator:
    """4-bit Qwen generator. Construction is intentionally CUDA-only."""

    def __init__(self, config: CandidateGenerationConfig) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as error:
            raise RuntimeError("install the project training dependencies first") from error
        if not torch.cuda.is_available():
            raise RuntimeError("candidate generation requires a CUDA GPU")
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
        self.author_id = f"model:{config.model_id}@{config.model_revision}"

    def generate(self, task: dict[str, Any]) -> tuple[str, str]:
        source = task["source_chunk"]
        messages = [
            {
                "role": "system",
                "content": (
                    "你是中文 Kubernetes 数据标注员。只根据给定证据写一个自然、明确的问题"
                    "及简洁答案。不要使用外部知识。仅输出 JSON 对象，键为 question 和 answer。"
                ),
            },
            {
                "role": "user",
                "content": f"证据 ID：{source['chunk_id']}\n证据：\n{source['text']}",
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
        return parse_candidate_output(text)


def parse_candidate_output(text: str) -> tuple[str, str]:
    """Accept JSON first, then Qwen's common Chinese labelled fallback format."""
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            raw, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(raw, dict):
            return _validated_candidate_fields(raw)

    labelled = re.search(
        r"(?:\*{1,3}\s*)?(?:问题|question)(?:\s*\*{1,3})?\s*[:：]\s*(.+?)"
        r"\s*(?:\*{1,3}\s*)?(?:答案|answer)(?:\s*\*{1,3})?\s*[:：]\s*(.+)\Z",
        text.strip(),
        flags=re.IGNORECASE | re.DOTALL,
    )
    if labelled:
        return _validated_candidate_fields(
            {"question": labelled.group(1), "answer": labelled.group(2)}
        )

    preview = " ".join(text.split())[:240]
    raise ValueError(f"model output contained no parseable question/answer: {preview!r}")


def _validated_candidate_fields(raw: dict[str, Any]) -> tuple[str, str]:
    question = raw.get("question")
    answer = raw.get("answer")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("model output has no question")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("model output has no answer")
    return question.strip(), answer.strip()
