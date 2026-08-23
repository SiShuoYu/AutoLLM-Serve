"""Unified, provenance-recorded GPU inference for Base/RAG/QLoRA variants."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Protocol

import yaml

from .dataset import load_dataset
from .retrieval import BM25Retriever, RetrievalResult
from .schemas import ModelPrediction

InferenceMode = Literal["base", "rag", "qlora", "rag_qlora"]


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    system_name: str
    mode: InferenceMode
    model_id: str
    model_revision: str
    adapter_path: Path | None
    corpus_path: Path
    top_k: int
    max_new_tokens: int
    temperature: float
    seed: int


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    answer: str
    latency_ms: float


@dataclass(frozen=True, slots=True)
class InferenceReport:
    system_name: str
    mode: str
    model_id: str
    model_revision: str
    dataset_sha256: str
    corpus_sha256: str
    output_sha256: str
    generated_prediction_count: int
    mean_latency_ms: float
    peak_gpu_memory_bytes: int | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class AnswerGenerator(Protocol):
    peak_gpu_memory_bytes: int | None

    def generate(self, question: str, evidence: list[RetrievalResult]) -> GeneratedAnswer: ...


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def load_inference_config(path: str | Path) -> InferenceConfig:
    config_path = Path(path)
    raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("inference"), dict):
        raise ValueError("config must contain an inference mapping")
    inference = raw["inference"]
    mode = inference.get("mode")
    if mode not in {"base", "rag", "qlora", "rag_qlora"}:
        raise ValueError("inference.mode must be base, rag, qlora, or rag_qlora")
    model = inference.get("model")
    if not isinstance(model, dict):
        raise ValueError("inference.model must be a mapping")
    revision = _text(model.get("revision"), "inference.model.revision")
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("inference.model.revision must be a full commit hash")
    adapter = inference.get("adapter_path")
    if mode in {"qlora", "rag_qlora"} and (not isinstance(adapter, str) or not adapter):
        raise ValueError(f"inference.adapter_path is required for {mode}")
    if mode in {"base", "rag"} and adapter is not None:
        raise ValueError(f"inference.adapter_path must be null for {mode}")
    root = config_path.parent
    top_k = inference.get("top_k")
    max_new_tokens = inference.get("max_new_tokens")
    temperature = inference.get("temperature")
    seed = inference.get("seed")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("inference.top_k must be a positive integer")
    if (
        not isinstance(max_new_tokens, int)
        or isinstance(max_new_tokens, bool)
        or max_new_tokens <= 0
    ):
        raise ValueError("inference.max_new_tokens must be a positive integer")
    if (
        not isinstance(temperature, (int, float))
        or isinstance(temperature, bool)
        or temperature < 0
    ):
        raise ValueError("inference.temperature must be non-negative")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("inference.seed must be an integer")
    return InferenceConfig(
        system_name=_text(inference.get("system_name"), "inference.system_name"),
        mode=mode,
        model_id=_text(model.get("id"), "inference.model.id"),
        model_revision=revision,
        adapter_path=(root / adapter).resolve() if isinstance(adapter, str) else None,
        corpus_path=(root / _text(inference.get("corpus_path"), "inference.corpus_path")).resolve(),
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        temperature=float(temperature),
        seed=seed,
    )


def _citation_ids(answer: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(match.group(1).strip() for match in re.finditer(r"\[([^\]]+)]", answer))
    )


def _answer_without_citations(answer: str) -> str:
    return re.sub(r"\s*\[[^\]]+]", "", answer).strip()


def generate_predictions(
    config: InferenceConfig,
    dataset_path: str | Path,
    output_path: str | Path,
    generator: AnswerGenerator,
) -> InferenceReport:
    """Generate one prediction per locked example without computing quality scores."""
    dataset = Path(dataset_path)
    output = Path(output_path)
    examples = load_dataset(dataset)
    retriever = BM25Retriever.from_jsonl(config.corpus_path)
    use_retrieval = config.mode in {"rag", "rag_qlora"}
    predictions: list[ModelPrediction] = []
    for example in examples:
        evidence = retriever.search(example.question, top_k=config.top_k) if use_retrieval else []
        generated = generator.generate(example.question, evidence)
        predictions.append(
            ModelPrediction(
                example_id=example.example_id,
                system_name=config.system_name,
                answer=_answer_without_citations(generated.answer),
                citation_ids=_citation_ids(generated.answer),
                latency_ms=generated.latency_ms,
            )
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(
                json.dumps(prediction.as_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            )
    return InferenceReport(
        system_name=config.system_name,
        mode=config.mode,
        model_id=config.model_id,
        model_revision=config.model_revision,
        dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
        corpus_sha256=hashlib.sha256(config.corpus_path.read_bytes()).hexdigest(),
        output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        generated_prediction_count=len(predictions),
        mean_latency_ms=sum(prediction.latency_ms or 0 for prediction in predictions)
        / len(predictions),
        peak_gpu_memory_bytes=generator.peak_gpu_memory_bytes,
    )


class QwenAnswerGenerator:  # pragma: no cover - exercised on CUDA hardware
    """CUDA-only Base or QLoRA generator that records real per-request latency."""

    def __init__(self, config: InferenceConfig) -> None:
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as error:
            raise RuntimeError("install the project training dependencies first") from error
        if not torch.cuda.is_available():
            raise RuntimeError("prediction generation requires a CUDA GPU")
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
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            revision=config.model_revision,
            quantization_config=quantization,
            device_map={"": 0},
        )
        if config.adapter_path is not None:
            if not config.adapter_path.exists():
                raise ValueError(f"adapter path does not exist: {config.adapter_path}")
            model = PeftModel.from_pretrained(model, str(config.adapter_path))
        model.eval()
        self._model = model
        torch.cuda.reset_peak_memory_stats()
        self.peak_gpu_memory_bytes: int | None = None

    def generate(self, question: str, evidence: list[RetrievalResult]) -> GeneratedAnswer:
        if evidence:
            evidence_text = "\n\n".join(f"[{item.chunk_id}] {item.text}" for item in evidence)
        else:
            evidence_text = "（未提供检索证据）"
        messages = [
            {
                "role": "system",
                "content": (
                    "你是 Kubernetes 中文问答助手。仅在证据足够时回答；证据不足时明确说明证据不足。"
                    "如使用证据，回答末尾列出实际使用的引用 ID，格式为 [chunk-id]。"
                ),
            },
            {"role": "user", "content": f"问题：{question}\n\n证据：\n{evidence_text}"},
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
        self._torch.cuda.synchronize()
        started = perf_counter()
        with self._torch.no_grad():
            generated = self._model.generate(**inputs, **generation_args)
        self._torch.cuda.synchronize()
        elapsed_ms = (perf_counter() - started) * 1000
        self.peak_gpu_memory_bytes = self._torch.cuda.max_memory_allocated()
        answer = self._tokenizer.decode(
            generated[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        )
        return GeneratedAnswer(answer=answer, latency_ms=elapsed_ms)


def inference_dependencies_present() -> dict[str, bool]:
    return {
        package: importlib.util.find_spec(package) is not None
        for package in ("torch", "transformers", "bitsandbytes", "peft")
    }
