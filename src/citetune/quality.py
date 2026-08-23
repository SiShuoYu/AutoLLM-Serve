"""Deterministic source-quality filtering before QA authoring."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class CorpusQualityManifest:
    input_sha256: str
    input_chunk_count: int
    kept_chunk_count: int
    kept_document_count: int
    minimum_characters: int
    minimum_chinese_ratio: float
    maximum_template_markers: int
    rejection_counts: dict[str, int]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _non_whitespace_length(text: str) -> int:
    return len("".join(text.split()))


def _chinese_ratio(text: str) -> float:
    total = _non_whitespace_length(text)
    return sum("\u4e00" <= character <= "\u9fff" for character in text) / total if total else 0.0


def _template_markers(text: str) -> int:
    return sum(text.count(marker) for marker in ("{{<", "{{%", "<!--", "-->", "php-template"))


def filter_corpus_for_authoring(
    corpus_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
    *,
    minimum_characters: int = 240,
    minimum_chinese_ratio: float = 0.12,
    maximum_template_markers: int = 0,
) -> CorpusQualityManifest:
    """Keep source chunks likely to support a self-contained Chinese QA item.

    This is a transparent source-selection filter, not a claim that a retained
    chunk is a correct question-answer pair. Human review remains mandatory.
    """
    if (
        minimum_characters <= 0
        or not 0 <= minimum_chinese_ratio <= 1
        or maximum_template_markers < 0
    ):
        raise ValueError("invalid corpus-quality threshold")
    corpus = Path(corpus_path)
    source_bytes = corpus.read_bytes()
    rows: list[dict[str, Any]] = []
    rejections: Counter[str] = Counter()
    for line_number, line in enumerate(source_bytes.decode("utf-8").splitlines(), start=1):
        try:
            row: Any = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on line {line_number} of {corpus}") from error
        if not isinstance(row, dict) or not isinstance(row.get("text"), str):
            raise ValueError(f"corpus line {line_number} has no text")
        if not isinstance(row.get("document_id"), str):
            raise ValueError(f"corpus line {line_number} has no document_id")
        text = row["text"]
        reasons: list[str] = []
        if _non_whitespace_length(text) < minimum_characters:
            reasons.append("too_short")
        if _chinese_ratio(text) < minimum_chinese_ratio:
            reasons.append("low_chinese_ratio")
        if _template_markers(text) > maximum_template_markers:
            reasons.append("too_many_template_markers")
        if reasons:
            rejections.update(reasons)
        else:
            rows.append(row)
    if not rows:
        raise ValueError("quality filter removed every corpus chunk")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = CorpusQualityManifest(
        input_sha256=hashlib.sha256(source_bytes).hexdigest(),
        input_chunk_count=sum(
            1 for line in source_bytes.decode("utf-8").splitlines() if line.strip()
        ),
        kept_chunk_count=len(rows),
        kept_document_count=len({row["document_id"] for row in rows}),
        minimum_characters=minimum_characters,
        minimum_chinese_ratio=minimum_chinese_ratio,
        maximum_template_markers=maximum_template_markers,
        rejection_counts=dict(sorted(rejections.items())),
    )
    manifest_file = Path(manifest_path)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(
        json.dumps(manifest.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
