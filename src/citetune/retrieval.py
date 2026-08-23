"""A dependency-free, deterministic BM25 baseline for locked source corpora."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_./-]+")


def tokenize(text: str) -> list[str]:
    """Tokenize CJK characters and technical terms without a hidden model dependency."""
    return [match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text)]


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    chunk_id: str
    score: float
    text: str
    source_url: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _Chunk:
    chunk_id: str
    text: str
    source_url: str


class BM25Retriever:
    """In-memory BM25 retriever intended as a transparent RAG baseline."""

    def __init__(self, chunks: list[_Chunk], *, k1: float = 1.5, b: float = 0.75) -> None:
        if not chunks:
            raise ValueError("retriever needs at least one corpus chunk")
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 > 0 and 0 <= b <= 1")
        self._chunks = chunks
        self._k1 = k1
        self._b = b
        self._term_frequencies = [Counter(tokenize(chunk.text)) for chunk in chunks]
        self._lengths = [sum(frequencies.values()) for frequencies in self._term_frequencies]
        self._average_length = sum(self._lengths) / len(self._lengths)
        document_frequencies: Counter[str] = Counter()
        for frequencies in self._term_frequencies:
            document_frequencies.update(frequencies.keys())
        self._inverse_document_frequency = {
            term: math.log(1 + (len(chunks) - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequencies.items()
        }

    @classmethod
    def from_jsonl(cls, path: str | Path) -> BM25Retriever:
        chunks: list[_Chunk] = []
        for line_number, line in enumerate(
            Path(path).read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                row: Any = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on line {line_number} of {path}") from error
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number} of {path} must contain an object")
            chunk_id = row.get("chunk_id")
            text = row.get("text")
            source_url = row.get("source_url")
            if not all(
                isinstance(value, str) and value.strip() for value in (chunk_id, text, source_url)
            ):
                raise ValueError(
                    f"line {line_number} of {path} lacks chunk_id, text, or source_url"
                )
            assert isinstance(chunk_id, str)
            assert isinstance(text, str)
            assert isinstance(source_url, str)
            chunks.append(_Chunk(chunk_id=chunk_id, text=text, source_url=source_url))
        return cls(chunks)

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievalResult]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_terms = set(tokenize(query))
        scored: list[RetrievalResult] = []
        for chunk, frequencies, length in zip(
            self._chunks, self._term_frequencies, self._lengths, strict=True
        ):
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self._k1 * (
                    1 - self._b + self._b * length / self._average_length
                )
                score += (
                    self._inverse_document_frequency[term]
                    * frequency
                    * (self._k1 + 1)
                    / denominator
                )
            if score > 0:
                scored.append(
                    RetrievalResult(
                        chunk_id=chunk.chunk_id,
                        score=score,
                        text=chunk.text,
                        source_url=chunk.source_url,
                    )
                )
        return sorted(scored, key=lambda result: (-result.score, result.chunk_id))[:top_k]
