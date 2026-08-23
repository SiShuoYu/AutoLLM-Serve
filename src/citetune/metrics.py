"""Deterministic grounded-answer metrics with explicit limitations."""

from __future__ import annotations

from collections import Counter


def normalize_chinese_text(text: str) -> str:
    """Normalize whitespace and case; evaluation remains character based in V1."""
    return "".join(text.lower().split())


def exact_match(reference: str, prediction: str) -> float:
    return float(normalize_chinese_text(reference) == normalize_chinese_text(prediction))


def character_f1(reference: str, prediction: str) -> float:
    reference_tokens = list(normalize_chinese_text(reference))
    prediction_tokens = list(normalize_chinese_text(prediction))
    if not reference_tokens or not prediction_tokens:
        return 0.0
    overlap = sum((Counter(reference_tokens) & Counter(prediction_tokens)).values())
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def set_precision(expected: set[str], observed: set[str]) -> float | None:
    if not observed:
        return None
    return len(expected & observed) / len(observed)


def set_recall(expected: set[str], observed: set[str]) -> float:
    return len(expected & observed) / len(expected) if expected else 1.0
