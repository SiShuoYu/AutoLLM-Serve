"""Leakage-resistant, deterministic document-level corpus splits."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SPLITS = ("train", "validation", "test")


@dataclass(frozen=True, slots=True)
class SplitManifest:
    input_sha256: str
    seed: int
    document_counts: dict[str, int]
    chunk_counts: dict[str, int]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _assignment(document_id: str, seed: int) -> str:
    bucket = int(hashlib.sha256(f"{seed}:{document_id}".encode()).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def split_corpus_by_document(
    corpus_path: str | Path,
    output_directory: str | Path,
    manifest_path: str | Path,
    *,
    seed: int = 42,
) -> SplitManifest:
    """Assign whole documents to train/validation/test, never individual chunks."""
    corpus = Path(corpus_path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(corpus.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid corpus JSON at line {line_number}") from error
        if not isinstance(row, dict) or not isinstance(row.get("document_id"), str):
            raise ValueError(f"corpus line {line_number} has no string document_id")
        rows.append(row)
    if not rows:
        raise ValueError("corpus has no rows")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["document_id"]].append(row)
    assigned_documents: dict[str, set[str]] = {split: set() for split in SPLITS}
    assigned_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for document_id, document_rows in sorted(grouped.items()):
        split = _assignment(document_id, seed)
        assigned_documents[split].add(document_id)
        assigned_rows[split].extend(document_rows)
    for split in SPLITS:
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in assigned_rows[split]:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    overlaps = [
        first
        for index, first in enumerate(SPLITS)
        for second in SPLITS[index + 1 :]
        if assigned_documents[first] & assigned_documents[second]
    ]
    if overlaps:
        raise RuntimeError("document leakage detected across splits")
    manifest = SplitManifest(
        input_sha256=hashlib.sha256(corpus.read_bytes()).hexdigest(),
        seed=seed,
        document_counts={split: len(assigned_documents[split]) for split in SPLITS},
        chunk_counts={split: len(assigned_rows[split]) for split in SPLITS},
    )
    output_manifest = Path(manifest_path)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        json.dumps(manifest.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
