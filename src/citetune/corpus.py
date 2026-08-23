"""Pinned-source acquisition and deterministic Markdown corpus construction."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

KUBERNETES_REPOSITORY = "https://github.com/kubernetes/website.git"
KUBERNETES_DOCS_SUBDIR = Path("content/zh-cn/docs")
KUBERNETES_LICENSE = "CC-BY-4.0"
CONTENT_CLEANING_POLICY = "front-matter-and-html-comments-outside-fences-v2"


@dataclass(frozen=True, slots=True)
class CorpusBuildManifest:
    source_repository: str
    source_revision: str
    source_subdirectory: str
    license: str
    document_count: int
    chunk_count: int
    chunk_size_characters: int
    content_cleaning_policy: str
    corpus_sha256: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def fetch_kubernetes_docs(destination: str | Path, ref: str = "main") -> str:
    """Sparse-clone Chinese Kubernetes documentation and return its exact revision."""
    target = Path(destination)
    if target.exists() and any(target.iterdir()):
        raise ValueError(f"destination must be absent or empty: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--no-checkout",
            KUBERNETES_REPOSITORY,
            str(target),
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(target), "sparse-checkout", "init", "--cone"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(target),
            "sparse-checkout",
            "set",
            "--no-cone",
            str(KUBERNETES_DOCS_SUBDIR),
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(target), "fetch", "--depth", "1", "origin", ref], check=True)
    subprocess.run(["git", "-C", str(target), "checkout", "--detach", "FETCH_HEAD"], check=True)
    return _git_revision(target)


def _git_revision(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        capture_output=True,
        check=True,
        text=True,
    )
    return completed.stdout.strip()


def _strip_front_matter(text: str) -> str:
    if not text.startswith("---"):
        return text
    closing = re.search(r"^---\s*$", text[3:], flags=re.MULTILINE)
    return text[closing.end() + 3 :] if closing else text


def _strip_html_comments_outside_fences(text: str) -> str:
    """Remove hidden HTML comments while preserving literal examples in code fences."""
    result: list[str] = []
    prose: list[str] = []
    in_fence = False
    fence_character = ""

    def flush_prose() -> None:
        if prose:
            result.append(re.sub(r"<!--.*?(?:-->|\Z)", "", "".join(prose), flags=re.DOTALL))
            prose.clear()

    for line in text.splitlines(keepends=True):
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            flush_prose()
            current_character = marker.group(1)[0]
            if not in_fence:
                in_fence = True
                fence_character = current_character
            elif current_character == fence_character:
                in_fence = False
                fence_character = ""
            result.append(line)
        elif in_fence:
            result.append(line)
        else:
            prose.append(line)
    flush_prose()
    return "".join(result)


def _chunk_markdown(text: str, maximum_characters: int) -> list[str]:
    """Split Markdown by paragraphs without inventing or rewriting source text."""
    cleaned = _strip_html_comments_outside_fences(_strip_front_matter(text))
    blocks = [block.strip() for block in re.split(r"\n\s*\n", cleaned) if block.strip()]
    chunks: list[str] = []
    current = ""
    for block in blocks:
        if len(block) > maximum_characters:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(
                block[start : start + maximum_characters]
                for start in range(0, len(block), maximum_characters)
            )
        elif current and len(current) + 2 + len(block) > maximum_characters:
            chunks.append(current)
            current = block
        else:
            current = f"{current}\n\n{block}" if current else block
    if current:
        chunks.append(current)
    return chunks


def build_kubernetes_corpus(
    source_root: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
    *,
    revision: str | None = None,
    chunk_size_characters: int = 1_200,
) -> CorpusBuildManifest:
    """Build JSONL chunks with source hashes and immutable source references."""
    if chunk_size_characters < 200:
        raise ValueError("chunk_size_characters must be at least 200")
    root = Path(source_root)
    docs_root = root / KUBERNETES_DOCS_SUBDIR
    if not docs_root.is_dir():
        raise ValueError(f"Kubernetes Chinese documentation not found at {docs_root}")
    source_revision = revision or _git_revision(root)
    files = sorted(path for path in docs_root.rglob("*.md") if path.is_file())
    if not files:
        raise ValueError("no Markdown documentation files found")
    chunks: list[dict[str, object]] = []
    for document_path in files:
        relative_path = document_path.relative_to(root).as_posix()
        document_text = document_path.read_text(encoding="utf-8")
        document_sha256 = hashlib.sha256(document_text.encode("utf-8")).hexdigest()
        document_id = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:16]
        for chunk_index, text in enumerate(_chunk_markdown(document_text, chunk_size_characters)):
            chunks.append(
                {
                    "chunk_id": f"{document_id}-{chunk_index:04d}",
                    "document_id": document_id,
                    "source_path": relative_path,
                    "source_url": (
                        "https://github.com/kubernetes/website/blob/"
                        f"{source_revision}/{relative_path}"
                    ),
                    "source_revision": source_revision,
                    "license": KUBERNETES_LICENSE,
                    "document_sha256": document_sha256,
                    "text": text,
                }
            )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False, sort_keys=True) + "\n")
    corpus_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = CorpusBuildManifest(
        source_repository=KUBERNETES_REPOSITORY,
        source_revision=source_revision,
        source_subdirectory=KUBERNETES_DOCS_SUBDIR.as_posix(),
        license=KUBERNETES_LICENSE,
        document_count=len(files),
        chunk_count=len(chunks),
        chunk_size_characters=chunk_size_characters,
        content_cleaning_policy=CONTENT_CLEANING_POLICY,
        corpus_sha256=corpus_sha256,
    )
    manifest_file = Path(manifest_path)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(
        json.dumps(manifest.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
