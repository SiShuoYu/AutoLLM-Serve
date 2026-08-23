import json
from pathlib import Path

from citetune.corpus import KUBERNETES_DOCS_SUBDIR, build_kubernetes_corpus


def test_builds_deterministic_corpus_with_source_metadata(tmp_path: Path) -> None:
    root = tmp_path / "website"
    docs = root / KUBERNETES_DOCS_SUBDIR / "concepts"
    docs.mkdir(parents=True)
    (docs / "pods.md").write_text(
        "---\ntitle: Pods\n---\n\n# Pod\n\nPod 是 Kubernetes 中最小的可部署单元。\n\n"
        "## 生命周期\n\nPod 可以被创建和删除。\n",
        encoding="utf-8",
    )
    output = tmp_path / "chunks.jsonl"
    manifest_path = tmp_path / "manifest.json"
    manifest = build_kubernetes_corpus(
        root, output, manifest_path, revision="test-revision", chunk_size_characters=200
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert manifest.document_count == 1
    assert manifest.chunk_count == 1
    assert rows[0]["source_revision"] == "test-revision"
    assert rows[0]["license"] == "CC-BY-4.0"
    assert "title:" not in rows[0]["text"]
    assert manifest_path.is_file()
