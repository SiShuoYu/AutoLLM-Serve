import json
from pathlib import Path

from citetune.quality import filter_corpus_for_authoring


def test_quality_filter_keeps_only_self_contained_chinese_source_chunks(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    rows = (
        {"chunk_id": "keep", "document_id": "doc-1", "text": "中" * 300},
        {"chunk_id": "short", "document_id": "doc-2", "text": "中" * 20},
        {"chunk_id": "english", "document_id": "doc-3", "text": "pod lifecycle " * 80},
        {"chunk_id": "template", "document_id": "doc-4", "text": "中" * 300 + "{{< x >}}" * 4},
    )
    corpus.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    output = tmp_path / "filtered.jsonl"
    manifest = filter_corpus_for_authoring(corpus, output, tmp_path / "manifest.json")
    kept = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["chunk_id"] for row in kept] == ["keep"]
    assert manifest.rejection_counts == {
        "low_chinese_ratio": 1,
        "too_many_template_markers": 1,
        "too_short": 1,
    }
