import json
from pathlib import Path

from citetune.splits import split_corpus_by_document


def test_split_keeps_every_document_in_exactly_one_partition(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    with corpus.open("w", encoding="utf-8") as handle:
        for document_number in range(20):
            for chunk_number in range(2):
                handle.write(
                    json.dumps({"document_id": f"doc-{document_number}", "chunk_id": chunk_number})
                    + "\n"
                )
    output = tmp_path / "splits"
    manifest = split_corpus_by_document(corpus, output, tmp_path / "split-manifest.json", seed=7)
    seen: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        for line in (output / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            document_id = json.loads(line)["document_id"]
            assert document_id not in seen or seen[document_id] == split
            seen[document_id] = split
    assert sum(manifest.document_counts.values()) == 20
    assert sum(manifest.chunk_counts.values()) == 40
