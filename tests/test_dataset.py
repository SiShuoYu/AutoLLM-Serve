from citetune.dataset import dataset_manifest, load_dataset


def test_sample_dataset_has_deterministic_manifest() -> None:
    examples = load_dataset("data/samples/grounded_qa.jsonl")
    manifest = dataset_manifest("data/samples/grounded_qa.jsonl", examples)
    assert manifest.example_count == 3
    assert manifest.split_counts == {"test": 3}
    assert len(manifest.sha256) == 64
