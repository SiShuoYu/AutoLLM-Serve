from pathlib import Path

from citetune.experiment import load_experiment_config, run_experiment


def test_run_persists_auditable_artifacts(tmp_path: Path) -> None:
    repository_root = Path.cwd()
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        f"""experiment:
  name: test-run
  seed: 7
  dataset_path: {repository_root / "data/samples/grounded_qa.jsonl"}
  predictions_path: {repository_root / "data/samples/baseline_predictions.jsonl"}
  output_dir: output
""",
        encoding="utf-8",
    )
    config = load_experiment_config(config_path)
    summary = run_experiment(config)
    assert summary.example_count == 3
    assert (tmp_path / "output" / "run.json").is_file()
    assert (tmp_path / "output" / "summary.json").is_file()
    assert (tmp_path / "output" / "per_example.jsonl").is_file()
