"""Fast unit tests of E1 evidence capture; full MLflow runs are opt-in."""

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from experiments.reproducibility_e1 import (
    METRIC_NAMES,
    prepare_workspace,
    sha256_file,
    sha256_labels,
    source_file_hashes,
    summarize,
    use_current_python_for_mlflow,
    write_report,
)


def make_source(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    model_source = repo / "mlproject"
    model_source.mkdir(parents=True)
    for name in ("MLproject", "modelling.py", "conda.yaml", "requirements.txt"):
        (model_source / name).write_text(f"source for {name}\n", encoding="utf-8")
    (model_source / "mlruns").mkdir()
    (model_source / "mlruns" / "stale.txt").write_text("stale", encoding="utf-8")
    (model_source / "model.joblib").write_text("stale artifact", encoding="utf-8")
    dataset = repo / "data/processed/breast_cancer_preprocessing.csv"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("feature,target\n1,0\n2,1\n", encoding="utf-8")
    return repo, dataset


def test_sha256_file_exact_bytes(tmp_path):
    sample = tmp_path / "sample.txt"
    sample.write_bytes(b"e1-data")
    assert sha256_file(sample) == hashlib.sha256(b"e1-data").hexdigest()


def test_label_hash_is_deterministic_and_order_sensitive():
    assert sha256_labels([0, 1, 1]) == sha256_labels([0, 1, 1])
    assert sha256_labels([0, 1, 1]) != sha256_labels([1, 1, 0])


def test_label_hash_rejects_unknown_classes():
    with pytest.raises(ValueError, match="0=malignant"):
        sha256_labels([0, 2])


def test_prepare_workspace_excludes_stale_model_and_mlflow_data(tmp_path):
    repo, dataset = make_source(tmp_path)
    workspace = tmp_path / "workspace"
    prepare_workspace(repo, workspace, dataset)
    assert (workspace / "data/processed/breast_cancer_preprocessing.csv").read_bytes() == dataset.read_bytes()
    assert (workspace / "mlproject/MLproject").is_file()
    assert not (workspace / "mlproject/mlruns").exists()
    assert not (workspace / "mlproject/model.joblib").exists()
    assert source_file_hashes(repo) == source_file_hashes(workspace)


def test_prepare_workspace_refuses_existing_workspace(tmp_path):
    repo, dataset = make_source(tmp_path)
    destination = tmp_path / "existing"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        prepare_workspace(repo, destination, dataset)


def test_summarize_all_identical_runs_and_metrics():
    records = [
        {"status": "FINISHED", "metrics": {key: 0.95 for key in METRIC_NAMES},
         "prediction_sha256": "same", "model_pickle_sha256": "same",
         "duration_seconds": seconds}
        for seconds in [1.0, 2.0, 3.0]
    ]
    report = summarize(records, repeats=3)
    assert report["all_successful"]
    assert report["prediction_hashes_identical"]
    assert report["model_pickle_hashes_identical"]
    assert report["duration_seconds"]["mean"] == 2.0
    assert report["metrics"]["accuracy"]["range"] == 0.0


def test_summarize_partial_failure_and_divergence():
    records = [
        {"status": "FINISHED", "metrics": {"accuracy": 0.91},
         "prediction_sha256": "abc", "model_pickle_sha256": "xyz", "duration_seconds": 2},
        {"status": "FINISHED", "metrics": {"accuracy": 0.92},
         "prediction_sha256": "def", "model_pickle_sha256": "xyz", "duration_seconds": 4},
        {"status": "FAILED", "duration_seconds": 1},
    ]
    report = summarize(records, repeats=3)
    assert not report["all_successful"]
    assert report["failed_runs"] == 1
    assert not report["prediction_hashes_identical"]
    assert not report["model_pickle_hashes_identical"]  # requires 3/3 successes
    assert report["metrics"]["accuracy"]["range"] == pytest.approx(0.01)


def test_atomic_report_write(tmp_path):
    destination = tmp_path / "results" / "e1.json"
    write_report({"value": [1, 2]}, destination)
    assert json.loads(destination.read_text()) == {"value": [1, 2]}
    assert not destination.with_name("e1.json.tmp").exists()


def test_mlflow_subprocess_python_matches_current_interpreter(monkeypatch):
    # Intentionally simulate calling .venv/bin/python without activating .venv.
    monkeypatch.setenv("PATH", "/usr/bin")
    with use_current_python_for_mlflow() as subprocess_python:
        assert Path(subprocess_python).resolve() == Path(sys.executable).resolve()
        assert os.environ["PATH"].split(os.pathsep)[0] == str(Path(sys.executable).absolute().parent)
    assert os.environ["PATH"] == "/usr/bin"


def test_mlflow_python_path_restored_when_originally_absent(monkeypatch):
    monkeypatch.delenv("PATH", raising=False)
    with use_current_python_for_mlflow() as subprocess_python:
        assert Path(subprocess_python).resolve() == Path(sys.executable).resolve()
    assert "PATH" not in os.environ
