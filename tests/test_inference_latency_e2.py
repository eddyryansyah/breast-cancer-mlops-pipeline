"""Fast E2 benchmark unit tests; no model training, Docker, or HTTP required."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.inference_latency_e2 import (
    balanced_samples,
    benchmark_pair,
    inspect_container,
    latency_stats,
    percentile_linear,
    validate_labels,
    verify_e1_reference,
    write_json_atomic,
)


def sample_dataset():
    return pd.DataFrame({
        "f1": list(range(64)),
        "f2": [float(x) for x in range(64)],
        "target": [0] * 32 + [1] * 32,
    })


@pytest.mark.parametrize("size,expected_indices", [
    (1, [0]),
    (8, [0, 32, 1, 33, 2, 34, 3, 35]),
    (32, [i for pair in zip(range(16), range(32, 48)) for i in pair]),
])
def test_balanced_sample_excludes_target_and_interleaves_classes(size, expected_indices):
    dataset = sample_dataset()
    result = balanced_samples(dataset, size)
    assert result.index.tolist() == expected_indices
    assert "target" not in result.columns
    assert len(result) == size


def test_balanced_sample_rejects_missing_class_and_too_large_batch():
    dataset = sample_dataset()
    with pytest.raises(ValueError, match="Not enough"):
        balanced_samples(dataset[dataset["target"] == 0], 8)
    with pytest.raises(ValueError, match="Not enough"):
        balanced_samples(dataset, 100)


def test_validate_labels_accepts_numpy_integer_types():
    assert validate_labels(np.array([0, 1, 0], dtype=np.int64), 3) == [0, 1, 0]


@pytest.mark.parametrize("values", [[0, 2], [True, 1], [0.0, 1.0], [0], [0, -1]])
def test_validate_labels_rejects_invalid_classes(values):
    with pytest.raises(ValueError):
        validate_labels(values, 2)


def test_percentile_p95_linear_interpolation_and_stats():
    assert percentile_linear([1, 2, 3, 4, 5], 0.95) == pytest.approx(4.8)
    stats = latency_stats([1.0, 2.0, 3.0, 4.0, 5.0], 8)
    assert stats["median_ms"] == 3
    assert stats["p95_ms"] == pytest.approx(4.8)
    assert stats["n"] == 5
    assert stats["sequential_throughput_samples_per_second"] == pytest.approx(8000 / 3)


@pytest.mark.parametrize("values", [[], [0], [-1], [float("nan")], [float("inf")]])
def test_latency_stats_rejects_nonpositive_or_nonfinite(values):
    with pytest.raises(ValueError):
        latency_stats(values, 8)


def test_pair_measurement_alternates_and_excludes_warmup():
    events = []
    ticks = iter(range(0, 100_000_000, 1_000_000))

    def local():
        events.append("local")
        return [0, 1]

    def http():
        events.append("http")
        return '{"predictions": [0, 1]}'

    result = benchmark_pair(
        local, http, lambda x: x, lambda x: json.loads(x)["predictions"],
        [0, 1], warmup=2, iterations=3, clock_ns=lambda: next(ticks),
    )
    assert events == ["local", "http", "http", "local", "local", "http", "http", "local", "local", "http"]
    assert result == {"local": [1.0] * 3, "http": [1.0] * 3}


def test_pair_measurement_fails_on_prediction_mismatch():
    with pytest.raises(AssertionError, match="HTTP|http"):
        benchmark_pair(lambda: [0], lambda: [1], lambda x: x, lambda x: x,
                       [0], warmup=0, iterations=2)


def test_pair_measurement_rejects_zero_iterations():
    with pytest.raises(ValueError, match="iterations"):
        benchmark_pair(lambda: [0], lambda: [0], lambda x: x, lambda x: x,
                       [0], warmup=2, iterations=0)


def test_verify_e1_reference_checks_model_and_dataset_hashes(tmp_path):
    model = tmp_path / ".research_artifacts" / "e1" / "one" / "model"
    model.mkdir(parents=True)
    (model / "MLmodel").write_text("flavors: {}\n", encoding="utf-8")
    (model / "model.pkl").write_bytes(b"model")
    dataset = tmp_path / "dataset.csv"
    dataset.write_bytes(b"f1,target\n3,0\n")
    model_hash = hashlib.sha256(b"model").hexdigest()
    data_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
    report = {
        "summary": {"all_successful": True, "successful_runs": 5, "model_pickle_hashes_identical": True},
        "runs": [{"status": "FINISHED", "mlflow_run_id": str(i),
                  "model_artifact": str(model), "model_pickle_sha256": model_hash} for i in range(5)],
        "dataset": {"sha256": data_hash},
    }
    path, record = verify_e1_reference(report, tmp_path, dataset)
    assert path == model
    assert record["mlflow_run_id"] == "0"
    (model / "model.pkl").write_bytes(b"altered")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_e1_reference(report, tmp_path, dataset)


def test_inspect_container_binds_running_image_to_loopback_port(monkeypatch):
    payload = [{
        "State": {"Running": True},
        "Config": {"Image": "breast-cancer-e2:local"},
        "Image": "sha256:abcde",
        "NetworkSettings": {"Ports": {"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "5001"}]}},
    }]

    class Completed:
        stdout = json.dumps(payload)

    monkeypatch.setattr("subprocess.run", lambda *a, **kw: Completed())
    result = inspect_container("mlflow-e2-benchmark", "breast-cancer-e2:local",
                               "http://127.0.0.1:5001/invocations")
    assert result["image_id"] == "sha256:abcde"
    assert result["published_port"] == 5001
    with pytest.raises(ValueError, match="Unexpected image"):
        inspect_container("mlflow-e2-benchmark", "different-image:local",
                          "http://127.0.0.1:5001/invocations")
    with pytest.raises(ValueError, match="not bound"):
        inspect_container("mlflow-e2-benchmark", "breast-cancer-e2:local",
                          "http://127.0.0.1:5002/invocations")
    with pytest.raises(ValueError, match="127.0.0.1"):
        inspect_container("mlflow-e2-benchmark", "breast-cancer-e2:local",
                          "http://192.168.1.2:5001/invocations")


def test_atomic_json_file(tmp_path):
    destination = tmp_path / "out" / "e2.json"
    write_json_atomic(destination, {"completed": False})
    write_json_atomic(destination, {"completed": True})
    assert json.loads(destination.read_text()) == {"completed": True}
    assert not destination.with_name(destination.name + ".tmp").exists()
