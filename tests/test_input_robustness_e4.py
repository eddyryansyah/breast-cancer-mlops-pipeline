"""Unit and isolated integration tests for preregistered E4 synthetic robustness."""

import hashlib
import json
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from experiments import input_robustness_e4 as e4


@pytest.fixture
def small_frames():
    train = pd.DataFrame({"f0": [1.0, 3.0, 5.0], "f1": [4.0, 8.0, 12.0]}, index=[0, 1, 2])
    test = pd.DataFrame({"f0": [2.0, 4.0], "f1": [6.0, 10.0]}, index=[7, 9])
    return train, test


def test_checked_labels_accepts_python_and_numpy_ints():
    assert e4.checked_binary_labels([0, np.int64(1), 0], 3).tolist() == [0, 1, 0]


@pytest.mark.parametrize("bad", [[True, 0], [0.0, 1.0], [0, 2], ["0", "1"],
                                 [[0], [1]], [0], [np.nan, 1]])
def test_checked_labels_reject_invalid(bad):
    with pytest.raises(ValueError):
        e4.checked_binary_labels(bad, 2)


def test_metrics_explicit_class_order_and_baseline():
    y = [0, 0, 1, 1]
    p = [0, 1, 0, 1]
    m = e4.evaluate_predictions(y, p, p)
    assert m["confusion_matrix_labels_0_1"] == [[1, 1], [1, 1]]
    assert m["accuracy"] == 0.5
    assert m["malignant_recall"] == m["benign_recall"] == 0.5
    assert m["malignant_to_benign_errors"] == 1
    assert m["benign_to_malignant_errors"] == 1
    assert m["prediction_flip_count"] == 0
    assert m["new_malignant_false_negatives"] == 0


def test_metrics_counts_new_malignant_error_separately_from_flips():
    y = [0, 0, 1, 1]
    baseline = [0, 1, 0, 1]
    p = [1, 0, 1, 1]
    m = e4.evaluate_predictions(y, p, baseline)
    assert m["prediction_flip_count"] == 3
    assert m["prediction_flip_rate"] == 0.75
    assert m["new_malignant_false_negatives"] == 1
    assert m["malignant_to_benign_errors"] == 1
    assert m["benign_to_malignant_errors"] == 0
    assert m["accuracy"] == 0.75


def test_metrics_reject_single_class():
    with pytest.raises(ValueError, match="both classes"):
        e4.evaluate_predictions([0, 0], [0, 0], [0, 0])


def test_perturbation_equation_and_preserves_schema(small_frames):
    train, test = small_frames
    z = np.array([[1.0, -2.0], [0.0, 3.0]])
    std = train.std(axis=0, ddof=0).to_numpy()
    noisy = e4.perturb_features(test, std, z, 0.1)
    np.testing.assert_allclose(noisy.to_numpy(), test.to_numpy() + 0.1 * std * z)
    assert noisy.index.equals(test.index)
    assert noisy.columns.equals(test.columns)
    pd.testing.assert_frame_equal(test, pd.DataFrame({"f0": [2.0, 4.0], "f1": [6.0, 10.0]}, index=[7, 9]))


@pytest.mark.parametrize("alpha", [-0.1, 0.0, float("inf"), float("nan"), True, "0.1"])
def test_perturbation_rejects_invalid_alpha(small_frames, alpha):
    train, test = small_frames
    with pytest.raises(ValueError, match="alpha"):
        e4.perturb_features(test, np.ones(2), np.ones((2, 2)), alpha)


@pytest.mark.parametrize("std,z", [([1.0], np.ones((2,2))), ([1.0,1.0], np.ones((3,2))),
                                         ([1.0,0.0], np.ones((2,2))), ([float('nan'),1.0], np.ones((2,2))),
                                         ([1.0,1.0], np.array([[1.0,float('inf')],[1.0,1.0]]))])
def test_perturbation_rejects_bad_shape_or_values(small_frames, std, z):
    _, test = small_frames
    with pytest.raises(ValueError):
        e4.perturb_features(test, np.asarray(std), z, 0.1)


def test_perturbation_noise_is_paired_across_alpha(small_frames):
    _, test = small_frames
    std = np.array([2.0, 4.0])
    z = np.random.default_rng(4200).standard_normal(test.shape)
    noise_005 = e4.perturb_features(test, std, z, 0.05).to_numpy() - test.to_numpy()
    noise_020 = e4.perturb_features(test, std, z, 0.20).to_numpy() - test.to_numpy()
    np.testing.assert_allclose(noise_020, noise_005 * 4)


def test_perturbation_diagnostics_count_outliers(small_frames):
    train, test = small_frames
    noisy = pd.DataFrame({"f0": [-1.0, 4.0], "f1": [6.0, 15.0]}, index=test.index)
    diagnostics = e4.perturbation_diagnostics(noisy, train)
    assert diagnostics == {
        "negative_feature_cells": 1,
        "outside_training_range_feature_cells": 2,
        "rows_with_outside_training_range_features": 2,
    }


def test_perturbation_diagnostics_rejects_reordered_columns(small_frames):
    train, test = small_frames
    with pytest.raises(ValueError, match="columns/order"):
        e4.perturbation_diagnostics(test[list(reversed(test.columns))], train)


def test_stats_population_not_sample_variance():
    s = e4.descriptive_stats([1, 3])
    assert s == {"n": 2, "mean": 2, "median": 2, "population_std": 1, "min": 1, "max": 3}


@pytest.mark.parametrize("xs", [[], [1, float("nan")], [float("inf")]])
def test_stats_rejects_empty_or_nonfinite(xs):
    with pytest.raises(ValueError):
        e4.descriptive_stats(xs)


def make_condition(alpha, seed, flip_count=1):
    return {"alpha": alpha, "seed": seed,
            "metrics": {k: flip_count for k in e4.METRIC_FIELDS},
            "diagnostics": {k: seed for k in e4.DIAGNOSTIC_FIELDS}}


def test_summarize_conditions_uses_each_seed_once():
    rows = [make_condition(alpha, seed, seed-4199) for seed in (4200, 4201)
            for alpha in (0.05, 0.20)]
    s = e4.summarize_conditions(rows, alphas=(0.05, 0.20), seeds=(4200, 4201))
    assert len(s) == 2
    assert s[0]["metrics"]["accuracy"]["mean"] == 1.5
    assert s[0]["metrics"]["accuracy"]["population_std"] == 0.5
    assert s[0]["diagnostics"]["negative_feature_cells"]["n"] == 2


@pytest.mark.parametrize("alter", [lambda x: x[:-1], lambda x: x[:-1]+[x[0]]])
def test_summarize_conditions_rejects_missing_or_duplicated_seed(alter):
    rows = [make_condition(0.05, seed) for seed in (4200, 4201)]
    with pytest.raises(ValueError):
        e4.summarize_conditions(alter(rows), alphas=(0.05,), seeds=(4200, 4201))


def test_training_scaler_matches_training_only_stats(small_frames):
    train, _ = small_frames
    scaler = StandardScaler().fit(train)
    model = types.SimpleNamespace(named_steps={"scaler": scaler})
    np.testing.assert_allclose(e4.validate_training_scaler(model, train), train.std(ddof=0).to_numpy())


def test_scaler_rejects_mismatch_with_training_stats(small_frames):
    train, test = small_frames
    scaler = StandardScaler().fit(test)
    model = types.SimpleNamespace(named_steps={"scaler": scaler})
    with pytest.raises(AssertionError):
        e4.validate_training_scaler(model, train)


def test_scaler_rejects_other_preprocessor(small_frames):
    train, _ = small_frames
    model = types.SimpleNamespace(named_steps={"scaler": object()})
    with pytest.raises(ValueError, match="StandardScaler"):
        e4.validate_training_scaler(model, train)


def test_scaler_rejects_zero_variance():
    train = pd.DataFrame({"f0": [1.0,1.0,1.0], "f1": [3.0,4.0,5.0]})
    model = types.SimpleNamespace(named_steps={"scaler": StandardScaler().fit(train)})
    with pytest.raises(ValueError, match="zero-variance"):
        e4.validate_training_scaler(model, train)


def test_split_reproduces_expected_class_counts():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(569, 30))
    dataset = pd.DataFrame(X, columns=[f"f{i}" for i in range(30)])
    dataset["target"] = np.array([0]*212+[1]*357, dtype=int)
    train, test, y_train, y_test = e4.validate_split(dataset)
    assert (train.shape, test.shape) == ((455, 30), (114, 30))
    assert np.bincount(y_train, minlength=2).tolist() == [170,285]
    assert np.bincount(y_test, minlength=2).tolist() == [42,72]
    assert set(train.index).isdisjoint(set(test.index))


@pytest.mark.parametrize("mutator", [
    lambda d: d.drop(columns=["f0"]),
    lambda d: d.assign(f0=np.nan),
    lambda d: d.assign(f0="bad"),
    lambda d: d.assign(target=3),
])
def test_split_rejects_bad_dataset(mutator):
    rng = np.random.default_rng(7)
    dataset = pd.DataFrame(rng.normal(size=(569, 30)), columns=[f"f{i}" for i in range(30)])
    dataset["target"] = [0]*212+[1]*357
    with pytest.raises((ValueError, AssertionError)):
        e4.validate_split(mutator(dataset))


def test_atomic_json_report_and_sha(tmp_path):
    p = tmp_path / "sub" / "report.json"
    e4.write_json_atomic(p, {"a": [1,2], "b": True})
    assert json.loads(p.read_text()) == {"a": [1,2], "b": True}
    assert e4.sha256_file(p) == hashlib.sha256(p.read_bytes()).hexdigest()
    assert not list(p.parent.glob("*.tmp"))


def test_atomic_write_disallows_nonfinite_json(tmp_path):
    p = tmp_path / "out.json"
    with pytest.raises(ValueError):
        e4.write_json_atomic(p, {"bad": float("nan")})
    assert not p.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_relative_path_disallows_outside_repository(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    assert e4.relative_path(root / "docs" / "protocol.md", root) == "docs/protocol.md"
    with pytest.raises(ValueError):
        e4.relative_path(tmp_path / "outside.json", root)


def test_frozen_e4_end_to_end_with_mock_artifact(tmp_path, monkeypatch):
    """Exercise all 30 observations and report structure without actual MLflow/artifacts."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(size=(569, 30)), columns=[f"f{i}" for i in range(30)])
    df["target"] = [0]*212 + [1]*357
    dataset = tmp_path / "data" / "processed" / "breast_cancer_preprocessing.csv"
    dataset.parent.mkdir(parents=True)
    df.to_csv(dataset, index=False)
    e1 = tmp_path / "results" / "reproducibility_e1_verified.json"
    e1.parent.mkdir()
    e1.write_text('{"runs": []}')
    model_dir = tmp_path / ".research_artifacts" / "e1" / "model"
    model_dir.mkdir(parents=True)

    X_train, X_test, _, y_test = e4.validate_split(pd.read_csv(dataset))
    fitted_scaler = StandardScaler().fit(X_train)
    baseline = y_test.copy()
    baseline[np.where(y_test == 0)[0][:3]] = 1
    baseline[np.where(y_test == 1)[0][:2]] = 0
    reference_preds = dict(zip(X_test.index.tolist(), baseline.tolist()))

    class FakeModel:
        named_steps = {"scaler": fitted_scaler}

        def predict(self, frame):
            return np.array([reference_preds[index] for index in frame.index], dtype=np.int64)

    fake_e2 = types.ModuleType("experiments.inference_latency_e2")
    fake_e2.verify_e1_reference = lambda *_: (model_dir, {
        "mlflow_run_id": e4.E1_RUN_ID,
        "model_pickle_sha256": e4.E1_MODEL_SHA256,
    })
    fake_mlflow = types.ModuleType("mlflow")
    fake_mlflow.__path__ = []
    fake_mlflow.__version__ = "2.19.0"
    fake_mlflow_sklearn = types.ModuleType("mlflow.sklearn")
    fake_mlflow_sklearn.load_model = lambda *_: FakeModel()
    fake_mlflow.sklearn = fake_mlflow_sklearn
    monkeypatch.setitem(sys.modules, "experiments.inference_latency_e2", fake_e2)
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setitem(sys.modules, "mlflow.sklearn", fake_mlflow_sklearn)
    monkeypatch.setattr(e4, "ROOT", tmp_path)
    output = tmp_path / "results" / "input_robustness_e4_verified.json"
    report = e4.run_experiment(e1, dataset, output)
    assert report["completed"] is True
    assert len(report["conditions"]) == 30
    assert len(report["summary"]) == 3
    assert report["baseline"]["metrics"]["confusion_matrix_labels_0_1"] == [[39,3],[2,70]]
    assert json.loads(output.read_text()) == report
    assert sorted({c["seed"] for c in report["conditions"]}) == list(e4.SEEDS)
    assert sorted({c["alpha"] for c in report["conditions"]}) == list(e4.ALPHAS)
    assert all(c["metrics"]["prediction_flip_count"] == 0 for c in report["conditions"])
    assert all(len(c["predictions"]) == 114 for c in report["conditions"])
    assert report["reference"]["model_path"] == ".research_artifacts/e1/model"
    with pytest.raises(FileExistsError):
        e4.run_experiment(e1, dataset, output)
