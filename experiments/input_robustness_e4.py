"""E4: fixed-model, paired Gaussian perturbation sensitivity on held-out test rows.

Run from the repository root after validating the canonical E1 model:
    .venv/bin/python -m experiments.input_robustness_e4

This is synthetic input sensitivity, not real-world drift or clinical validation.
The existing model is loaded without fitting, and no source data are changed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from numbers import Integral
import os
from pathlib import Path
import platform
import statistics
import sys
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/processed/breast_cancer_preprocessing.csv"
E1_REPORT = ROOT / "results/reproducibility_e1_verified.json"
DEFAULT_OUTPUT = ROOT / "results/input_robustness_e4_verified.json"
E1_RUN_ID = "e89cf131e5f7419fb373212c29b0fba0"
E1_MODEL_SHA256 = "9c8eae1371e50141ee21cd46c5a2d515b9edbf1fded97a94d375d58a73050265"
SEEDS = tuple(range(4200, 4210))
ALPHAS = (0.05, 0.10, 0.20)
EXPECTED_CONFUSION = [[39, 3], [2, 70]]
METRIC_FIELDS = (
    "accuracy", "malignant_recall", "benign_recall", "prediction_flip_rate",
    "prediction_flip_count", "malignant_to_benign_errors", "benign_to_malignant_errors",
    "new_malignant_false_negatives",
)
DIAGNOSTIC_FIELDS = (
    "negative_feature_cells", "outside_training_range_feature_cells",
    "rows_with_outside_training_range_features",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path: Path, root: Path | None = None) -> str:
    """Forbid embedding machine-local absolute paths in the public report."""
    if root is None:
        root = ROOT
    return path.resolve().relative_to(root.resolve()).as_posix()


def checked_binary_labels(values, expected_count: int) -> np.ndarray:
    """Accept only one-dimensional integer class labels 0 or 1 (never bool)."""
    values = np.asarray(values, dtype=object)  # Do not coerce mixed bool/int into pure ints.
    if values.ndim != 1 or len(values) != expected_count:
        raise ValueError("Expected a one-dimensional prediction/label array of the required length")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) or v not in (0, 1) for v in values):
        raise ValueError("Class labels must be integer 0 or 1, not float/bool")
    return values.astype(np.int64, copy=False)


def evaluate_predictions(y_true, predicted, baseline_predicted) -> dict:
    """Explicit class semantics and per-row flips against the frozen baseline."""
    truth = checked_binary_labels(y_true, len(y_true))
    observed = checked_binary_labels(predicted, len(truth))
    baseline = checked_binary_labels(baseline_predicted, len(truth))
    matrix = confusion_matrix(truth, observed, labels=[0, 1]).tolist()
    correct = int(np.count_nonzero(truth == observed))
    malignant = int(np.count_nonzero(truth == 0))
    benign = int(np.count_nonzero(truth == 1))
    if malignant == 0 or benign == 0:
        raise ValueError("Held-out evaluation must contain both classes")
    flips = int(np.count_nonzero(observed != baseline))
    return {
        "accuracy": correct / len(truth),
        "malignant_recall": matrix[0][0] / malignant,
        "benign_recall": matrix[1][1] / benign,
        "confusion_matrix_labels_0_1": matrix,
        "malignant_to_benign_errors": matrix[0][1],
        "benign_to_malignant_errors": matrix[1][0],
        "prediction_flip_count": flips,
        "prediction_flip_rate": flips / len(truth),
        "new_malignant_false_negatives": int(np.count_nonzero(
            (truth == 0) & (baseline == 0) & (observed == 1)
        )),
    }


def validate_split(dataset: pd.DataFrame):
    """Match the canonical 569-row / 30-feature stratified split exactly."""
    if dataset.shape != (569, 31) or "target" not in dataset.columns:
        raise ValueError("Expected 569 dataset rows, 30 features and target")
    if dataset.columns.duplicated().any() or dataset.duplicated().any():
        raise ValueError("Unexpected duplicate columns or rows")
    X = dataset.drop(columns=["target"])
    y = checked_binary_labels(dataset["target"].to_numpy(), len(dataset))
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
        raise ValueError("Every input feature must be numeric")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Input features contain missing/nonfinite data")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    if len(X_train) != 455 or len(X_test) != 114:
        raise AssertionError("Unexpected train/test partition sizes")
    if set(X_train.index) & set(X_test.index):
        raise AssertionError("Train/test rows overlap")
    if (np.bincount(y_train, minlength=2).tolist() != [170, 285]
            or np.bincount(y_test, minlength=2).tolist() != [42, 72]):
        raise AssertionError("Unexpected class counts in stratified split")
    return X_train, X_test, y_train, y_test


def validate_training_scaler(model, X_train: pd.DataFrame) -> np.ndarray:
    """Confirm preprocessing learned only from training observations."""
    if not hasattr(model, "named_steps") or "scaler" not in model.named_steps:
        raise ValueError("Expected an MLflow sklearn Pipeline with a scaler step")
    scaler = model.named_steps["scaler"]
    if type(scaler).__name__ != "StandardScaler":
        raise ValueError("Expected sklearn StandardScaler in the frozen pipeline")
    values = X_train.to_numpy(dtype=float)
    means = values.mean(axis=0)
    std = values.std(axis=0, ddof=0)
    if not np.isfinite(means).all() or not np.isfinite(std).all() or not (std > 0).all():
        raise ValueError("Nonfinite or zero-variance training feature")
    if getattr(scaler, "n_features_in_", None) != len(X_train.columns):
        raise ValueError("Frozen scaler's feature count does not match the dataset")
    if hasattr(scaler, "feature_names_in_") and list(scaler.feature_names_in_) != list(X_train.columns):
        raise ValueError("Frozen scaler feature names/order do not match")
    np.testing.assert_allclose(scaler.mean_, means, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(scaler.scale_, std, rtol=1e-8, atol=1e-10)
    return std


def perturb_features(X_test: pd.DataFrame, train_std: np.ndarray, z: np.ndarray, alpha: float) -> pd.DataFrame:
    """Add unconstrained noise in original units, preserving schema and order."""
    values = X_test.to_numpy(dtype=float)
    std = np.asarray(train_std, dtype=float)
    noise = np.asarray(z, dtype=float)
    if not isinstance(alpha, (float, int)) or isinstance(alpha, bool) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Perturbation alpha must be a finite positive number")
    if std.shape != (values.shape[1],) or noise.shape != values.shape:
        raise ValueError("Noise / training standard deviation dimensions do not match inputs")
    if not np.isfinite(values).all() or not np.isfinite(std).all() or not np.isfinite(noise).all() or not (std > 0).all():
        raise ValueError("Perturbation inputs must be finite and standard deviations positive")
    noisy = values + alpha * std * noise
    if not np.isfinite(noisy).all():
        raise ValueError("Perturbation produced nonfinite values")
    return pd.DataFrame(noisy, index=X_test.index.copy(), columns=X_test.columns.copy())


def perturbation_diagnostics(noisy: pd.DataFrame, X_train: pd.DataFrame) -> dict:
    values = noisy.to_numpy(dtype=float)
    lower = X_train.min(axis=0).to_numpy(dtype=float)
    upper = X_train.max(axis=0).to_numpy(dtype=float)
    if list(noisy.columns) != list(X_train.columns):
        raise ValueError("Feature columns/order differ between training and perturbed testing")
    if not np.isfinite(values).all():
        raise ValueError("Perturbed features must be finite")
    out_of_range = (values < lower) | (values > upper)
    return {
        "negative_feature_cells": int(np.count_nonzero(values < 0)),
        "outside_training_range_feature_cells": int(np.count_nonzero(out_of_range)),
        "rows_with_outside_training_range_features": int(np.count_nonzero(out_of_range.any(axis=1))),
    }


def descriptive_stats(values) -> dict:
    seq = [float(v) for v in values]
    if not seq or not all(math.isfinite(v) for v in seq):
        raise ValueError("Statistics require nonempty finite numeric observations")
    return {
        "n": len(seq),
        "mean": statistics.mean(seq),
        "median": statistics.median(seq),
        "population_std": statistics.pstdev(seq),
        "min": min(seq),
        "max": max(seq),
    }


def summarize_conditions(conditions: list[dict], alphas=ALPHAS, seeds=SEEDS) -> list[dict]:
    """Keep per-seed evidence and independently recomputable level summaries."""
    if len(conditions) != len(alphas) * len(seeds):
        raise ValueError("Missing/extra perturbation conditions")
    results = []
    for alpha in alphas:
        rows = [row for row in conditions if row["alpha"] == alpha]
        if len(rows) != len(seeds) or sorted(row["seed"] for row in rows) != sorted(seeds):
            raise ValueError("Perturbation level is missing a seed or repeats a seed")
        results.append({
            "alpha": alpha,
            "seeds": list(seeds),
            "metrics": {key: descriptive_stats(row["metrics"][key] for row in rows) for key in METRIC_FIELDS},
            "diagnostics": {key: descriptive_stats(row["diagnostics"][key] for row in rows) for key in DIAGNOSTIC_FIELDS},
        })
    return results


def write_json_atomic(path: Path, data: dict) -> None:
    """Write an auditable report without truncating it if interrupted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temp_name = handle.name
            json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def run_experiment(e1_report_path: Path, dataset_path: Path, output: Path) -> dict:
    """Run preregistered E4 (1 baseline + 30 paired perturbations)."""
    from experiments.inference_latency_e2 import verify_e1_reference
    import mlflow
    import mlflow.sklearn
    import sklearn

    # Reject paths outside the repository before copying any provenance to output.
    relative_path(e1_report_path)
    relative_path(dataset_path)
    relative_path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing E4 evidence: {output}")
    if output.resolve() == e1_report_path.resolve() or output.resolve() == dataset_path.resolve():
        raise ValueError("Report output must not replace source evidence")

    e1_report = json.loads(e1_report_path.read_text(encoding="utf-8"))
    model_path, e1_run = verify_e1_reference(e1_report, ROOT, dataset_path)
    if e1_run["mlflow_run_id"] != E1_RUN_ID or e1_run["model_pickle_sha256"] != E1_MODEL_SHA256:
        raise ValueError("E1 reference differs from the preregistered canonical run/hash")
    dataset = pd.read_csv(dataset_path)
    X_train, X_test, y_train, y_test = validate_split(dataset)
    model = mlflow.sklearn.load_model(str(model_path))
    train_std = validate_training_scaler(model, X_train)
    baseline_predictions = checked_binary_labels(model.predict(X_test), len(X_test))
    baseline_metrics = evaluate_predictions(y_test, baseline_predictions, baseline_predictions)
    if baseline_metrics["confusion_matrix_labels_0_1"] != EXPECTED_CONFUSION:
        raise AssertionError("Frozen model baseline confusion matrix does not match preregistration")
    if baseline_metrics["prediction_flip_count"] != 0:
        raise AssertionError("Baseline must have zero prediction flips")

    report = {
        "protocol": "E4-fixed-E1-model-training-scaled-Gaussian-paired-noise-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Single held-out split; synthetic unconstrained Gaussian noise, not clinical validation or real-world drift",
        "completed": False,
        "reference": {
            "e1_run_id": E1_RUN_ID,
            "model_path": relative_path(model_path),
            "model_pickle_sha256": E1_MODEL_SHA256,
            "dataset_path": relative_path(dataset_path),
            "dataset_sha256": sha256_file(dataset_path),
            "e1_report_path": relative_path(e1_report_path),
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__, "mlflow": mlflow.__version__,
            "platform": platform.platform(),
        },
        "split": {
            "test_size": 0.2, "random_state": 42, "stratify": True,
            "train_rows": len(X_train), "test_rows": len(X_test),
            "feature_count": X_train.shape[1],
            "class_labels": {"0": "malignant", "1": "benign"},
            "train_class_counts": {"0": int(np.count_nonzero(y_train == 0)), "1": int(np.count_nonzero(y_train == 1))},
            "test_class_counts": {"0": int(np.count_nonzero(y_test == 0)), "1": int(np.count_nonzero(y_test == 1))},
            "test_row_indices": [int(v) for v in X_test.index],
        },
        "settings": {
            "alpha_levels": [0.0, *ALPHAS], "seeds": list(SEEDS),
            "noise_generator": "numpy.random.default_rng(seed).standard_normal((114,30))",
            "pairing": "same seed matrix Z reused for alpha 0.05, 0.10, and 0.20",
            "train_std_ddof": 0,
            "feature_units": "original numeric units (unscaled; no clipping)",
            "noise_generation": "independent standard normal cells per seed",
            "evaluation": "1 baseline plus 3 nonzero alphas x 10 paired seeds",
            "model_retraining": False,
        },
        "baseline": {
            "y_true": [int(v) for v in y_test],
            "predictions": [int(v) for v in baseline_predictions],
            "metrics": baseline_metrics,
        },
        "conditions": [],
        "summary": [],
    }
    write_json_atomic(output, report)
    for seed in SEEDS:
        z = np.random.default_rng(seed).standard_normal(X_test.shape)
        for alpha in ALPHAS:
            noisy = perturb_features(X_test, train_std, z, alpha)
            preds = checked_binary_labels(model.predict(noisy), len(X_test))
            row = {
                "seed": seed, "alpha": alpha,
                "predictions": [int(v) for v in preds],
                "metrics": evaluate_predictions(y_test, preds, baseline_predictions),
                "diagnostics": perturbation_diagnostics(noisy, X_train),
            }
            report["conditions"].append(row)
            write_json_atomic(output, report)
            print(f"[E4] seed={seed}, alpha={alpha:.2f}: "
                  f"accuracy={row['metrics']['accuracy']:.6f}, "
                  f"malignant_recall={row['metrics']['malignant_recall']:.6f}, "
                  f"flips={row['metrics']['prediction_flip_count']}", flush=True)
    report["summary"] = summarize_conditions(report["conditions"])
    report["completed"] = True
    write_json_atomic(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e1-report", type=Path, default=E1_REPORT)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run_experiment(args.e1_report.resolve(), args.dataset.resolve(), args.output.resolve())
    print("E4 report:", args.output)
    print(json.dumps({
        "completed": report["completed"],
        "e1_run_id": report["reference"]["e1_run_id"],
        "conditions": len(report["conditions"]),
        "baseline_accuracy": report["baseline"]["metrics"]["accuracy"],
        "summaries": [{"alpha": s["alpha"],
                       "accuracy_mean": s["metrics"]["accuracy"]["mean"],
                       "malignant_recall_mean": s["metrics"]["malignant_recall"]["mean"],
                       "flip_rate_mean": s["metrics"]["prediction_flip_rate"]["mean"]}
                      for s in report["summary"]],
    }, indent=2))


if __name__ == "__main__":
    main()
