"""Evaluate the v1.0.0 production model with explicit class-specific metrics.

Run from the repository root, after installing mlproject/requirements.txt:

    python -m experiments.evaluate_baseline \
        --dataset data/processed/breast_cancer_preprocessing.csv \
        --output results/model_evaluation.json

This uses the existing production train_model() and its original stratified
80/20 split. It does not tune or change the production model. The resulting
single-split estimate is a baseline, not a complete scientific evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

CLASS_NAMES = {0: "malignant", 1: "benign"}
LABEL_ORDER = [0, 1]


def evaluate_predictions(
    y_true: Any,
    y_pred: Any,
    malignant_probabilities: Any | None = None,
) -> dict[str, Any]:
    """Return explicit metrics; class 0 (malignant) is the target of interest.

    Confusion-matrix rows are *true* labels and columns are *predicted* labels,
    in the order [malignant (0), benign (1)].
    """
    true = np.asarray(y_true).ravel()
    pred = np.asarray(y_pred).ravel()

    if len(true) == 0 or len(true) != len(pred):
        raise ValueError("Nonempty true/predicted labels must have matching lengths.")
    if not set(np.unique(np.concatenate([true, pred]))).issubset(set(LABEL_ORDER)):
        raise ValueError("Expected only dataset labels 0 (malignant) and 1 (benign).")

    matrix = confusion_matrix(true, pred, labels=LABEL_ORDER)
    precision, recall, f1, support = precision_recall_fscore_support(
        true, pred, labels=LABEL_ORDER, zero_division=0
    )
    by_class = {
        CLASS_NAMES[label]: {
            "support": int(support[index]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1_score": float(f1[index]),
        }
        for index, label in enumerate(LABEL_ORDER)
    }

    malignant_count = int(matrix[0, :].sum())
    false_negative_count = int(matrix[0, 1])  # Truly malignant, predicted benign.
    malignant_auc = None
    if malignant_probabilities is not None:
        probabilities = np.asarray(malignant_probabilities, dtype=float).ravel()
        if len(probabilities) != len(true) or not np.isfinite(probabilities).all():
            raise ValueError("Malignant probabilities must be finite and match label length.")
        if ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError("Probabilities must lie between 0 and 1.")
        if len(np.unique(true)) == 2:
            # roc_auc_score treats 1 as positive; explicitly recode malignant to 1.
            malignant_auc = float(roc_auc_score((true == 0).astype(int), probabilities))

    return {
        "label_mapping": {"0": "malignant", "1": "benign"},
        "accuracy": float(accuracy_score(true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "per_class": by_class,
        "confusion_matrix": {
            "labels": [CLASS_NAMES[label] for label in LABEL_ORDER],
            "rows": "true labels",
            "columns": "predicted labels",
            "counts": matrix.tolist(),
        },
        "malignant_false_negative_count": false_negative_count,
        "malignant_false_negative_rate": (
            float(false_negative_count / malignant_count) if malignant_count else None
        ),
        "malignant_roc_auc": malignant_auc,
    }


def run_evaluation(dataset_path: Path, output_path: Path) -> dict[str, Any]:
    """Train exactly as the production v1.0.0 entrypoint, then report metrics."""
    from mlproject.modelling import load_dataset, train_model

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Processed dataset not found: {dataset_path}")
    dataset_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    dataframe = load_dataset(str(dataset_path))
    model, X_train, X_test, y_train, y_test, legacy_metrics = train_model(dataframe)

    class_values = list(model.classes_)
    if 0 not in class_values or 1 not in class_values:
        raise ValueError(f"Expected model classes [0, 1], got {class_values}")
    malignant_col = class_values.index(0)
    probabilities = model.predict_proba(X_test)[:, malignant_col]

    report = {
        "protocol": "v1.0.0-stratified-holdout-baseline",
        "warning": "Single stratified 80/20 split: not a cross-validation or clinical validation.",
        "dataset": {
            "path": str(dataset_path),
            "sha256": dataset_sha256,
            "rows": int(len(dataframe)),
            "feature_count": int(X_test.shape[1]),
        },
        "split": {
            "method": "stratified train_test_split",
            "test_size": 0.2,
            "random_state": 42,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "train_class_counts": {
                str(label): int((y_train == label).sum()) for label in LABEL_ORDER
            },
            "test_class_counts": {
                str(label): int((y_test == label).sum()) for label in LABEL_ORDER
            },
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "original_binary_metrics_positive_label": 1,
        "original_binary_metrics_positive_class": "benign",
        "original_binary_metrics": {
            name: float(value) for name, value in legacy_metrics.items()
        },
        "explicit_metrics": evaluate_predictions(
            y_test, model.predict(X_test), malignant_probabilities=probabilities
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/breast_cancer_preprocessing.csv"),
        help="Processed dataset, relative to repository root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/model_evaluation.json"),
        help="Machine-readable report location.",
    )
    args = parser.parse_args()
    report = run_evaluation(args.dataset, args.output)
    metrics = report["explicit_metrics"]
    print(f"Report written: {args.output}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Malignant recall (sensitivity): {metrics['per_class']['malignant']['recall']:.4f}")
    print(f"Benign recall: {metrics['per_class']['benign']['recall']:.4f}")
    print(f"Malignant false negatives: {metrics['malignant_false_negative_count']}")


if __name__ == "__main__":
    main()
