"""Guard against confusing benign recall with malignant sensitivity."""

import numpy as np
import pytest

from experiments.evaluate_baseline import evaluate_predictions


def test_all_benign_predictions_expose_missed_malignancies():
    results = evaluate_predictions([0, 0, 1, 1], [1, 1, 1, 1])
    assert results["per_class"]["malignant"]["recall"] == 0.0
    assert results["per_class"]["benign"]["recall"] == 1.0
    assert results["malignant_false_negative_count"] == 2
    assert results["malignant_false_negative_rate"] == 1.0
    assert results["confusion_matrix"]["counts"] == [[0, 2], [0, 2]]
    assert results["malignant_roc_auc"] is None


def test_perfect_predictions_and_malignant_auc():
    results = evaluate_predictions(
        np.array([0, 1, 0, 1]),
        np.array([0, 1, 0, 1]),
        malignant_probabilities=[0.9, 0.2, 0.8, 0.1],
    )
    assert results["accuracy"] == 1.0
    assert results["per_class"]["malignant"]["precision"] == 1.0
    assert results["per_class"]["malignant"]["recall"] == 1.0
    assert results["malignant_roc_auc"] == 1.0
    assert results["malignant_false_negative_rate"] == 0.0


def test_invalid_class_mapping_rejected():
    with pytest.raises(ValueError, match="Expected only dataset labels"):
        evaluate_predictions([0, 1, 2], [0, 1, 1])


def test_probability_length_mismatch_rejected():
    with pytest.raises(ValueError, match="match label length"):
        evaluate_predictions([0, 1], [0, 1], malignant_probabilities=[0.6])
