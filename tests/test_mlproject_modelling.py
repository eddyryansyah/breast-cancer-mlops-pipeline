import pandas as pd
import pytest

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlproject.modelling import load_dataset, train_model


def test_load_dataset_success(tmp_path):
    dataset_path = tmp_path / "dataset.csv"

    expected_df = pd.DataFrame(
        {
            "feature_1": [1.0, 2.0],
            "feature_2": [3.0, 4.0],
            "target": [0, 1],
        }
    )
    expected_df.to_csv(dataset_path, index=False)

    result = load_dataset(str(dataset_path))

    pd.testing.assert_frame_equal(result, expected_df)


def test_load_dataset_missing_file(tmp_path):
    dataset_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="Dataset tidak ditemukan"):
        load_dataset(str(dataset_path))


def test_train_model_returns_expected_pipeline_and_split():
    row_count = 40
    feature_count = 30

    data = {
        f"feature_{i}": [
            float(row + i)
            for row in range(row_count)
        ]
        for i in range(feature_count)
    }
    data["target"] = [0, 1] * (row_count // 2)

    df = pd.DataFrame(data)

    model_pipeline, X_train, X_test, y_train, y_test, metrics = train_model(df)

    assert isinstance(model_pipeline, Pipeline)
    assert isinstance(model_pipeline.named_steps["scaler"], StandardScaler)
    assert isinstance(
        model_pipeline.named_steps["model"],
        RandomForestClassifier,
    )

    model = model_pipeline.named_steps["model"]
    assert model.n_estimators == 200
    assert model.max_depth == 5
    assert model.min_samples_split == 2
    assert model.min_samples_leaf == 2
    assert model.random_state == 42

    assert X_train.shape == (32, 30)
    assert X_test.shape == (8, 30)
    assert len(y_train) == 32
    assert len(y_test) == 8

    assert y_train.value_counts().to_dict() == {0: 16, 1: 16}
    assert y_test.value_counts().to_dict() == {0: 4, 1: 4}

    assert set(metrics) == {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
    }

    for metric_value in metrics.values():
        assert 0.0 <= metric_value <= 1.0

    predictions = model_pipeline.predict(X_test)
    assert len(predictions) == len(X_test)
