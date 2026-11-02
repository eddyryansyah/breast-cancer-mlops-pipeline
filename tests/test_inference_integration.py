"""Integration tests using the real dataset, training pipeline, and MLflow serialization.

These do not start a Docker container. The separate serving smoke test in
GitHub Actions checks the real container's /ping and /invocations endpoints.
"""

from pathlib import Path

import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from mlflow.models import infer_signature

from mlproject.modelling import load_dataset, train_model
from monitoring.inference import load_sample


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed" / "breast_cancer_preprocessing.csv"


@pytest.fixture(scope="module")
def trained_and_serialized(tmp_path_factory):
    df = load_dataset(str(DATASET))
    model, _, X_test, _, _, _ = train_model(df)
    model_path = tmp_path_factory.mktemp("mlflow-integration") / "model"
    mlflow.sklearn.save_model(
        sk_model=model,
        path=str(model_path),
        signature=infer_signature(X_test, model.predict(X_test)),
        input_example=X_test.head(2),
    )
    assert (model_path / "MLmodel").is_file()
    return model, X_test, model_path


def test_sklearn_mlflow_round_trip_preserves_predictions(trained_and_serialized):
    original, X_test, model_path = trained_and_serialized
    loaded = mlflow.sklearn.load_model(str(model_path))
    batch = X_test.head(12)
    np.testing.assert_array_equal(original.predict(batch), loaded.predict(batch))
    np.testing.assert_allclose(
        original.predict_proba(batch), loaded.predict_proba(batch), rtol=0, atol=1e-12
    )
    assert list(loaded.classes_) == [0, 1]


def test_pyfunc_mlflow_predictions_match_sklearn_and_preserve_order(trained_and_serialized):
    original, X_test, model_path = trained_and_serialized
    pyfunc = mlflow.pyfunc.load_model(str(model_path))
    batch = X_test.head(12).iloc[::-1]
    actual = np.asarray(pyfunc.predict(batch)).reshape(-1)
    np.testing.assert_array_equal(original.predict(batch), actual)
    assert len(actual) == len(batch)


def test_model_signature_rejects_missing_feature(trained_and_serialized):
    _, X_test, model_path = trained_and_serialized
    pyfunc = mlflow.pyfunc.load_model(str(model_path))
    incomplete = X_test.head(2).drop(columns=[X_test.columns[0]])
    with pytest.raises(Exception, match="missing|Missing|required"):
        pyfunc.predict(incomplete)


def test_http_input_sample_retains_training_feature_order_and_excludes_target(
    trained_and_serialized,
):
    _, X_test, _ = trained_and_serialized
    sample = load_sample(str(DATASET), limit=8)
    assert "target" not in sample.columns
    assert sample.shape == (8, 30)
    assert sample.columns.tolist() == X_test.columns.tolist()
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in sample.dtypes)
