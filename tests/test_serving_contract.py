"""Fast validation of the expected MLflow HTTP serving contract."""

import pytest

from scripts.verify_serving_endpoint import (
    assert_prediction_parity,
    health_url,
    parse_predictions,
)


def test_build_health_url_from_local_invocations_endpoint():
    assert health_url("http://127.0.0.1:5001/invocations") == "http://127.0.0.1:5001/ping"


def test_rejects_invalid_invocations_url():
    with pytest.raises(ValueError, match="/invocations"):
        health_url("http://127.0.0.1:5001/other")


def test_accepts_valid_mlflow_classification_response():
    assert parse_predictions('{"predictions": [0, 1, 0]}', 3) == [0, 1, 0]


@pytest.mark.parametrize("body", [
    "not-json", '[]', '{"outputs": [0, 1]}', '{"predictions": [0]}',
    '{"predictions": [0, 2]}', '{"predictions": [true, 1]}',
])
def test_rejects_malformed_or_invalid_serving_responses(body):
    with pytest.raises(ValueError):
        parse_predictions(body, 2)


def test_prediction_parity_accepts_exact_match():
    assert_prediction_parity([0, 1, 1], [0, 1, 1])


def test_prediction_parity_rejects_class_difference():
    with pytest.raises(AssertionError, match="row indices"):
        assert_prediction_parity([0, 1, 1], [0, 0, 1])
