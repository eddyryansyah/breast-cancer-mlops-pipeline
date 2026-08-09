import json

import pandas as pd

from monitoring import inference
from monitoring import prometheus_exporter as exporter


def test_exporter_load_sample_removes_target_and_updates_cursor(
    tmp_path,
    monkeypatch,
):
    dataset_path = tmp_path / "dataset.csv"

    df = pd.DataFrame(
        {
            "feature_1": [1, 2, 3, 4, 5],
            "feature_2": [10, 20, 30, 40, 50],
            "target": [0, 1, 0, 1, 0],
        }
    )
    df.to_csv(dataset_path, index=False)

    monkeypatch.setattr(exporter, "DATASET_PATH", str(dataset_path))
    monkeypatch.setattr(exporter, "SAMPLE_LIMIT", 2)
    monkeypatch.setattr(exporter, "_dataset_cache", None)
    monkeypatch.setattr(exporter, "_batch_cursor", 0)

    sample = exporter.load_sample()

    assert sample.shape == (2, 2)
    assert "target" not in sample.columns
    assert exporter._batch_cursor == 2


def test_exporter_load_sample_wraps_around(monkeypatch):
    cached_df = pd.DataFrame(
        {
            "feature_1": [0, 1, 2, 3, 4],
        }
    )

    monkeypatch.setattr(exporter, "_dataset_cache", cached_df)
    monkeypatch.setattr(exporter, "_batch_cursor", 4)
    monkeypatch.setattr(exporter, "SAMPLE_LIMIT", 3)

    sample = exporter.load_sample()

    assert sample["feature_1"].tolist() == [4, 0, 1]
    assert exporter._batch_cursor == 2


def test_inference_load_sample_respects_limit_and_removes_target(tmp_path):
    dataset_path = tmp_path / "dataset.csv"

    df = pd.DataFrame(
        {
            "feature_1": [1, 2, 3, 4],
            "feature_2": [10, 20, 30, 40],
            "target": [0, 1, 0, 1],
        }
    )
    df.to_csv(dataset_path, index=False)

    sample = inference.load_sample(str(dataset_path), limit=2)

    assert sample.shape == (2, 2)
    assert "target" not in sample.columns
    assert sample["feature_1"].tolist() == [1, 2]


def test_inference_send_prediction_request_uses_dataframe_split(monkeypatch):
    captured = {}

    class FakeResponse:
        status = 200

        def read(self):
            return b'{"predictions": [0, 1]}'

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(inference.urllib.request, "urlopen", fake_urlopen)

    sample = pd.DataFrame(
        {
            "feature_1": [1.0, 2.0],
            "feature_2": [3.0, 4.0],
        }
    )

    status, response_body = inference.send_prediction_request(
        "http://example.test/invocations",
        sample,
    )

    payload = json.loads(captured["request"].data.decode("utf-8"))

    assert status == 200
    assert response_body == '{"predictions": [0, 1]}'
    assert captured["timeout"] == 30
    assert captured["request"].get_method() == "POST"

    assert payload == {
        "dataframe_split": {
            "columns": ["feature_1", "feature_2"],
            "data": [
                [1.0, 3.0],
                [2.0, 4.0],
            ],
        }
    }
