"""Smoke-test real MLflow HTTP serving against its canonical local model.

Run from the repository root as ``python -m scripts.verify_serving_endpoint``.
This is an operational parity check, not an independent clinical evaluation.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def health_url(invocations_url: str) -> str:
    """Build the MLflow /ping URL from a complete /invocations URL."""
    parsed = urllib.parse.urlsplit(invocations_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("A complete HTTP(S) serving URL is required")
    if parsed.path.rstrip("/") != "/invocations":
        raise ValueError("Serving URL must end with /invocations")
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "/ping", "", ""))


def parse_predictions(body: str, expected_count: int) -> list[int]:
    """Validate the MLflow serving prediction response and target domain."""
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError("Serving returned invalid JSON") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("predictions"), list):
        raise ValueError("Response must contain a predictions array")
    values = payload["predictions"]
    if len(values) != expected_count:
        raise ValueError(f"Expected {expected_count} predictions, received {len(values)}")
    if any(type(value) is not int or value not in (0, 1) for value in values):
        raise ValueError("Predictions must be integer class labels: 0 or 1")
    return values


def assert_prediction_parity(expected: list[int], observed: list[int]) -> None:
    if len(expected) != len(observed):
        raise AssertionError(f"Expected {len(expected)} predictions, received {len(observed)}")
    if expected != observed:
        mismatch = [i for i, (a, b) in enumerate(zip(expected, observed)) if a != b]
        raise AssertionError(f"Serving/local prediction mismatch at row indices {mismatch}")


def wait_for_health(invocations_url: str, timeout_seconds: float) -> None:
    """Poll the real serving process; fail if it never reports HTTP 200."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    url = health_url(invocations_url)
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status == 200:
                    return
                last_error = RuntimeError(f"/ping returned HTTP {response.status}")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
        time.sleep(min(2, max(0, deadline - time.monotonic())))
    raise RuntimeError(f"MLflow serving did not become ready at {url}: {last_error}")


def verify_serving(
    invocations_url: str, model_path: Path, dataset_path: Path,
    sample_count: int, timeout_seconds: float,
) -> dict[str, object]:
    """Compare actual HTTP predictions with the exact local MLflow model artifact."""
    from monitoring.inference import load_sample, send_prediction_request
    from scripts.run_mlflow_project import validate_model_directory

    if sample_count < 1:
        raise ValueError("sample_count must be at least 1")
    model_dir = validate_model_directory(model_path)
    samples = load_sample(str(dataset_path), limit=sample_count)
    if len(samples) != sample_count:
        raise ValueError(f"Dataset has fewer than {sample_count} samples")
    if "target" in samples.columns:
        raise AssertionError("Never send ground-truth target labels as model features")

    import mlflow.sklearn

    local_model = mlflow.sklearn.load_model(str(model_dir))
    expected = [int(value) for value in local_model.predict(samples)]

    wait_for_health(invocations_url, timeout_seconds)
    status_code, response_body = send_prediction_request(invocations_url, samples)
    if status_code != 200:
        raise RuntimeError(f"Serving returned HTTP {status_code}")
    observed = parse_predictions(response_body, expected_count=len(expected))
    assert_prediction_parity(expected, observed)

    return {
        "status": "passed",
        "sample_count": len(samples),
        "feature_count": samples.shape[1],
        "local_and_http_predictions_equal": True,
        "model_path": str(model_dir),
        "serving_url": invocations_url,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:5001/invocations")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--dataset-path", type=Path,
        default=Path("data/processed/breast_cancer_preprocessing.csv"),
    )
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=120)
    args = parser.parse_args()
    report = verify_serving(
        args.url, args.model_path, args.dataset_path,
        sample_count=args.limit, timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
