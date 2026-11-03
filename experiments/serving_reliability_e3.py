"""E3: local-only MLflow serving reliability and explicit restart recovery.

Run only against the named E3 container on 127.0.0.1:5001. In particular,
--enable-restart is required before any ``docker restart`` will be issued.
This is an operational experiment, not a clinical model evaluation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import platform
from pathlib import Path
import statistics
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
E3_CONTAINER = "breast-cancer-e3-serving"
E2_IMAGE = "breast-cancer-e2:local"
INVOCATIONS = "http://127.0.0.1:5001/invocations"


def validate_local_endpoint(url: str) -> str:
    """Prevent accidentally probing a remotely deployed or production service."""
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme, parsed.hostname, parsed.port, parsed.path,
            parsed.username, parsed.password, parsed.query, parsed.fragment) != (
            "http", "127.0.0.1", 5001, "/invocations", None, None, "", ""
    ):
        raise ValueError("E3 only permits http://127.0.0.1:5001/invocations")
    return url


def inspect_container(name: str, image: str, expected_id: str) -> dict:
    """Reject any container not matching the isolated E3 name/image/port."""
    if name != E3_CONTAINER or image != E2_IMAGE:
        raise ValueError("E3 requires the dedicated research container and image")
    result = subprocess.run(
        ["docker", "inspect", name], check=True, capture_output=True, text=True, timeout=15
    )
    document = json.loads(result.stdout)
    if len(document) != 1 or not isinstance(document[0], dict):
        raise ValueError("Expected exactly one Docker container")
    details = document[0]
    bindings = details.get("NetworkSettings", {}).get("Ports", {}).get("8080/tcp")
    if (
        details.get("Name") != f"/{name}"
        or not details.get("State", {}).get("Running")
        or details.get("Config", {}).get("Image") != image
        or details.get("Image") != expected_id
        or bindings != [{"HostIp": "127.0.0.1", "HostPort": "5001"}]
    ):
        raise ValueError("Docker container does not match the E3 isolated setup")
    return {"name": name, "image": image, "image_id": expected_id,
            "host": "127.0.0.1", "host_port": 5001, "container_port": 8080}


def request(method: str, url: str, payload: bytes | None = None, timeout: float = 10) -> tuple[int, str]:
    """Return HTTP errors as observations; propagate transport failures."""
    if method not in {"GET", "POST"}:
        raise ValueError("Unsupported method")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = urllib.request.Request(url, data=payload, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, response.read(128 * 1024).decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        with error:
            return error.code, error.read(128 * 1024).decode("utf-8", errors="replace")


def encode_frame(samples) -> bytes:
    return json.dumps({"dataframe_split": {
        "columns": samples.columns.tolist(), "data": samples.values.tolist()
    }}, allow_nan=False, separators=(",", ":")).encode("utf-8")


def invalid_payloads(samples) -> list[tuple[str, bytes]]:
    """Non-adversarial malformed requests; never include real patient details."""
    columns = samples.columns.tolist()
    data = samples.values.tolist()
    return [
        ("malformed_json", b'{"dataframe_split":'),
        ("unsupported_root_array", b"[]"),
        ("missing_required_feature", json.dumps({"dataframe_split": {
            "columns": columns[1:], "data": [row[1:] for row in data]
        }}).encode("utf-8")),
        ("inconsistent_row_length", json.dumps({"dataframe_split": {
            "columns": columns, "data": [data[0][:-1]]
        }}).encode("utf-8")),
        ("nonnumeric_feature", json.dumps({"dataframe_split": {
            "columns": columns, "data": [["not-a-number", *data[0][1:]]]
        }}).encode("utf-8")),
    ]


def parse_labels(body: str, count: int) -> list[int]:
    from scripts.verify_serving_endpoint import parse_predictions
    return parse_predictions(body, expected_count=count)


def assert_valid_prediction(url: str, payload: bytes, expected: list[int]) -> None:
    from scripts.verify_serving_endpoint import assert_prediction_parity
    code, body = request("POST", url, payload)
    if code != 200:
        raise AssertionError(f"Valid inference request returned HTTP {code}")
    assert_prediction_parity(expected, parse_labels(body, len(expected)))


def healthy(url: str) -> bool:
    ping = urllib.parse.urlsplit(url)._replace(path="/ping").geturl()
    try:
        code, _ = request("GET", ping, timeout=3)
        return code == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def wait_for_ready(url: str, timeout: float, interval: float = 0.25) -> float:
    """Elapsed time until first 200 /ping, measured using monotonic time."""
    if timeout <= 0 or interval <= 0:
        raise ValueError("Readiness timeout and interval must be positive")
    started = time.perf_counter()
    while time.perf_counter() - started < timeout:
        if healthy(url):
            return time.perf_counter() - started
        time.sleep(min(interval, max(0, timeout - (time.perf_counter() - started))))
    raise TimeoutError("E3 serving container did not recover within readiness timeout")


def latency_summary(values: list[float]) -> dict:
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Nonempty finite nonnegative latencies required")
    ordered = sorted(values)
    p = (len(ordered) - 1) * 0.95
    low = math.floor(p)
    high = math.ceil(p)
    p95 = ordered[low] + (ordered[high] - ordered[low]) * (p - low)
    return {
        "n": len(values), "min_ms": min(values), "median_ms": statistics.median(values),
        "p95_ms": p95, "max_ms": max(values), "mean_ms": statistics.mean(values),
        "population_std_ms": statistics.pstdev(values),
    }


def write_report(report: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + ".tmp")
    temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(destination)


def run_reliability(
    e1_report: Path, e2_report: Path, dataset: Path, output: Path,
    url: str, container: str, image: str, valid_requests: int,
    restarts: int, enable_restart: bool, readiness_timeout: float,
) -> dict:
    """Run valid/invalid probes and optional *explicit* E3-only restarts."""
    validate_local_endpoint(url)
    if container != E3_CONTAINER or image != E2_IMAGE:
        raise ValueError("Dedicated E3 container/image required")
    if not (1 <= valid_requests <= 1000 and 0 <= restarts <= 10):
        raise ValueError("valid_requests must be 1..1000 and restarts 0..10")
    if restarts and not enable_restart:
        raise ValueError("--enable-restart required for any restart")
    if readiness_timeout <= 0:
        raise ValueError("readiness_timeout must be positive")

    import mlflow.sklearn
    import pandas as pd
    from experiments.inference_latency_e2 import balanced_samples, verify_e1_reference
    from experiments.reproducibility_e1 import sha256_file
    from scripts.verify_serving_endpoint import assert_prediction_parity

    e1 = json.loads(e1_report.read_text(encoding="utf-8"))
    e2 = json.loads(e2_report.read_text(encoding="utf-8"))
    model_path, run = verify_e1_reference(e1, ROOT, dataset)
    if not e2.get("completed") or len(e2.get("batches", [])) != 3:
        raise ValueError("Verified E2 report required")
    expected_id = e2.get("environment", {}).get("docker", {}).get("image_id")
    if not isinstance(expected_id, str) or not expected_id.startswith("sha256:"):
        raise ValueError("Missing immutable E2 image ID")
    if run["model_pickle_sha256"] != e2["reference"]["model_pickle_sha256"]:
        raise ValueError("E1 and E2 model reference mismatch")
    if e1["dataset"]["sha256"] != e2["reference"]["dataset_sha256"]:
        raise ValueError("E1 and E2 dataset mismatch")
    container_info = inspect_container(container, image, expected_id)
    data = pd.read_csv(dataset)
    samples = balanced_samples(data, 8)
    model = mlflow.sklearn.load_model(str(model_path))
    expected = [int(x) for x in model.predict(samples)]
    assert set(expected).issubset({0, 1}) and len(expected) == len(samples)
    payload = encode_frame(samples)
    wait_for_ready(url, readiness_timeout)
    assert_valid_prediction(url, payload, expected)

    report = {
        "protocol": "E3-local-serving-reliability-and-recovery-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "One host and a dedicated localhost Docker container; no production traffic or clinical evaluation.",
        "reference": {
            "e1_run_id": run["mlflow_run_id"],
            "model_pickle_sha256": sha256_file(model_path / "model.pkl"),
            "dataset_sha256": e1["dataset"]["sha256"],
            "e2_docker_image_id": expected_id,
        },
        "environment": {"platform": platform.platform(), "container": container_info},
        "settings": {"url": url, "valid_request_attempts": valid_requests,
                     "restart_attempts": restarts, "restart_explicitly_enabled": enable_restart,
                     "readiness_timeout_seconds": readiness_timeout,
                     "sample_count": 8, "feature_count": samples.shape[1]},
        "valid_requests": {"attempted": valid_requests, "succeeded": 0, "errors": [],
                           "latencies_ms": [], "summary_ms": None},
        "invalid_requests": [], "recovery": [], "summary": {"completed": False},
    }
    write_report(report, output)
    # Each request is checked against the *same* local model predictions.
    for index in range(1, valid_requests + 1):
        started = time.perf_counter()
        try:
            status, body = request("POST", url, payload)
            elapsed_ms = (time.perf_counter() - started) * 1000
            if status != 200:
                raise AssertionError(f"HTTP {status}")
            assert_prediction_parity(expected, parse_labels(body, len(expected)))
            report["valid_requests"]["succeeded"] += 1
            report["valid_requests"]["latencies_ms"].append(elapsed_ms)
        except Exception as exc:
            report["valid_requests"]["errors"].append(
                {"attempt": index, "error_type": type(exc).__name__, "message": str(exc)[:240]}
            )
    successful_ms = report["valid_requests"]["latencies_ms"]
    if successful_ms:
        report["valid_requests"]["summary_ms"] = latency_summary(successful_ms)
    write_report(report, output)
    print(f"[E3] Valid HTTP requests: {report['valid_requests']['succeeded']}/{valid_requests}", flush=True)

    for name, invalid in invalid_payloads(samples):
        case = {"name": name, "status_code": None, "rejected": False,
                "client_error_4xx": False, "server_error_5xx": False,
                "healthy_after": False, "valid_prediction_after": False}
        try:
            code, _ = request("POST", url, invalid)
            case.update({"status_code": code, "rejected": not 200 <= code < 300,
                         "client_error_4xx": 400 <= code < 500,
                         "server_error_5xx": 500 <= code < 600})
            case["healthy_after"] = healthy(url)
            assert_valid_prediction(url, payload, expected)
            case["valid_prediction_after"] = True
        except Exception as exc:
            case["error_type"] = type(exc).__name__
            case["error_message"] = str(exc)[:240]
        report["invalid_requests"].append(case)
        write_report(report, output)
        print(f"[E3] Invalid {name}: HTTP {case['status_code']}; rejected={case['rejected']}", flush=True)

    for index in range(1, restarts + 1):
        # This is the *only* container mutation performed by E3.
        record = {"attempt": index, "status": "FAILED", "time_to_ready_seconds": None,
                  "time_to_prediction_seconds": None, "prediction_parity": False}
        started = time.perf_counter()
        try:
            inspect_container(container, image, expected_id)
            subprocess.run(["docker", "restart", "--time", "2", container],
                           capture_output=True, text=True, check=True, timeout=45)
            wait_for_ready(url, readiness_timeout)
            record["time_to_ready_seconds"] = time.perf_counter() - started
            # Readiness alone is not enough; verify *model* serving is usable.
            assert_valid_prediction(url, payload, expected)
            record["time_to_prediction_seconds"] = time.perf_counter() - started
            record["prediction_parity"] = True
            record["status"] = "FINISHED"
        except Exception as exc:
            record["error_type"] = type(exc).__name__
            record["error_message"] = str(exc)[:240]
        report["recovery"].append(record)
        write_report(report, output)
        print(f"[E3] Restart {index}/{restarts}: {record['status']}", flush=True)
        if record["status"] != "FINISHED":
            break

    valid_ok = report["valid_requests"]["succeeded"] == valid_requests
    invalid_ok = all(item["rejected"] and item["healthy_after"] and
                     item["valid_prediction_after"] for item in report["invalid_requests"])
    recovery_ok = len(report["recovery"]) == restarts and all(
        item["status"] == "FINISHED" for item in report["recovery"]
    )
    report["summary"] = {
        "completed": valid_ok and invalid_ok and recovery_ok,
        "valid_requests_succeeded": report["valid_requests"]["succeeded"],
        "valid_requests_attempted": valid_requests,
        "invalid_requests_rejected": sum(item["rejected"] for item in report["invalid_requests"]),
        "invalid_requests_attempted": len(report["invalid_requests"]),
        "invalid_requests_all_4xx": all(item["client_error_4xx"] for item in report["invalid_requests"]),
        "invalid_requests_any_5xx": any(item["server_error_5xx"] for item in report["invalid_requests"]),
        "restarts_recovered": sum(item["status"] == "FINISHED" for item in report["recovery"]),
        "restarts_requested": restarts,
        "recovery_ms": latency_summary([item["time_to_prediction_seconds"] * 1000
            for item in report["recovery"] if item["status"] == "FINISHED"]) if
            any(item["status"] == "FINISHED" for item in report["recovery"]) else None,
    }
    write_report(report, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e1-report", type=Path, default=ROOT / "results/reproducibility_e1_verified.json")
    parser.add_argument("--e2-report", type=Path, default=ROOT / "results/inference_latency_e2_public.json")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/processed/breast_cancer_preprocessing.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results/serving_reliability_e3_verified.json")
    parser.add_argument("--url", default=INVOCATIONS)
    parser.add_argument("--container", default=E3_CONTAINER)
    parser.add_argument("--image", default=E2_IMAGE)
    parser.add_argument("--valid-requests", type=int, default=50)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--enable-restart", action="store_true")
    parser.add_argument("--readiness-timeout", type=float, default=120)
    args = parser.parse_args()
    report = run_reliability(
        e1_report=args.e1_report.resolve(), e2_report=args.e2_report.resolve(),
        dataset=args.dataset.resolve(), output=args.output.resolve(),
        url=args.url, container=args.container, image=args.image,
        valid_requests=args.valid_requests, restarts=args.restarts,
        enable_restart=args.enable_restart, readiness_timeout=args.readiness_timeout,
    )
    print("E3 report:", args.output)
    print(json.dumps(report["summary"], indent=2))
    if not report["summary"]["completed"]:
        raise SystemExit("E3 incomplete: inspect individual request/recovery evidence")


if __name__ == "__main__":
    main()
