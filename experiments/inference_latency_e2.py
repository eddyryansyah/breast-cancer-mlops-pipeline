"""E2: paired, sequential local-vs-MLflow-HTTP inference latency benchmark.

Usage (repository root, after building and starting the matching E1 Docker image):
    .venv/bin/python -m experiments.inference_latency_e2 \
      --e1-report results/reproducibility_e1_verified.json \
      --output results/inference_latency_e2_verified.json

HTTP latency includes request serialization, loopback transport, serving, and
response receipt. Local latency times model.predict() only. Neither measures
startup or out-of-sample accuracy. This is a single-host exploratory benchmark.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Callable
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/processed/breast_cancer_preprocessing.csv"
DEFAULT_URL = "http://127.0.0.1:5001/invocations"
DEFAULT_IMAGE = "breast-cancer-e2:local"
DEFAULT_CONTAINER = "mlflow-e2-benchmark"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_e1_reference(report: dict, repo_root: Path, dataset_path: Path) -> tuple[Path, dict]:
    """Only use an intact E1 canonical artifact with matching model/data hashes."""
    if not report.get("summary", {}).get("all_successful") or report["summary"].get("successful_runs") != 5:
        raise ValueError("E1 report must contain five successful runs")
    runs = report.get("runs", [])
    if len(runs) != 5 or any(item.get("status") != "FINISHED" for item in runs):
        raise ValueError("E1 evidence must contain five finished runs")
    if not report.get("summary", {}).get("model_pickle_hashes_identical"):
        raise ValueError("E1 models are not byte-identical; cannot select first as common reference")
    model = Path(runs[0]["model_artifact"]).resolve()
    if not model.is_relative_to(repo_root.resolve()):
        raise ValueError("Canonical model must be within repository-local research artifacts")
    if not (model / "MLmodel").is_file() or not (model / "model.pkl").is_file():
        raise FileNotFoundError("E1 canonical MLflow model artifact missing")
    if sha256_file(model / "model.pkl") != runs[0]["model_pickle_sha256"]:
        raise ValueError("E1 model.pkl SHA-256 mismatch")
    if sha256_file(dataset_path) != report["dataset"]["sha256"]:
        raise ValueError("Processed dataset differs from E1 dataset hash")
    return model, runs[0]


def balanced_samples(df, size: int):
    """Alternate malignant/benign rows, excluding target from model inputs.

    Selection by labels is solely for operational parity; NO inference accuracy
    is computed on these (potentially training-set) examples.
    """
    if size < 1:
        raise ValueError("Batch size must be positive")
    if "target" not in df.columns:
        raise ValueError("Dataset must contain the target column")
    if not set(df["target"].unique()).issubset({0, 1}):
        raise ValueError("Expected target mapping 0=malignant, 1=benign")
    malignant = df.index[df["target"] == 0].tolist()
    benign = df.index[df["target"] == 1].tolist()
    malignant_count = (size + 1) // 2
    benign_count = size // 2
    if len(malignant) < malignant_count or len(benign) < benign_count:
        raise ValueError("Not enough rows from both classes for balanced batch")
    selected = []
    for i in range(malignant_count):
        selected.append(malignant[i])
        if i < benign_count:
            selected.append(benign[i])
    data = df.loc[selected].drop(columns=["target"])
    if data.shape[0] != size or "target" in data.columns:
        raise AssertionError("Invalid benchmark sample selection")
    return data


def validate_labels(values, expected_count: int) -> list[int]:
    """Do not silently cast floats, booleans, or invalid predicted labels."""
    from numbers import Integral
    result = list(values)
    if len(result) != expected_count:
        raise ValueError("Unexpected local prediction count")
    if any(isinstance(x, bool) or not isinstance(x, Integral) or x not in (0, 1) for x in result):
        raise ValueError("Local predictions must be integer binary class labels")
    return [int(x) for x in result]


def percentile_linear(values: list[float], quantile: float) -> float:
    if not values or not (0 <= quantile <= 1):
        raise ValueError("Nonempty values and quantile in [0,1] required")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * quantile
    low, high = math.floor(pos), math.ceil(pos)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def latency_stats(samples_ms: list[float], batch_size: int) -> dict:
    if not samples_ms or batch_size < 1 or any(not math.isfinite(x) or x <= 0 for x in samples_ms):
        raise ValueError("Latencies must be positive, finite, and nonempty")
    mean_ms = statistics.mean(samples_ms)
    return {
        "n": len(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "mean_ms": mean_ms,
        "median_ms": statistics.median(samples_ms),
        "p95_ms": percentile_linear(samples_ms, .95),
        "population_std_ms": statistics.pstdev(samples_ms),
        "sequential_throughput_samples_per_second": 1000 * batch_size / mean_ms,
    }


def inspect_container(container: str, expected_image: str, url: str) -> dict:
    """Bind endpoint to a specific running Docker container and published port."""
    parsed = urlsplit(url)
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.path != "/invocations":
        raise ValueError("E2 only supports http://127.0.0.1:<port>/invocations")
    port = parsed.port
    if not port or not 1 <= port <= 65535:
        raise ValueError("Explicit local serving port required")
    command = subprocess.run(["docker", "inspect", container], capture_output=True, text=True, check=True, timeout=15)
    entries = json.loads(command.stdout)
    if len(entries) != 1:
        raise ValueError("Expected exactly one Docker container")
    entry = entries[0]
    if not entry.get("State", {}).get("Running"):
        raise ValueError("Benchmark container is not running")
    image_name = entry.get("Config", {}).get("Image")
    if image_name != expected_image:
        raise ValueError(f"Unexpected image: {image_name!r} (expected {expected_image!r})")
    ports = entry.get("NetworkSettings", {}).get("Ports", {}).get("8080/tcp") or []
    if not any(p.get("HostIp") == "127.0.0.1" and p.get("HostPort") == str(port) for p in ports):
        raise ValueError("Container port 8080 is not bound to requested 127.0.0.1 port")
    image_id = entry.get("Image")
    if not image_id:
        raise ValueError("Missing Docker image ID")
    return {
        "container_name": container,
        "image_name": image_name,
        "image_id": image_id,
        "published_port": port,
        "serving_url": url,
    }


def benchmark_pair(
    local_call: Callable[[], object], http_call: Callable[[], object],
    decode_local: Callable[[object], list[int]], decode_http: Callable[[object], list[int]],
    expected_labels: list[int], warmup: int, iterations: int,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, list[float]]:
    """Alternating sequential measurements reduce systematic run-order bias.

    Only the call is timed. Decoding/parity assertions are outside measured
    intervals, but HTTP call includes request encoding and response receipt.
    """
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be >=0, iterations must be >=1")
    measured = {"local": [], "http": []}
    operations = {"local": (local_call, decode_local), "http": (http_call, decode_http)}
    for iteration in range(warmup + iterations):
        order = ("local", "http") if iteration % 2 == 0 else ("http", "local")
        for kind in order:
            call, decode = operations[kind]
            start = clock_ns()
            raw = call()
            elapsed_ns = clock_ns() - start
            labels = decode(raw)
            if labels != expected_labels:
                raise AssertionError(f"E2 {kind} prediction mismatch at iteration {iteration}")
            if iteration >= warmup:
                if elapsed_ns <= 0:
                    raise ValueError("Measured monotonic duration must be positive")
                measured[kind].append(elapsed_ns / 1_000_000)
    return measured


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def run_benchmark(
    e1_report: Path, dataset_path: Path, destination: Path, url: str,
    container: str, image_name: str, batch_sizes: tuple[int, ...],
    warmup: int, iterations: int,
) -> dict:
    """Execute benchmark only against the verified E1 model and live container."""
    import mlflow
    import mlflow.sklearn
    import pandas as pd
    import sklearn
    from monitoring.inference import send_prediction_request
    from scripts.verify_serving_endpoint import parse_predictions, wait_for_health

    if not batch_sizes or any(x < 1 for x in batch_sizes) or len(set(batch_sizes)) != len(batch_sizes):
        raise ValueError("Batch sizes must be distinct positive integers")
    if warmup < 0 or iterations < 1:
        raise ValueError("Invalid warmup/iterations")
    e1 = json.loads(e1_report.read_text(encoding="utf-8"))
    model_dir, first_run = verify_e1_reference(e1, ROOT, dataset_path)
    docker = inspect_container(container, image_name, url)
    wait_for_health(url, timeout_seconds=120)
    model = mlflow.sklearn.load_model(str(model_dir))
    df = pd.read_csv(dataset_path)
    if df.shape[1] != 31 or len(df) != 569:
        raise ValueError("Expected repository dataset: 569 rows, 30 features, and target")
    if df.drop(columns=["target"]).isna().any().any():
        raise ValueError("Dataset contains missing input features")

    report = {
        "protocol": "E2-paired-sequential-local-vs-http-latency-v1",
        "limitations": "Single host, sequential requests, loopback HTTP; not cloud/network throughput, clinical evaluation, or cross-host latency.",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "reference": {
            "e1_run_id": first_run["mlflow_run_id"],
            "model_path": model_dir.relative_to(ROOT).as_posix(),
            "model_pickle_sha256": first_run["model_pickle_sha256"],
            "dataset_sha256": e1["dataset"]["sha256"],
        },
        "environment": {
            "python": sys.version.split()[0], "platform": platform.platform(),
            "mlflow": mlflow.__version__, "scikit_learn": sklearn.__version__,
            "docker": docker,
            "cpu_count": os.cpu_count(),
        },
        "settings": {
            "batch_sizes": list(batch_sizes), "warmup_per_method_per_batch": warmup,
            "measured_iterations_per_method_per_batch": iterations,
            "request_mode": "sequential urllib.request; no configured connection pooling",
            "schedule": "alternate local and HTTP first within each iteration",
            "timing": "local=model.predict only; HTTP=request serialization + network + server + body receipt",
            "sampling": "deterministic alternating 0=malignant and 1=benign dataset rows; targets never sent",
            "accuracy_claim": "none; train/test membership not separated in latency samples",
        },
        "batches": [],
        "completed": False,
    }
    write_json_atomic(destination, report)
    for size in batch_sizes:
        samples = balanced_samples(df, size)
        expected = validate_labels(model.predict(samples), size)
        print(f"[E2] Batch {size}: warmup={warmup}, timed={iterations} per method ...", flush=True)

        def local_call():
            return model.predict(samples)

        def http_call():
            return send_prediction_request(url, samples)

        def decode_http(raw):
            status, body = raw
            if status != 200:
                raise RuntimeError(f"MLflow serving returned HTTP {status}")
            return parse_predictions(body, expected_count=size)

        measured = benchmark_pair(
            local_call, http_call,
            lambda raw: validate_labels(raw, size), decode_http,
            expected, warmup, iterations,
        )
        # Pure operational parity fingerprint: class counts are predictions,
        # not clinical outcomes or test-set classification metrics.
        batch_report = {
            "batch_size": size,
            "parity_passed": True,
            "prediction_counts": {str(k): expected.count(k) for k in (0, 1)},
            "local": latency_stats(measured["local"], size),
            "http": latency_stats(measured["http"], size),
            "raw_latency_ms": measured,
        }
        report["batches"].append(batch_report)
        write_json_atomic(destination, report)
        print(
            f"[E2] Batch {size} OK: local median {batch_report['local']['median_ms']:.3f} ms, "
            f"HTTP median {batch_report['http']['median_ms']:.3f} ms", flush=True,
        )
    report["completed"] = True
    write_json_atomic(destination, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e1-report", type=Path, default=ROOT / "results/reproducibility_e1_verified.json")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, default=ROOT / "results/inference_latency_e2_verified.json")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--batch-sizes", default="1,8,32")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    try:
        sizes = tuple(int(value.strip()) for value in args.batch_sizes.split(","))
    except ValueError as exc:
        parser.error(f"Invalid batch sizes: {exc}")
    report = run_benchmark(
        args.e1_report.resolve(), args.dataset.resolve(), args.output.resolve(),
        args.url, args.container, args.image, sizes, args.warmup, args.iterations,
    )
    print("E2 report:", args.output)
    print(json.dumps({
        "completed": report["completed"],
        "e1_run_id": report["reference"]["e1_run_id"],
        "docker_image_id": report["environment"]["docker"]["image_id"],
        "batches": [{"batch_size": b["batch_size"],
                     "local_median_ms": b["local"]["median_ms"],
                     "http_median_ms": b["http"]["median_ms"],
                     "local_p95_ms": b["local"]["p95_ms"],
                     "http_p95_ms": b["http"]["p95_ms"]} for b in report["batches"]],
    }, indent=2))


if __name__ == "__main__":
    main()
