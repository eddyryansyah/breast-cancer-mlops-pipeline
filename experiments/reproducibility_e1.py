"""E1: rerun the *unchanged* production MLflow Project in isolated workspaces.

Run from the repository root:

    .venv/bin/python -m experiments.reproducibility_e1 --repeats 5

This reuses the active Python environment; it does NOT reinstall dependencies
for every run. Predictions on all 569 records are only a parity fingerprint,
never an independent test-set accuracy estimate.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
DATASET_REL = Path("data/processed/breast_cancer_preprocessing.csv")
SOURCE_FILES = (
    "mlproject/MLproject",
    "mlproject/modelling.py",
    "mlproject/conda.yaml",
    "mlproject/requirements.txt",
)
METRIC_NAMES = ("accuracy", "precision", "recall", "f1_score")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_labels(values) -> str:
    labels = [int(value) for value in values]
    if any(label not in (0, 1) for label in labels):
        raise ValueError("Model predictions must use 0=malignant, 1=benign")
    return hashlib.sha256(json.dumps(labels, separators=(",", ":")).encode("utf-8")).hexdigest()


def source_file_hashes(repo_root: Path) -> dict[str, str]:
    return {name: sha256_file(repo_root / name) for name in SOURCE_FILES}


def prepare_workspace(repo_root: Path, workspace: Path, dataset: Path) -> None:
    """Copy only the MLflow Project source and processed data; omit stale runs."""
    if workspace.exists():
        raise FileExistsError(f"Workspace already exists: {workspace}")
    workspace.mkdir(parents=True)
    shutil.copytree(
        repo_root / "mlproject",
        workspace / "mlproject",
        ignore=shutil.ignore_patterns("mlruns", "__pycache__", "*.pyc", "*.joblib", "*.pkl"),
    )
    processed = workspace / DATASET_REL
    processed.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dataset, processed)
    for name in SOURCE_FILES:
        if sha256_file(repo_root / name) != sha256_file(workspace / name):
            raise RuntimeError(f"Training source changed in workspace: {name}")
    if sha256_file(dataset) != sha256_file(processed):
        raise RuntimeError("Dataset checksum changed during workspace preparation")


def git_head(repo_root: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root,
            check=True, capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.strip() or None


def summarize(records: list[dict], repeats: int) -> dict:
    successful = [record for record in records if record["status"] == "FINISHED"]
    metric_keys = set(METRIC_NAMES)
    metric_keys.update(key for record in successful for key in record["metrics"])
    spans = {}
    for name in sorted(metric_keys):
        values = [record["metrics"][name] for record in successful if name in record["metrics"]]
        if values:
            spans[name] = {
                "min": min(values), "max": max(values),
                "range": max(values) - min(values),
                "population_std": statistics.pstdev(values),
                "observations": len(values),
            }
    prediction_hashes = [record["prediction_sha256"] for record in successful]
    model_hashes = [record["model_pickle_sha256"] for record in successful]
    duration_values = [record["duration_seconds"] for record in successful]
    return {
        "attempts": len(records),
        "requested_repeats": repeats,
        "successful_runs": len(successful),
        "failed_runs": len(records) - len(successful),
        "all_successful": len(successful) == repeats and len(records) == repeats,
        "prediction_hashes_identical": (
            len(prediction_hashes) == repeats and len(set(prediction_hashes)) == 1
        ),
        "model_pickle_hashes_identical": (
            len(model_hashes) == repeats and len(set(model_hashes)) == 1
        ),
        "metrics": spans,
        "duration_seconds": (
            {"min": min(duration_values), "max": max(duration_values),
             "mean": statistics.mean(duration_values)} if duration_values else None
        ),
    }


def write_report(report: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement prevents a partial JSON report on interruption.
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(destination)


@contextmanager
def use_current_python_for_mlflow():
    """Ensure MLflow Project's `python` command uses this process's interpreter.

    `env_manager="local"` inherits PATH for the training subprocess. Running
    this script as `.venv/bin/python` alone does not normally activate .venv.
    Keep the temporary PATH override scoped to the synchronous MLflow call.
    """
    python_dir = Path(sys.executable).absolute().parent
    if not (python_dir / "python").is_file():
        raise RuntimeError(f"Cannot find 'python' beside the E1 interpreter: {python_dir}")

    original_path = os.environ.get("PATH")
    os.environ["PATH"] = str(python_dir) + os.pathsep + (original_path or "")
    try:
        result = subprocess.run(
            ["python", "-c", "import sys; print(sys.executable)"],
            check=True, capture_output=True, text=True, timeout=30,
        )
        child_python = Path(result.stdout.strip())
        if child_python.resolve() != Path(sys.executable).resolve():
            raise RuntimeError(
                "MLflow's Python command resolved to an unexpected interpreter: "
                f"{child_python} (expected {sys.executable})"
            )
        yield str(child_python)
    finally:
        if original_path is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = original_path


def run_experiment(
    repo_root: Path, dataset_path: Path, destination: Path,
    artifact_root: Path, repeats: int = 5,
) -> dict:
    """Execute repeated isolated MLflow Project runs; retain all raw run stores."""
    if not 1 <= repeats <= 30:
        raise ValueError("repeats must be between 1 and 30")
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not (repo_root / "mlproject" / "MLproject").is_file():
        raise FileNotFoundError("MLflow Project missing from repository root")
    for path in (dataset_path, repo_root / "mlproject"):
        if path.is_symlink():
            raise ValueError(f"Do not use symlinks for research inputs: {path}")
    import joblib
    import mlflow
    import mlflow.sklearn
    import numpy
    import pandas
    import sklearn
    from mlflow.artifacts import download_artifacts

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    batch_dir = artifact_root / f"batch_{timestamp}_{uuid4().hex[:8]}"
    batch_dir.mkdir(parents=True, exist_ok=False)
    dataset_hash = sha256_file(dataset_path)
    sample_frame = pandas.read_csv(dataset_path)
    if "target" not in sample_frame.columns:
        raise ValueError("Dataset must have a target column")
    if len(sample_frame) != 569 or len(sample_frame.columns) != 31:
        raise ValueError("Expected 569 rows and 30 features plus target")
    features = sample_frame.drop(columns=["target"])

    report = {
        "protocol": "E1-isolated-workspaces-fixed-environment-v1",
        "scope": "Repeat independent MLflow runs using copied source/data and the SAME installed Python environment; not clean dependency installs or clinical evaluation.",
        "target_mapping": {"0": "malignant", "1": "benign"},
        "legacy_metrics_positive_class": "benign (label=1)",
        "dataset": {"path": DATASET_REL.as_posix(), "sha256": dataset_hash,
                    "rows": len(sample_frame), "feature_count": features.shape[1]},
        "source_sha256": source_file_hashes(repo_root),
        "git_head": git_head(repo_root),
        "environment": {
            "python": sys.version.split()[0], "python_executable": sys.executable,
            "platform": platform.platform(),
            "mlflow": mlflow.__version__, "scikit_learn": sklearn.__version__,
            "numpy": numpy.__version__, "pandas": pandas.__version__,
            "joblib": joblib.__version__,
        },
        "requested_repeats": repeats,
        "artifact_directory": str(batch_dir),
        "runs": [],
        "summary": summarize([], repeats),
    }
    write_report(report, destination)
    for index in range(1, repeats + 1):
        workspace = batch_dir / f"run_{index:02d}"
        item = {"attempt": index, "status": "FAILED", "mlflow_run_id": None}
        started = time.perf_counter()
        print(f"[E1] Running {index}/{repeats} ...", flush=True)
        try:
            prepare_workspace(repo_root, workspace, dataset_path)
            mlflow.set_tracking_uri((workspace / "mlruns").resolve().as_uri())
            with use_current_python_for_mlflow() as training_python:
                item["mlflow_project_python"] = training_python
                submitted = mlflow.projects.run(
                    uri=str((workspace / "mlproject").resolve()),
                    env_manager="local", synchronous=True,
                )
            run_id = submitted.run_id
            if not run_id:
                raise RuntimeError("MLflow Project returned no run ID")
            item["mlflow_run_id"] = run_id
            logged_run = mlflow.get_run(run_id)
            if logged_run.info.status != "FINISHED":
                raise RuntimeError(f"Training run status: {logged_run.info.status}")
            model_dir = Path(download_artifacts(run_id=run_id, artifact_path="model"))
            if not (model_dir / "MLmodel").is_file() or not (model_dir / "model.pkl").is_file():
                raise RuntimeError("Incomplete canonical sklearn MLflow artifact")
            model = mlflow.sklearn.load_model(str(model_dir))
            labels = model.predict(features)
            if len(labels) != len(features):
                raise AssertionError("Unexpected inference count")
            metrics = {}
            for name in METRIC_NAMES:
                if name not in logged_run.data.metrics:
                    raise RuntimeError(f"Missing logged metric: {name}")
                value = float(logged_run.data.metrics[name])
                if not math.isfinite(value):
                    raise RuntimeError(f"Nonfinite logged metric: {name}")
                metrics[name] = value
            item.update({
                "status": "FINISHED", "metrics": metrics,
                "model_pickle_sha256": sha256_file(model_dir / "model.pkl"),
                "prediction_sha256": sha256_labels(labels),
                "predictions_checked": len(labels),
                "feature_count": features.shape[1],
                "model_artifact": str(model_dir),
            })
        except Exception as error:
            item["error_type"] = type(error).__name__
            item["error_message"] = str(error)
        finally:
            item["duration_seconds"] = round(time.perf_counter() - started, 4)
            report["runs"].append(item)
            report["summary"] = summarize(report["runs"], repeats)
            write_report(report, destination)
            print(f"[E1] {index}/{repeats}: {item['status']} ({item['duration_seconds']:.2f}s)", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dataset", type=Path, default=ROOT / DATASET_REL)
    parser.add_argument("--output", type=Path, default=ROOT / "results/reproducibility_e1.json")
    parser.add_argument("--artifact-root", type=Path, default=ROOT / ".research_artifacts/e1")
    args = parser.parse_args()
    report = run_experiment(
        repo_root=ROOT, dataset_path=args.dataset.resolve(),
        destination=args.output.resolve(), artifact_root=args.artifact_root.resolve(),
        repeats=args.repeats,
    )
    print("E1 report:", args.output)
    print(json.dumps(report["summary"], indent=2))
    if not report["summary"]["all_successful"]:
        raise SystemExit("E1 incomplete: one or more runs failed; inspect report and artifacts")


if __name__ == "__main__":
    main()
