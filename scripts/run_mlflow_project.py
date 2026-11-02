"""Run the MLflow Project and resolve the model belonging to that exact run.

Run from the repository root with ``python scripts/run_mlflow_project.py``.
GitHub Actions reads MODEL_PATH and MLFLOW_RUN_ID through GITHUB_ENV.
"""

from pathlib import Path
import os


ROOT = Path(__file__).resolve().parents[1]


def validate_model_directory(path: str | Path) -> Path:
    """Refuse missing or incomplete model artifacts before publishing."""
    model_dir = Path(path).resolve()
    if not model_dir.is_dir() or not (model_dir / "MLmodel").is_file():
        raise RuntimeError(f"MLflow model artifact is missing MLmodel: {model_dir}")
    return model_dir


def run_project_and_resolve_model(mlflow, download_artifacts, root: Path = ROOT):
    """Select a logged model using the ID returned by this training invocation."""
    mlflow.set_tracking_uri((root / "mlruns").resolve().as_uri())

    submitted = mlflow.projects.run(
        uri=str(root / "mlproject"),
        env_manager="local",
        synchronous=True,
    )
    run_id = submitted.run_id
    if not run_id:
        raise RuntimeError("MLflow Project did not return a run ID")

    status = mlflow.get_run(run_id).info.status
    if status != "FINISHED":
        raise RuntimeError(f"MLflow run {run_id} did not finish successfully: {status}")

    # No filesystem scan: use only the artifact from the returned run ID.
    model_path = download_artifacts(run_id=run_id, artifact_path="model")
    return run_id, validate_model_directory(model_path)


def github_env_content(run_id: str, model_path: Path) -> str:
    """Generate GitHub Actions environment exports for downstream steps."""
    if not run_id or any(char in run_id for char in "\r\n"):
        raise ValueError("Invalid MLflow run ID")
    path = str(model_path)
    if any(char in path for char in "\r\n"):
        raise ValueError("Model path cannot contain line breaks")
    return f"MLFLOW_RUN_ID={run_id}\nMODEL_PATH={path}\n"


def main() -> None:
    import mlflow
    from mlflow.artifacts import download_artifacts

    run_id, model_path = run_project_and_resolve_model(mlflow, download_artifacts)
    exports = github_env_content(run_id, model_path)

    github_env = os.getenv("GITHUB_ENV")
    if github_env:
        with open(github_env, "a", encoding="utf-8") as handle:
            handle.write(exports)

    print(f"MLflow run ID: {run_id}")
    print(f"Validated model artifact: {model_path}")
    if not github_env:
        print(exports, end="")


if __name__ == "__main__":
    main()
