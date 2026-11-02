"""Tests for deterministic MLflow model selection in the CI workflow."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts.run_mlflow_project import (
    github_env_content,
    run_project_and_resolve_model,
    validate_model_directory,
)


def make_model(tmp_path: Path) -> Path:
    path = tmp_path / "mlruns" / "run-new" / "artifacts" / "model"
    path.mkdir(parents=True)
    (path / "MLmodel").write_text("flavors: {}\n", encoding="utf-8")
    return path


def test_resolves_artifact_from_exact_run_id(tmp_path):
    expected_path = make_model(tmp_path)
    mlflow = Mock()
    mlflow.projects.run.return_value.run_id = "run-new"
    mlflow.get_run.return_value.info.status = "FINISHED"
    download_artifacts = Mock(return_value=str(expected_path))

    run_id, model_path = run_project_and_resolve_model(
        mlflow, download_artifacts, root=tmp_path
    )

    assert run_id == "run-new"
    assert model_path == expected_path.resolve()
    download_artifacts.assert_called_once_with(run_id="run-new", artifact_path="model")
    mlflow.projects.run.assert_called_once_with(
        uri=str(tmp_path / "mlproject"), env_manager="local", synchronous=True
    )
    mlflow.set_tracking_uri.assert_called_once_with((tmp_path / "mlruns").resolve().as_uri())


def test_rejects_non_finished_training(tmp_path):
    mlflow = Mock()
    mlflow.projects.run.return_value.run_id = "run-failed"
    mlflow.get_run.return_value.info.status = "FAILED"
    download_artifacts = Mock()

    with pytest.raises(RuntimeError, match="did not finish"):
        run_project_and_resolve_model(mlflow, download_artifacts, root=tmp_path)

    download_artifacts.assert_not_called()


def test_rejects_missing_run_id(tmp_path):
    mlflow = Mock()
    mlflow.projects.run.return_value.run_id = ""
    download_artifacts = Mock()
    with pytest.raises(RuntimeError, match="did not return a run ID"):
        run_project_and_resolve_model(mlflow, download_artifacts, root=tmp_path)
    download_artifacts.assert_not_called()


def test_rejects_missing_model_metadata(tmp_path):
    invalid_model_dir = tmp_path / "model"
    invalid_model_dir.mkdir()
    with pytest.raises(RuntimeError, match="missing MLmodel"):
        validate_model_directory(invalid_model_dir)


def test_github_env_lines(tmp_path):
    path = make_model(tmp_path)
    actual = github_env_content("new-run", path)
    assert actual == f"MLFLOW_RUN_ID=new-run\nMODEL_PATH={path}\n"


def test_rejects_newlines_in_export_values(tmp_path):
    with pytest.raises(ValueError, match="Invalid MLflow run ID"):
        github_env_content("abc\nINJECTED=1", tmp_path)
    with pytest.raises(ValueError, match="line breaks"):
        github_env_content("abc", Path("unsafe\nname"))
