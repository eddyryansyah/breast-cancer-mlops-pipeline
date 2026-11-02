# Stage 2 — Deterministic artifact selection and deployment traceability

This patch is based on `.github/workflows/main.yml` on `main` after PR #18.

## Changes

- Replace `find ... | head -n 1` with an MLflow Project run that explicitly captures the returned run ID.
- Resolve the model via `download_artifacts(run_id=<that run>, artifact_path="model")` and validate the `MLmodel` file.
- Export `MLFLOW_RUN_ID` and `MODEL_PATH` into GitHub Actions for both Docker and Hugging Face deployment paths.
- Publish Docker Hub and GHCR images with the Git commit SHA tag `sha-<full-git-sha>` **and** the existing `latest` tag for backward compatibility.
- Restrict registry logins, image publishing, and Hugging Face deployment to `main` branch push/manual workflows (never PR or dispatch on a feature branch).
- Record the source run ID and commit SHA in the Hugging Face deployment commit message.

## Local validation

```bash
.venv/bin/python -m pytest tests/test_mlflow_ci_artifacts.py -q
.venv/bin/python -m pytest tests -q
```

A local unit test run validates helper logic, but a real GitHub Actions training/build run is still necessary to confirm integration with MLflow 2.19.0, Docker Hub, GHCR, and Hugging Face.

## Important

This is source control and CI/CD code; applying this patch alone does not deploy anything. Merging to `main` can trigger the publishing/deployment workflow. Inspect the PR CI checks before merging.
