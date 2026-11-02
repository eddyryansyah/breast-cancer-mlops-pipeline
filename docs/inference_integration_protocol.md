# Stage 3 — Inference integration and container serving smoke tests

## Scope

This stage tests that the serialized canonical MLflow model preserves the same
prediction behavior as its in-memory scikit-learn pipeline, that named numerical
features are ordered consistently, and that an actual Docker model-serving
container returns the same labels as the canonical MLflow model artifact.

The dataset target mapping is **0 = malignant, 1 = benign**. The demo is
non-diagnostic. Samples drawn from the full repository dataset are for
**functional/operational parity only**, not out-of-sample accuracy evaluation.

## Automated tests (`test` job)

```bash
.venv/bin/python -m pytest tests/test_inference_integration.py tests/test_serving_contract.py -q
.venv/bin/python -m pytest tests -q
```

- Fit the existing production `train_model()` once on the repository CSV.
- Save using `mlflow.sklearn.save_model()` and reload using both the sklearn and
  pyfunc flavors; compare prediction labels (and sklearn class probabilities).
- Verify required feature schema enforcement and the request feature order.
- Validate the expected MLflow HTTP response format and reject malformed output.

## Real serving smoke test (`train-and-build-docker` job)

After `mlflow models build-docker`, the CI job starts **that newly built image**
on `127.0.0.1:5001`, waits for `/ping`, calls `/invocations` using the same
`dataframe_split` request format as `monitoring/inference.py`, and compares the
returned classes to `mlflow.sklearn.load_model(MODEL_PATH)` for the same 8 samples.
The container is removed on success or failure.

Local reproduction after creating/downloading a canonical MLflow model directory:

```bash
docker run --rm -d --name mlflow-smoke-local -p 127.0.0.1:5001:8080 breast-cancer-mlops-pipeline:latest
.venv/bin/python -m scripts.verify_serving_endpoint \
  --url http://127.0.0.1:5001/invocations \
  --model-path /absolute/path/to/canonical/model \
  --dataset-path data/processed/breast_cancer_preprocessing.csv \
  --limit 8
# When finished:
docker rm -f mlflow-smoke-local
```

Do not run the above Docker command if the image is not built or port 5001 is
already occupied. The exact local model path is obtained from the successful
`run_mlflow_project.py` output (`Validated model artifact: ...`).

## Evidence limits and follow-up

- The in-CI smoke test proves local artifact-versus-new-build parity for a
  particular commit and training run; it does not independently pull the
  published Docker Hub/GHCR images or call the remote Hugging Face Space.
- No production prediction inputs, trained model architecture, or deployment
  mechanism are changed in this stage.
- True cross-environment E2 parity will separately compare immutable published
  image digests and remote inference outputs. Performance experiments, failure
  injection, and statistical assessment remain separate stages.
