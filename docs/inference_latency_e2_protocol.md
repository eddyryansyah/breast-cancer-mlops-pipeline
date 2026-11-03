# E2 — Single-host inference latency benchmark

## Research question

For the **same** E1 canonical model artifact, what are the observed local
`model.predict()` timings and end-to-end loopback MLflow `/invocations`
request timings for batch sizes 1, 8, and 32?

The timings have **different scopes**. Local timing measures `predict()` only;
HTTP timing includes client JSON serialization, new-request connection
handling, network loopback, MLflow serving, and response receipt. HTTP results
are **not** solely model inference time. Do not interpret their ratio as a pure
algorithm speedup or penalty.

## Conditions and limitations

- Single local macOS host, one sequential requesting client, one Docker container.
  No concurrent load, remote network, GPU, or cross-host experiment is claimed.
- Use exactly the first canonical E1 model artifact from the local E1 verified
  report. Check its `model.pkl` SHA-256 against E1 and check the dataset SHA-256.
  **Do not retrain** or compare an unrelated prebuilt model/image.
- The operator builds the image from that E1 canonical model. The benchmark
  verifies a live named container uses the expected image name and loopback port
  and logs the immutable Docker image ID. It cannot independently prove image
  contents match the model file without the documented build provenance.
- Each batch uses a deterministic, alternating selection of `0=malignant` and
  `1=benign` records from the **same processed dataset**. The target is used only
  for balanced sample selection and is never sent to the model/server. With
  batch 1, one malignant row is selected; batch 8 has 4+4 and batch 32 has 16+16.
  Records may have been in training; parity and timing are **not accuracy tests**.
- Warm up each method for 20 iterations and then take 100 measurements per
  method per batch, alternating which method runs first in each iteration.
  All runs are sequential and checked for class prediction parity. The
  benchmark stops on HTTP, parsing, or parity failure.
- Report mean, median, interpolated p95, min/max, population standard deviation,
  and sequential sample throughput. `p95` uses linear interpolation over sorted
  100 observations at `(n-1)*0.95`. Throughput is a derived single-client
  value, **not** maximum server capacity.
- Startup/model loading and `/ping` readiness happen **before** timed requests.
  The benchmark does not measure cold-start latency or concurrency behavior.
- HTTP client uses the existing `monitoring.inference.send_prediction_request`
  helper (`urllib.request`); **no connection pooling is configured**. TCP
  connection behavior is therefore part of reported end-to-end HTTP latency.
- Host processes, CPU thermal throttling, system load, container resources,
  loopback networking and serialization may affect results. Avoid publication
  of machine-independent performance claims based on one batch of measurements.

## Prerequisites

Run from repository root using an existing `.venv` and running Docker Engine.
Preserve `results/reproducibility_e1_verified.json` and the E1 artifacts in
`.research_artifacts/e1/`; these are intentionally not committed to Git.
Verify E1 reference and Docker availability before proceeding.

## Build a Docker image from the exact E1 artifact

Resolve the verified model path without manually copying a long absolute path:

```bash
E2_MODEL_PATH=$(.venv/bin/python - <<'PY'
import json
from pathlib import Path
from experiments.inference_latency_e2 import verify_e1_reference
root = Path.cwd().resolve()
report = json.loads(Path('results/reproducibility_e1_verified.json').read_text())
model, _ = verify_e1_reference(report, root, root / 'data/processed/breast_cancer_preprocessing.csv')
print(model)
PY
)
printf 'Verified E2 model: %s\n' "$E2_MODEL_PATH"
```

### Build method used in the verified E2 run

Two initial `mlflow models build-docker` attempts stopped while downloading
`mlflow==2.19.0` from PyPI due to `ReadTimeoutError`. The successful build used
MLflow's **generated** Dockerfile for the **same verified E1 model**. Only
pip network tolerance was changed; production CI/CD configuration and the
model were not modified:

```bash
.venv/bin/mlflow models generate-dockerfile \
  --model-uri "$E2_MODEL_PATH" \
  --output-directory .research_artifacts/e2/docker-build-context \
  --env-manager local
```

In the generated `Dockerfile`, directly after `WORKDIR /opt/mlflow`, add:

```dockerfile
ENV PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10
```

This sets pip's per-read timeout and retry count for subsequent installation
commands **inside the image**; it does not alter the package version. Then:

```bash
docker build --tag breast-cancer-e2:local \
  .research_artifacts/e2/docker-build-context

docker image inspect breast-cancer-e2:local --format 'Image ID: {{.Id}}'
```

The verified build completed in 1715.3 seconds, which is **build time**, not
inference latency. Its immutable local image ID was
`sha256:56783f46aab5922cdf500cf0662acc0228011b3785ecceabd97edb09634a4273`.
The generated Dockerfile used `python:3.12.7-slim` and `mlflow==2.19.0`.
Keep the generated build context under ignored `.research_artifacts/e2/`.
The image is **local**, not a published Docker Hub or GHCR digest.

Check that port 5001 is unused; start a **loopback-only** container:

```bash
lsof -nP -iTCP:5001 -sTCP:LISTEN
docker run --detach --rm --name breast-cancer-e2-serving \
  -p 127.0.0.1:5001:8080 breast-cancer-e2:local
```

The benchmark validates the running container name, image name, immutable image
ID, loopback port mapping, and HTTP `/ping` before collecting measurements.

## Execute and inspect

```bash
.venv/bin/python -m pytest tests/test_inference_latency_e2.py -q
.venv/bin/python -m pytest tests -q
.venv/bin/python -m experiments.inference_latency_e2 \
  --e1-report results/reproducibility_e1_verified.json \
  --container breast-cancer-e2-serving \
  --image breast-cancer-e2:local \
  --url http://127.0.0.1:5001/invocations \
  --output results/inference_latency_e2_verified.json \
  --warmup 20 --iterations 100 --batch-sizes 1,8,32
```

### Verified single-session results (3 November 2026)

All three batches passed prediction parity. There were 20 warm-up iterations
and 100 timed iterations per method/batch, giving **600 timed observations**
(300 local and 300 HTTP), with median and p95 independently recomputed from
raw measurements and confirmed to match the recorded statistics.

| Batch size | Local median (ms) | HTTP median (ms) | Local p95 (ms) | HTTP p95 (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1  | 3.658 | 9.791 | 4.611 | 31.630 |
| 8  | 3.641 | 9.461 | 4.130 | 12.852 |
| 32 | 3.823 | 10.676 | 8.865 | 31.820 |

The local/HTTP measurement scopes differ; these numbers do not isolate HTTP
transport cost or imply a model-level speedup/slowdown. This is one local
measurement session without concurrency, independent machines, or confidence
intervals across sessions. Do not generalize the results to production SLAs.

The public evidence file is `results/inference_latency_e2_public.json`, a
byte-identical copy of the locally verified report after checks for local
absolute-path markers. The local smoke-test and verified reports are retained
separately. The reported model is E1 run
`e89cf131e5f7419fb373212c29b0fba0`, whose `model.pkl` SHA-256 is
`9c8eae1371e50141ee21cd46c5a2d515b9edbf1fded97a94d375d58a73050265`.

An atomic local JSON report captures both summary statistics and **raw sample
latencies** for auditing. Its model path is repository-relative; no absolute
user home path is intentionally stored. Verify the report and redact any
machine-sensitive data before selecting a publication-ready subset.

When done, stop only this named benchmark container:

```bash
docker rm -f breast-cancer-e2-serving
```

Keep the verified E1 artifacts, report, and container build evidence until E2
has been reviewed and the report is committed. Do not commit raw training
artifacts, and do not change the production model or serving endpoint.
