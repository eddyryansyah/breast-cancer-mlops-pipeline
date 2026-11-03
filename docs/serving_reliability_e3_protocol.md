# E3 — Local serving reliability, input rejection, and restart recovery

## Research question

When the **exact** Docker image used for E2 serves the **exact** E1 model,
can the local MLflow `/invocations` endpoint handle sequential valid requests,
reject malformed requests without becoming unavailable, and recover after an
explicit container restart while preserving prediction parity?

## Boundaries and safeguards

- E3 runs **only** against `http://127.0.0.1:5001/invocations` and a specifically
  named `breast-cancer-e3-serving` container, using `breast-cancer-e2:local`.
  It must never touch remote Hugging Face/Docker Hub/GHCR/production endpoints.
- Validate the running container image ID against the verified E2 evidence
  and the E1 model `model.pkl` and dataset checksums before probing.
- Restarts are **opt-in**, using `--enable-restart`. Only the named local E3
  container is restarted by the runner; it never kills another container,
  changes production configuration, or modifies the trained model.
- All inference probes use selected records from the public 569-row dataset,
  eight total: four malignant (0), four benign (1). They exclude `target` in
  HTTP requests. No out-of-sample accuracy or clinical claims are made.
- Malformed probes: malformed JSON, unsupported root list, missing required
  feature, inconsistent row length, and non-numeric feature. Record actual HTTP
  status: do **not** pre-assume a particular 4xx or 5xx. The service must remain
  healthy and continue returning the reference predictions after each probe.
- A 5xx on malformed input counts as rejection but flags a server-side error;
  inspect this separately rather than claiming clean client-side validation.
- Restart timing starts **before** `docker restart` and ends after `/ping` first
  returns 200 and a valid request returns matching predictions. This includes
  Docker CLI/shutdown/startup overhead and polling; **not** a precise measure of
  actual network outage duration, MTTR over incidents, or production availability.
- This is single-host, sequential reliability testing, not concurrency/load,
  chaos testing of the production service, or a statistical SLA guarantee.

## Preconditions

- E1 local evidence: `results/reproducibility_e1_verified.json` and
  `.research_artifacts/e1/` are intact. Do not delete them.
- E2 public evidence: `results/inference_latency_e2_public.json` exists.
- `breast-cancer-e2:local` image matches the immutable image ID in E2 evidence.
- Port 5001 is unused; avoid starting E3 if another service owns it.
- Run unit/regression tests before using the live container.

Start *only* the dedicated research container (after verifying the port):

```bash
docker run --detach --rm --name breast-cancer-e3-serving \
  -p 127.0.0.1:5001:8080 breast-cancer-e2:local
```

Smoke run, with only one explicitly enabled restart:

```bash
.venv/bin/python -m experiments.serving_reliability_e3 \
  --valid-requests 5 --restarts 1 --enable-restart \
  --output results/serving_reliability_e3_smoke.json
```

After reviewing the smoke output, the verified experiment is:

```bash
.venv/bin/python -m experiments.serving_reliability_e3 \
  --valid-requests 50 --restarts 3 --enable-restart \
  --output results/serving_reliability_e3_verified.json
```

Both runs verify the reference model and check valid response parity. The
resulting JSON contains per-request success counts/latencies, per-invalid-case
HTTP status and post-request health, and per-restart readiness and first usable
prediction timings. The runner writes partial JSON evidence if a phase fails.
The official report must be inspected before any numerical publication claims.

Stop the dedicated container after experiments:

```bash
docker stop breast-cancer-e3-serving
```

Preserve raw research evidence locally. Before publishing a public copy of the
report, check for absolute paths and retain observed 4xx/5xx distinctions.
No production code, model or CI workflow needs modification for this stage.

## Verified single-host results (3 November 2026)

The official E3 run used the canonical E1 model artifact and the **same local
Docker image as E2**, with immutable image ID
`sha256:56783f46aab5922cdf500cf0662acc0228011b3785ecceabd97edb09634a4273`.
The reported model `model.pkl` SHA-256 is
`9c8eae1371e50141ee21cd46c5a2d515b9edbf1fded97a94d375d58a73050265`.
Provenance was checked against the E1/E2 reports.

- Unit tests: **30 passed**. Full regression suite: **97 passed**, with one
  non-blocking MLflow `datetime.utcnow()` deprecation warning.
- Smoke run: **5/5** valid HTTP requests succeeded, **5/5** malformed cases
  returned HTTP 400, and **1/1** container restart recovered; the observed
  time to first validated prediction after restart was **7,418.519 ms**.
- Verified run: **50/50** sequential valid requests succeeded with prediction
  parity; **5/5** malformed cases were rejected with HTTP 400, followed by a
  healthy endpoint and a valid matching prediction. No HTTP 5xx response was
  observed for those malformed requests. **3/3** intentional container
  restarts recovered and passed post-restart prediction parity.

### Observed restart recovery timing (verified run)

| Statistic | Duration (ms) |
| --- | ---: |
| Minimum | 5,944.464 |
| Median | 6,335.819 |
| Mean | 6,483.953 |
| Sample p95 (linear interpolation) | 7,088.001 |
| Maximum | 7,171.577 |
| Population standard deviation | 511.801 |

The three restart timings measure elapsed time from initiating the local
`docker restart` command until `/ping` returns HTTP 200 **and** a valid
prediction matches the canonical model. They include Docker command and
container startup overhead, readiness polling, and the validation request.
These timings do **not** isolate the actual service outage, establish
production mean time to recovery (MTTR), or constitute a reliable population
p95 estimate. Similarly, 50 successful sequential requests do **not** imply
100% availability or an SLA. The experiment involved only one local host,
one container, one client, and no concurrent traffic or independent sessions.

### Evidence and audit trail

- Publication evidence: `results/serving_reliability_e3_public.json`.
- Local verified evidence (not committed):
  `results/serving_reliability_e3_verified.json`.
- Local smoke evidence (not committed):
  `results/serving_reliability_e3_smoke.json`.
- The verified and public reports were checked to be byte-identical,
  retaining **50 raw valid-request latencies, 5 invalid-case records, and
  3 restart recovery records**. The reported SHA-256 is
  `7aafecdf812b558fcdeae26e03bc571c09279f6155b4d526e5e5082abb591aee`.
- Initial screening found no specified absolute-home-path or credential-token
  markers; manually review metadata before the public report is committed.

This is operational evidence about the tested local MLflow serving setup,
not clinical or deployment-wide reliability evidence.
