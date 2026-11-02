# E1 — Controlled MLOps training reproducibility

## Research question

When the same production MLflow Project, processed dataset, and installed
Python dependencies are used repeatedly, does the training pipeline execute
successfully and produce consistent metrics, model bytes, and predictions?

## Scope and limitations

- **Five independent runs** by default (`--repeats 5`). Each run gets its own
  workspace, fresh copied production training source/dataset, and isolated
  MLflow tracking directory. Stale `mlruns` and `model.joblib` files are excluded.
- **Same installed environment** across runs (`env_manager="local"`): this is
  controlled *re-execution*, **not** five fresh dependency installations,
  independent machines, or full clean Git clones. Those claims require another
  explicitly specified protocol and additional evidence.
- The model retains the existing fixed random state, stratified 80/20 split,
  feature processing, and hyperparameters. E1 measures operational repeatability,
  **not** out-of-sample generalization or clinical validity.
- Class mapping: `0=malignant`, `1=benign`. Original MLflow precision, recall,
  and F1-score use `pos_label=1` (benign). The class-specific research results
  remain in `results/model_evaluation.json` (Stage 1).
- Fingerprint predictions use *all 569 dataset records* exclusively to compare
  model outputs across runs; do **not** report them as test-set accuracy.

## Execution (repository root, existing `.venv`)

Run the new fast unit tests first:

```bash
.venv/bin/python -m pytest tests/test_reproducibility_e1.py -q
.venv/bin/python -m pytest tests -q
```

The runner temporarily prepends the directory of its own Python interpreter to
`PATH` when launching each synchronous MLflow Project. This prevents MLflow's
`python modelling.py` from accidentally selecting the global/pyenv interpreter
when you invoke `.venv/bin/python` without activating the environment. The
subprocess interpreter is checked and recorded in each run. This is still the
**same installed environment** for all runs, not five clean installations.

Then run the experiment on your own machine, after verifying the Python
interpreter and disk space. It creates per-run model/MLflow artifacts:

```bash
.venv/bin/python -m experiments.reproducibility_e1 --repeats 5
```

If an earlier E1 batch failed due to an interpreter mismatch, preserve it as
negative evidence. Use a different output path for the corrected five-run batch:

```bash
.venv/bin/python -m experiments.reproducibility_e1 --repeats 5 --output results/reproducibility_e1_verified.json
```

The default machine-readable summary is saved to
`results/reproducibility_e1.json`, or the custom `--output` file when specified.
Individual MLflow tracking directories and model artifacts remain under
`.research_artifacts/e1/batch_<timestamp>_<suffix>/run_*/` locally.
This large directory is ignored by Git; do not include raw model files in the PR.

## Evidence to review

- Five run IDs and terminal status (`FINISHED` or failure).
- Per-run MLflow accuracy, precision, recall, F1-score, durations, artifact
  model.pkl SHA-256, and prediction SHA-256 for a fixed dataset ordering.
- Source-file SHA-256 and dataset SHA-256; Python, MLflow, pandas, NumPy,
  scikit-learn, and joblib versions; OS/platform and Git commit.
- Number of successful runs, metric dispersion (range / population standard
  deviation), model hash parity, prediction hash parity, and timing summary.
- On failures, the script retains each run's error in the JSON report and
  exits nonzero after completing all requested attempts.

## Reporting standards

Interpret exact prediction agreement as consistency under this fixed setup;
identical metrics with a fixed seed are expected and do **not** establish
robustness to input drift, environment variation, or randomness. Interpret
model byte-level hashes separately from functional prediction equality.
MLflow tracking/model artifact timestamps and metadata may differ by run.
No paper-ready numerical claims may be made until the actual five-run report
is inspected and cross-checked against retained artifacts.
