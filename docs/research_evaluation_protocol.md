# Research Evaluation Protocol — Baseline v1.0.0

## Status and scope

This is a **proposed research addition**, not a change to production model selection,
container serving, or Hugging Face deployment. Preserve the existing `v1.0.0`
Git tag as the engineering baseline. The code reports metrics from the same
single stratified holdout split used by `mlproject/modelling.py`.

## Why this is needed

The Wisconsin Diagnostic dataset uses `0 = malignant` and `1 = benign`.
The existing `precision_score`, `recall_score`, and `f1_score` calls use
`pos_label=1` implicitly, thus reporting metrics for **benign**. They do not
report malignant sensitivity. This distinction must be explicit in a paper.

## Run from the GitHub repository root

1. Create a new branch, e.g. `research/explicit-class-evaluation`.
2. Apply the accompanying patch and install the existing project dependencies.
3. Run tests and the evaluation:

```bash
python -m pip install -r mlproject/requirements.txt
python -m pip install -r tests/requirements.txt
python -m pytest tests/test_research_evaluation.py -q
python -m experiments.evaluate_baseline \
  --dataset data/processed/breast_cancer_preprocessing.csv \
  --output results/model_evaluation.json
```

The JSON report includes dataset SHA-256, original binary metrics, class-specific
precision/recall/F1, confusion matrix, malignant false-negative count/rate,
malignant ROC-AUC, fixed split metadata and library versions.

## Interpretation rules

- **Do not** interpret `recall_score(y_test, y_pred)` as malignant sensitivity.
- **Do not** treat probabilities in the Gradio demo as validated clinical risk.
- **Do not** treat demo inferences from the full dataset as independent test results.
- **Do not** use this single holdout result as the sole statistical evidence for a paper.
- **Do** preserve the generated report together with the associated commit and dataset hash.

## Completion gate for this stage

- All new unit tests pass.
- Report is produced against the repository's actual processed CSV.
- Confusion matrix and per-class metrics are reviewed for consistency.
- Raw report, dataset hash and environment versions are retained.
- No production serving logic, CI/CD secrets, tags or release artifacts are changed.

## Next study design (not yet implemented)

Use repeated stratified evaluation and/or nested cross-validation as appropriate,
compare with a clearly defined baseline, and measure clean-run reproducibility,
model parity across deployment surfaces, latency, resource usage and failure
handling. Pre-register the metric definitions and run protocol before collecting
these additional results.
