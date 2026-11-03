# E4 — Controlled input perturbation robustness protocol

## Research question

For the **fixed, E1-verified MLflow model**, how do synthetic Gaussian
perturbations to the held-out test features affect prediction stability,
accuracy, and class-specific recall? This evaluates *sensitivity to a defined
synthetic perturbation*, not clinical robustness, adversarial security, or
real-world data drift.

## Preregistered reference and data partition

- Canonical E1 MLflow run: `e89cf131e5f7419fb373212c29b0fba0`.
- Canonical E1 `model.pkl` SHA-256:
  `9c8eae1371e50141ee21cd46c5a2d515b9edbf1fded97a94d375d58a73050265`.
- Load **this model artifact**, not a newly fitted or reconstructed estimator.
  Require verification against the local E1 verified report and processed
  dataset SHA-256 before any measurement.
- Dataset: `data/processed/breast_cancer_preprocessing.csv`, 569 rows,
  30 numeric input features, and a separate `target` column.
- Recreate the exact training split with scikit-learn
  `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)`.
  Training: 455 observations (170 malignant, 285 benign); testing: 114
  observations (42 malignant, 72 benign). Keep original feature names, order,
  and test-row order; never include `target` among model inputs.
- Labels: `0=malignant`, `1=benign`. Here, *malignant recall* has
  `pos_label=0`; default binary precision/recall with `pos_label=1` must not
  be mislabeled as malignant metrics.
- Verify no row-index overlap and that the MLflow pipeline's
  `StandardScaler.mean_` and `scale_` match the training-only feature means
  and population standard deviations (`ddof=0`). Fail if any training
  feature standard deviation is zero or nonfinite.
- Before perturbation, independently verify the frozen model's held-out
  confusion matrix (rows=true class, columns=predicted; labels `[0,1]`):
  `[[39, 3], [2, 70]]`. Expected baseline: accuracy `109/114`,
  malignant recall `39/42`, benign recall `70/72`, and 3 malignant records
  predicted benign. Abort rather than proceed with a different baseline.

## Synthetic perturbation design (fixed in advance)

- Perturb **only X_test**, without refitting any model component, and without
  modifying training data, labels, or saved dataset/model artifacts.
- Set per-feature sigma to the **training-only population standard deviation**
  of that feature in its original (unscaled) units.
- For each seed `4200, 4201, ..., 4209`, independently draw a matrix
  `Z` of shape `(114,30)` using `numpy.random.default_rng(seed).standard_normal`.
  Create each noisy matrix as
  `X_test_noisy[:, j] = X_test[:, j] + alpha * train_std[j] * Z[:, j]`.
- Predefined alpha levels: `0.00` (unperturbed baseline), `0.05`, `0.10`,
  and `0.20`. Run the baseline **once**; for each nonzero alpha run 10 seeds,
  giving **30 perturbed evaluations** plus the common baseline.
- Reuse the **same Z within a seed across all three nonzero alpha levels**.
  This creates a paired noise design, not 30 independent test datasets.
- Do not clip negative or out-of-range synthetic feature values unless
  a separate, explicitly documented sensitivity experiment is created.
  Report the number of perturbed feature cells below zero and outside the
  observed training min/max as diagnostics; such observations may be
  physically implausible. This experiment is an unconstrained additive-noise
  stress test, not a simulation of real measurement error.
- Preserve original row order, feature names, and full floating-point
  precision. Generate all noise only after the E1 reference, split, and
  baseline checks pass.

## Measures and computation

For the single baseline and **each** `(alpha, seed)` condition, compute:

1. Accuracy against the unchanged held-out labels.
2. Malignant recall (`pos_label=0`) and benign recall (`pos_label=1`).
3. Confusion matrix with explicit label order `[0, 1]`.
4. Malignant-to-benign errors (`true=0, predicted=1`) and
   benign-to-malignant errors (`true=1, predicted=0`).
5. Prediction flip rate relative to the **unperturbed predictions from the
   same 114 held-out rows**, plus raw flip count. A prediction flip is not
   necessarily a new error; distinguish it from classification correctness.
6. Count of previously correctly classified malignant rows that become
   predicted benign. This is a descriptive error-transition count, not a
   patient-level harm estimate.
7. Nonfinite-value checks for every perturbation and prediction-label checks
   (only integer class `0` or `1`).

For each nonzero alpha, aggregate the 10 per-seed evaluations with a mean,
median, population standard deviation, minimum, and maximum for the numeric
metrics. Keep all **30 per-seed observations** in the JSON report so reported
aggregates can be independently recomputed. Do not treat 10 perturbation
seeds as 10 independent patient cohorts, calculate clinical risk, or claim
inferential confidence intervals from these repeated same-test-set results.

## Operational reproducibility and evidence

- Implement in `experiments/input_robustness_e4.py`, with separate unit tests
  in `tests/test_input_robustness_e4.py`. Use existing E1 artifact verification
  from `experiments/inference_latency_e2.py` when appropriate.
- Save the verified local report as
  `results/input_robustness_e4_verified.json`; publish only a reviewed,
  location-safe copy as `results/input_robustness_e4_public.json`.
  Keep temporary smoke-test reports and E1/E2/E3 local reports untracked.
- Store reproducibility metadata: protocol version, UTC timestamp, E1 run ID,
  `model.pkl` SHA-256, dataset SHA-256, split parameters, exact alpha list,
  exact seed list, version information, baseline confusion matrix, raw
  per-condition results, descriptive summaries, and perturbation diagnostics.
- Keep model paths repository-relative in publication evidence. Audit for
  private paths and credentials before staging. Preserve verified raw metrics
  in public evidence; do not substitute manually rounded values.
- Run new tests and then `python -m pytest tests -q` before staging exactly
  the E4 script, tests, protocol, and reviewed public report.
- Do not retrain, change production serving/deployment, or merge until
  research evidence and GitHub CI have been reviewed.

## Verified results (3 November 2026)

The frozen E1 model and held-out data passed artifact, dataset, and
training-only scaler checks. The preregistered experiment completed a single
baseline and **30 perturbation conditions** (10 paired seeds per nonzero alpha),
with **114 held-out predictions per condition**, or 3,420 perturbed predictions.
The reference baseline was accuracy `109/114 = 0.956140350877193` and confusion
matrix `[[39, 3], [2, 70]]` (true/predicted label order `[0, 1]`). No model
refitting occurred.

The table reports descriptive means over the same 10 seeds for each alpha,
not independent patient cohorts or confidence intervals.

| Alpha | Accuracy mean | Malignant recall mean | Prediction flip rate mean |
| ---: | ---: | ---: | ---: |
| 0.00 (single baseline) | 95.614% | 92.857% | 0.000% |
| 0.05 | 95.439% | 92.857% | 0.175% |
| 0.10 | 94.912% | 93.095% | 1.404% |
| 0.20 | 94.211% | 93.571% | 3.333% |

The **mean accuracy** decreased and **mean prediction flip rate** increased
with alpha in this particular synthetic stress test. Malignant recall did not
monotonically decrease: for some seeded perturbations, changes corrected
pre-existing malignant classification errors. A prediction flip can improve
or worsen a prediction, so the mean malignant recall increase must **not** be
read as a clinical benefit. Inspect each condition's confusion matrix,
malignant-to-benign errors, and feature plausibility diagnostics in the raw
report rather than inferring risk from accuracy or flips alone.

The initial report used the misleading key
`newly_correct_malignant_to_benign` for the count of *previously correct
malignant predictions that became benign predictions* after noise. This key
was corrected to `new_malignant_false_negatives` consistently in the code,
tests, and report. The original and corrected JSON differ **only** by that
field-name substitution, without any numeric or prediction changes:

- Pre-correction local report SHA-256:
  `499a1d801a952f24edf141e7fec6eeba4b9878ada6371f936f4ed1912f5d3ee9`.
- Verified corrected and public report SHA-256:
  `faf6f78f6a8a45b2960105f1edfea3b380d9de634b59cd161ffc6ba974a76f79`.
- The pre-correction report is retained locally under
  `.research_artifacts/e4/input_robustness_e4_verified_before_rename.json`;
  it is not intended for Git.
- Verified local evidence: `results/input_robustness_e4_verified.json`.
  Publication evidence: `results/input_robustness_e4_public.json`, verified
  **byte-identical** to the corrected report. The 30 condition outputs,
  baseline predictions, and unrounded numerical results are retained.
- Post-correction validation: 46 E4 unit tests passed; full suite:
  **143 passed** (one non-blocking MLflow deprecation warning). The raw
  prediction/metric audit passed for all 30 conditions, and the corrected
  evidence audit confirmed the key rename was the only change.

These numbers describe a fixed model on one 114-row held-out split with
synthetic independent Gaussian input noise. The 10 seeds reuse **the same
held-out rows**, and each seed reuses paired noise at three amplitudes.
They cannot establish an independent external generalization estimate,
clinical safety, operational data drift, or performance guarantees.

## Interpretation and limitations

This is a single model and single held-out split of the Wisconsin Diagnostic
Breast Cancer dataset. Gaussian independent-feature noise is artificial;
real medical measurement errors can be correlated, biased, censored, bounded,
or caused by collection-process changes. The test set is reused across every
condition, and the chosen perturbation amplitudes are **experimental design
parameters**, not empirically calibrated clinical tolerances. Accuracy,
recall, and flip rates quantify behavior **only under these exact synthetic
conditions**; none establish clinical applicability, safety, robustness to
arbitrary distribution shift, or performance on an independent external set.
