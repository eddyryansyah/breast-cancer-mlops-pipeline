# E5 — Controlled Provenance Fault-Injection Evaluation

**Status:** Draft protocol v0.1 — 3 November 2026. **Not a preregistration of P0–F4:** the pilot cases and V3 run-ID check were designed and executed before this document. The additional challenge cases below are specified *before their execution*; record any later changes as protocol deviations.
**Project:** Breast Cancer MLOps Pipeline
**Base commit:** `9215659` (merged E4, PR #24)
**Working branch:** `research/provenance-fault-injection-e5`

## 1. Research objective and boundary

**Primary question:** For a fixed E1 model artifact and processed dataset, which controlled integrity faults are detected by (V1) structural artifact validation, (V2) existing E1 hash-based provenance validation, and (V3) V2 plus a check that `MLmodel.run_id` matches the E1 report's canonical run ID?

**Secondary question:** Which faults remain undetected, especially changes to other metadata or a co-modified reference report?

This is a small, deterministic **fault-detection case study**, not a security assessment, adversarial robustness benchmark, vulnerability exploit, clinical validation, or claim of generalized detection rates. The validators do not share an equal intended scope: V1 checks artifact structure, whereas V2/V3 also check model and dataset provenance. Differences therefore reflect added validation coverage, not a universally superior algorithm.

## 2. Fixed references, artifacts, and isolation

- Canonical E1 MLflow run: `e89cf131e5f7419fb373212c29b0fba0`.
- Canonical `model.pkl` SHA-256: `9c8eae1371e50141ee21cd46c5a2d515b9edbf1fded97a94d375d58a73050265`.
- Processed dataset SHA-256: `9b9e3a2fe53a2264f7e756aff00ab883450186c47bfb2027b4d90ca51d23347d`.
- E1 local source report: `results/reproducibility_e1_verified.json`; processed data: `data/processed/breast_cancer_preprocessing.csv`.
- Positive-control copy: `.research_artifacts/e5/positive_control/model/` and `.research_artifacts/e5/positive_control/dataset.csv`. It was verified byte-identical to E1 before any injection.
- Fault copies: one new directory per case inside `.research_artifacts/e5/faults/`. They must be ignored by Git and **never** modify the positive control or original E1 artifact/data/report. Reference-report mutations must occur in memory or in a separately isolated ignored copy.
- No retraining, model loading for malicious pickle inputs, inference, HTTP serving, Docker build, or production deployment is required for these checks.
- Fail rather than overwrite an existing scenario directory. Before and after each case, verify the original reference hashes and positive-control hashes. Verify the intended fault actually occurred and that unrelated files did not change.

## 3. Validator definitions — freeze before the challenge set

- **V1, structural:** `scripts.run_mlflow_project.validate_model_directory(model_dir)`. ACCEPT when the directory and `MLmodel` file exist; this validator does not inspect model bytes, dataset, YAML content, or MLflow run-ID consistency.
- **V2, hash-based provenance:** `experiments.inference_latency_e2.verify_e1_reference(report, root, dataset_path)`, with the E1 report's first run pointing to the *isolated copy*. Checks the E1 report's five completed runs and model-hash-identical flag, artifact files, the SHA-256 of `model.pkl` against the report, and dataset SHA-256 against the report. It does **not** hash `MLmodel` or validate its `run_id` field.
- **V3, metadata-aware (pilot prototype):** first run V2, then parse the isolated `MLmodel` with `yaml.safe_load`, require a mapping, and require its top-level `run_id` to equal the canonical run ID in the E1 reference. Reject on YAML/format errors. **V3 is not a complete metadata-integrity validator.** Freeze this implementation for the challenge-set evaluation; any stronger version must be named V4 and evaluated separately.

For every case, **ACCEPT** means the validator completes without its defined validation errors. **REJECT** means it raises an expected validation error. Record exception type and message for REJECT. Unexpected exceptions (for example, an assertion in the test harness) indicate **TEST ERROR**, not an automatic correct rejection.

Do not conflate the structural check with functional model validity, or a reference-report hash comparison with cryptographic authenticity of the reference report.

## 4. Previously executed exploratory pilot — observed, not preregistered

| Case | Injected condition | V1 observed | V2 observed | V3 observed |
|:---|:---|:---:|:---:|:---:|
| P0 | Clean copied artifact and dataset | ACCEPT | ACCEPT | ACCEPT |
| F1 | Remove `MLmodel` from fault copy | REJECT | REJECT | REJECT |
| F2 | Flip one byte in copied `model.pkl` | ACCEPT | REJECT | REJECT |
| F3 | Change exactly one numeric feature cell in copied CSV | ACCEPT | REJECT | REJECT |
| F4 | Change only top-level `MLmodel.run_id` to a different valid string | ACCEPT | ACCEPT | REJECT |

**Pilot detection counts among F1–F4:** V1 **1/4**, V2 **3/4**, V3 **4/4**. Positive control P0 was accepted by all three. These are descriptions of four chosen faults; they are **not statistical sensitivity, confidence intervals, or estimates of field reliability**. V3 was deliberately designed after seeing F4, so its pilot 4/4 is not an independent evaluation.

F4 first encountered a test-harness error: an unquoted all-zero run ID was parsed as an integer by YAML. The unsuccessful attempt made no fault directory. The corrected test used 32 `f` characters as a string and passed; document both runs in provenance of experimental development.

## 5. Additional challenge set — specified before execution

Each new condition starts from a fresh, byte-identical copy of P0. The expected decisions below follow the **frozen code contracts**, not measured results. Do not re-label an unexpected result as expected post hoc.

| ID | Controlled mutation | Expected V1 | Expected V2 | Expected V3 | Intended diagnostic |
|:---|:---|:---:|:---:|:---:|:---|
| C1 | Replace `MLmodel` with syntactically invalid YAML while keeping the file present; `model.pkl` and dataset unchanged | ACCEPT | ACCEPT | REJECT | V3 parsing adds coverage |
| C2 | Add one harmless, previously absent, top-level YAML metadata field to `MLmodel`, preserving existing fields | ACCEPT | ACCEPT | ACCEPT | V3 does **not** protect all metadata bytes |
| C3 | Delete top-level `MLmodel.run_id`, preserving other valid YAML keys | ACCEPT | ACCEPT | REJECT | Run-ID missing-value handling |
| C4 | Change the canonical run ID in the **in-memory E1 report** only; model, dataset and `MLmodel` untouched | ACCEPT | ACCEPT | REJECT | The validators' treatment of reference/report disagreement |
| C5 | Flip one byte in copied `model.pkl` **and update its expected SHA-256 in the isolated report**; do not change `MLmodel` or dataset | ACCEPT | ACCEPT | ACCEPT | Hash validation depends on trust in reference |
| C6 | Change one CSV feature digit **and update its expected dataset SHA-256 in the isolated report**; model and `MLmodel` untouched | ACCEPT | ACCEPT | ACCEPT | Dataset hash also depends on reference trust |

**Construction/validity gates:** C1 must retain a present `MLmodel` file but fail YAML parsing deterministically. C2 must produce a valid mapping, change exactly one added metadata field, and leave `run_id` unchanged. C3 must yield valid YAML without `run_id`. C4 must mutate only `runs[0].mlflow_run_id` in the *copy* of the reference report. For C5/C6, change exactly one byte or CSV feature digit and update only the corresponding expected SHA value; other invariants must hold. Never load or unpickle a mutated model. Verify the test fixture still meets the stated preconditions before reporting any validator outcomes.

**Interpretation limits for C5–C6:** They model **untrusted reference co-mutation**, not compromise of an authenticated registry or a practical attack against one. V2/V3 can accept internally self-consistent yet untrusted evidence; this does not show a hash collision or a failure of SHA-256.

## 6. Measurement and reporting plan

For every P0/F1–F4/C1–C6 scenario and V1/V2/V3 validator, record:

- Scenario ID, exact intended mutation, input SHA-256 values or mutation description, and whether the scenario's invariant checks passed.
- Decision: `ACCEPT`, `REJECT`, or `TEST_ERROR`; on REJECT/TEST_ERROR, exception type and diagnostic message. Report `TEST_ERROR` separately, not as successful detection.
- Expected decision (frozen before the challenge run), observed decision, and exact match indicator.
- Confirmation that the original artifact, reference report, and positive control remain unchanged; local environment and Python package versions relevant to parsing.

Report a **scenario-by-validator detection matrix** and **counts by suite** (pilot F1–F4 versus challenge C1–C6), without combining them into an unsupported population detection-rate estimate. P0 tests false rejection of one intact reference copy only; one accepted control cannot establish a general false-positive rate. No confidence intervals or inferential significance tests are justified from these hand-crafted cases.

If timing overhead is measured later, define a distinct timing protocol with warm-ups, repetitions, hardware and filesystem conditions, measurement scope, paired ordering, and descriptive quantiles **before** measurement. Do not infer overhead from the pilot fault outcomes.

## 7. Reproducible implementation and quality gates

1. Capture the existing P0/F1–F4 pilot as **exploratory evidence**, including the F4 YAML-type correction; verify it against an executable harness rather than transcribing expected results as observations.
2. Implement deterministic case builders and validators in `experiments/provenance_fault_injection_e5.py`. Keep the evaluator read-only with respect to E1 sources; build case copies only in an ignored sandbox or `tempfile.TemporaryDirectory`.
3. Write unit tests in `tests/test_provenance_fault_injection_e5.py`, using synthetic, disposable fixtures that can run in GitHub CI **without access to the private/local E1 MLflow model**. Do not assert E1 artifact paths in generic CI tests.
4. For the full local E1-based replication, require the original verified E1 report and model and enforce path-containment, no-overwrite, and integrity gates. Run the declared C1–C6 challenge set exactly once for the initial recorded measurement; preserve subsequent reruns with unique run identifiers.
5. Save the full verified local output under `results/provenance_fault_injection_e5_verified.json`; publish a reviewed, location-safe copy under `results/provenance_fault_injection_e5_public.json`. Use repository-relative model paths in public evidence. Audit for private absolute paths, credentials, and unexpectedly large serialized objects.
6. Run targeted tests and the full regression suite. Audit code, report, test outcomes and any deviations before selectively staging only the E5 protocol, implementation, tests and approved public evidence (`git add` with explicit paths; never `git add .`).
7. Open a PR against `main`, verify all GitHub Actions and deployment-related workflows, and only then merge. No direct changes to production-serving logic are needed.

## 8. Threats to validity, scope, and contribution

- **Selection and design bias:** P0–F4 were chosen while developing V3; the challenge set is still hand-designed and is not a statistically representative sample of real failures.
- **Reference integrity:** V2/V3 compare hashes against a potentially mutable report. Neither implements signed, externally anchored manifests; the C5/C6 cases deliberately expose this trust assumption.
- **Metadata coverage:** V3 checks only top-level `run_id` and basic YAML parseability. Other meaningful metadata can be modified without rejection; full MLflow schema validation is not claimed.
- **No behavioral check:** An artifact may satisfy structural/provenance checks yet fail to deserialize or predict correctly. Conversely, one-byte pickle corruption may not be loadable. This evaluation does not test model execution.
- **Single environment and model:** Evidence comes from one E1 model, one dataset and local file copies, not a representative fleet, multi-platform benchmark, or clinical setting.
- **Small purposeful suite:** Counts are diagnostic coverage of explicit conditions, not prevalence-weighted field failure detection, security efficacy, or production SLA.

**Candidate contribution:** An auditable, reproducible comparison of three *layered validation contracts* across controlled model, data and metadata faults, reporting both detections and blind spots. Whether this is sufficiently novel for a target journal must be assessed against prior work; E5 results alone do not prove novelty or clinical suitability.

## 9. Protocol change log

- **2026-11-03, draft v0.1:** documented completed pilot P0–F4 separately from a new, prospective C1–C6 challenge set; defined immutable validator contracts, expected outcomes, guardrails and reporting requirements. **No C1–C6 outcomes are reported in this version.**

- **2026-11-03, post-execution evidence addendum v0.2:** recorded the completed E5 run and independent audits in Section 10. This addendum does not retroactively change the v0.1 pilot/challenge design, validator definitions, or expected decisions.

## 10. Verified E5 execution — evidence addendum (3 November 2026)

### 10.1 Execution and evidence provenance

The initial real-E1-artifact E5 evaluation completed **11 scenarios**: five pilot/control scenarios (P0, F1–F4) and six previously specified challenge scenarios (C1–C6), each evaluated by three frozen validator contracts (**33 decisions**). The pilot outcomes are **replications of an exploratory development set**; C1–C6 are the separately specified challenge set, not a statistically independent sample of naturally occurring faults.

- Local verified report: `results/provenance_fault_injection_e5_verified.json` (retained locally; **not** for Git).
- Public report: `results/provenance_fault_injection_e5_public.json` (reviewed copy, **byte-identical** to the verified report).
- SHA-256 of **both** reports: `245e4693bda50901b0f859c640261dad32a65e9ac66cfd3f16a637e74fc31cf5`.
- Local ignored run workspace: `.research_artifacts/e5/runs/initial_e1_evaluation`.
- Canonical E1 MLflow run: `e89cf131e5f7419fb373212c29b0fba0`.
- Canonical E1 `model.pkl` SHA-256: `9c8eae1371e50141ee21cd46c5a2d515b9edbf1fded97a94d375d58a73050265`.
- Processed dataset SHA-256: `9b9e3a2fe53a2264f7e756aff00ab883450186c47bfb2027b4d90ca51d23347d`.
- Outcome deviations from the frozen decision table: **0**; unexpected `TEST_ERROR` results: **0**. The absence of deviations means the implementation behaved as predicted, **not** that all faults were detected.

### 10.2 Observed scenario-by-validator decisions

These decisions were read from and independently recomputed against the verified JSON. `ACCEPT` means the given validator contract accepted the input; it does **not** imply the model is executable, clinically valid, or the reference is authentic.

| Set | Case | Controlled condition | V1 observed | V2 observed | V3 observed |
|:---|:---|:---|:---:|:---:|:---:|
| Pilot/control | P0 | Unmodified positive-control copy | ACCEPT | ACCEPT | ACCEPT |
| Pilot | F1 | Missing `MLmodel` file | REJECT | REJECT | REJECT |
| Pilot | F2 | One modified byte in `model.pkl` | ACCEPT | REJECT | REJECT |
| Pilot | F3 | One modified CSV feature digit | ACCEPT | REJECT | REJECT |
| Pilot | F4 | Mismatched `MLmodel.run_id` | ACCEPT | ACCEPT | REJECT |
| Challenge | C1 | Invalid YAML in `MLmodel` | ACCEPT | ACCEPT | REJECT |
| Challenge | C2 | Harmless additional YAML metadata field | ACCEPT | ACCEPT | ACCEPT |
| Challenge | C3 | Missing `MLmodel.run_id` | ACCEPT | ACCEPT | REJECT |
| Challenge | C4 | Changed run ID in in-memory reference report | ACCEPT | ACCEPT | REJECT |
| Challenge | C5 | Mutated `model.pkl` plus matching updated reference hash | ACCEPT | ACCEPT | ACCEPT |
| Challenge | C6 | Mutated dataset plus matching updated reference hash | ACCEPT | ACCEPT | ACCEPT |

**Descriptive rejection counts:** For the four intentionally faulty pilot cases F1–F4, V1 rejected **1/4**, V2 **3/4**, and V3 **4/4**; all accepted the clean P0. For the six challenge cases C1–C6, V1 rejected **0/6**, V2 **0/6**, and V3 **3/6**. C2 is a deliberately benign metadata extension, and C5/C6 model co-modification of the reference used for verification. Thus **3/6 is a rejection count, not a statistically meaningful sensitivity estimate or an indication that all six cases should be rejected**.

### 10.3 What the results establish — and do not establish

- F2 and F3 demonstrate that V2's model/data hash comparisons detect two changes that V1's artifact-presence check does not inspect. This is a **difference in validation scope**, not an apples-to-apples comparison of equivalent-function detectors.
- F4, C1, C3 and C4 demonstrate the extra coverage of V3's narrow YAML parsing and top-level `run_id` consistency requirement. V3 was developed after observing F4 and must not be represented as prospectively validated on the pilot.
- C2 confirms that an additional benign metadata field does not trigger these validator contracts; this acceptance is expected and should not be counted as a missed harmful fault.
- C5 and C6 reveal a **trust-boundary limitation**: if the model/dataset and its expected hash in an *unauthenticated* reference are changed together, a matching hash does not establish provenance authenticity. These are **not** SHA-256 collisions and do not test signed manifests or a trusted registry.
- V3 verifies only basic YAML parseability, mapping structure and the `run_id` relationship after V2's checks. It does **not** ensure full `MLmodel` semantic integrity, artifact deserialization, successful inference, security against a malicious actor, clinical safety, or deployment reliability.
- The scenarios are intentionally hand-crafted on one local model/dataset. Do not calculate inferential confidence intervals, general-purpose detection rates, or production reliability guarantees from the counts above.

### 10.4 Verification and publication controls

The independent E5 evidence audit checked all **11 scenario directories**, **33 observed/expected validator decisions**, changed-file manifests, per-case file SHA-256 values, reference-change flags, suite summary counts, and unchanged original E1 inputs and positive-control copy. This audit passed. The initial local E5 implementation had **24 passing unit tests**, with **167 passing tests** in the full repository suite (one existing non-blocking MLflow `datetime.utcnow()` deprecation warning). These are local test results and do not imply GitHub CI has yet run the E5 branch.

The public-report safety screen checked for macOS/Linux/Windows absolute home paths, common access-token patterns, private-key markers, file URIs, and an unexpected report size. No patterns were found, and the public JSON was confirmed **byte-identical** to the 23,772-byte verified report. Pattern checks do not constitute a guarantee that all sensitive material is absent. The protocol and source evidence were not edited to match observed results; this section is a **post-execution evidence addendum**, distinct from the original challenge design.
