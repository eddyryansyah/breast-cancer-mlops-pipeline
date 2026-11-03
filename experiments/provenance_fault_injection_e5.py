"""E5: controlled provenance fault-injection evaluation (no model deserialization).

Execute only with the verified, local E1 report and an intact, isolated E5
positive control. All scenario mutations occur in new, ignored run folders.
Tests use disposable synthetic fixtures; no E1 model is required in CI.

    .venv/bin/python -m experiments.provenance_fault_injection_e5 \\
      --e1-report results/reproducibility_e1_verified.json \\
      --positive-control .research_artifacts/e5/positive_control \\
      --output results/provenance_fault_injection_e5_verified.json

The P0/F1-F4 pilot was exploratory. C1-C6 expectations are specified in the
reviewed E5 protocol prior to the first E1-based challenge run. This module
implements those frozen contracts, not a general artifact security validator.
"""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
from typing import Callable

import yaml

from experiments.inference_latency_e2 import sha256_file, verify_e1_reference
from scripts.run_mlflow_project import validate_model_directory

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_E1_REPORT = ROOT / "results/reproducibility_e1_verified.json"
DEFAULT_DATASET = ROOT / "data/processed/breast_cancer_preprocessing.csv"
DEFAULT_POSITIVE = ROOT / ".research_artifacts/e5/positive_control"
DEFAULT_OUTPUT = ROOT / "results/provenance_fault_injection_e5_verified.json"
E5_WORKSPACE = ROOT / ".research_artifacts/e5/runs"

PILOT_CASES = ("P0", "F1", "F2", "F3", "F4")
CHALLENGE_CASES = ("C1", "C2", "C3", "C4", "C5", "C6")
ALL_CASES = PILOT_CASES + CHALLENGE_CASES
VALIDATORS = ("V1", "V2", "V3")

# Frozen before the E1-based challenge-run results are collected.
EXPECTED_DECISIONS = {
    "P0": ("ACCEPT", "ACCEPT", "ACCEPT"),
    "F1": ("REJECT", "REJECT", "REJECT"),
    "F2": ("ACCEPT", "REJECT", "REJECT"),
    "F3": ("ACCEPT", "REJECT", "REJECT"),
    "F4": ("ACCEPT", "ACCEPT", "REJECT"),
    "C1": ("ACCEPT", "ACCEPT", "REJECT"),
    "C2": ("ACCEPT", "ACCEPT", "ACCEPT"),
    "C3": ("ACCEPT", "ACCEPT", "REJECT"),
    "C4": ("ACCEPT", "ACCEPT", "REJECT"),
    "C5": ("ACCEPT", "ACCEPT", "ACCEPT"),
    "C6": ("ACCEPT", "ACCEPT", "ACCEPT"),
}
CASE_DESCRIPTION = {
    "P0": "Intact artifact, metadata and dataset",
    "F1": "Remove the copied MLmodel file",
    "F2": "Flip one byte in copied model.pkl",
    "F3": "Change one feature digit in copied CSV",
    "F4": "Change copied MLmodel top-level run_id",
    "C1": "Replace MLmodel with invalid YAML, preserving file presence",
    "C2": "Add one harmless top-level MLmodel metadata field",
    "C3": "Remove top-level MLmodel run_id",
    "C4": "Change E1 report first-run ID in memory only",
    "C5": "Flip model.pkl byte and co-update in-memory expected hash",
    "C6": "Change CSV feature digit and co-update in-memory expected hash",
}
EXPECTED_REJECTION_ERRORS = (
    ValueError, FileNotFoundError, RuntimeError,
    UnicodeError, yaml.YAMLError,
)


def _fake_run_id(canonical: str) -> str:
    return "f" * 32 if canonical != "f" * 32 else "e" * 32


def _load_yaml(path: Path) -> dict:
    result = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("MLmodel metadata is not a mapping")
    return result


def _replace_top_level_run_id(text: str, canonical: str, replacement: str) -> str:
    matches = list(re.finditer(r"(?m)^run_id:[ \t]*([^\r\n]+)$", text))
    if len(matches) != 1:
        raise ValueError("Expected exactly one top-level MLmodel run_id")
    match = matches[0]
    if match.group(1).strip().strip("'\"") != canonical:
        raise ValueError("Unmatched canonical MLmodel run_id")
    return text[:match.start(1)] + replacement + text[match.end(1):]


def _drop_top_level_run_id(text: str, canonical: str) -> str:
    lines = text.splitlines(keepends=True)
    matched = [line for line in lines if re.match(r"^run_id:[ \t]*", line)]
    if len(matched) != 1 or matched[0].split(":", 1)[1].strip().strip("'\"") != canonical:
        raise ValueError("Unmatched canonical MLmodel run_id")
    filtered = [line for line in lines if not re.match(r"^run_id:[ \t]*", line)]
    if len(lines) - len(filtered) != 1:
        raise ValueError("Expected exactly one top-level MLmodel run_id to remove")
    result = "".join(filtered)
    return result


def _flip_one_model_byte(path: Path) -> None:
    content = bytearray(path.read_bytes())
    if len(content) < 3:
        raise ValueError("model.pkl fixture must contain at least three bytes")
    content[len(content) // 2] ^= 0x01
    path.write_bytes(content)


def _mutate_first_feature_digit(path: Path) -> None:
    original = path.read_bytes()
    first_data = original.find(b"\n") + 1
    if first_data < 1 or first_data >= len(original):
        raise ValueError("CSV missing data rows")
    first_delimiter = original.find(b",", first_data)
    if first_delimiter < 0:
        raise ValueError("CSV first feature delimiter missing")
    numeric_positions = [
        i for i in range(first_data, first_delimiter)
        if ord("0") <= original[i] <= ord("9")
    ]
    if not numeric_positions:
        raise ValueError("CSV first feature has no numeric digit")
    result = bytearray(original)
    pos = numeric_positions[-1]
    result[pos] = result[pos] + 1 if result[pos] < ord("9") else ord("8")
    before = list(csv.reader(io.StringIO(original.decode("utf-8"))))
    after = list(csv.reader(io.StringIO(result.decode("utf-8"))))
    if not before or before[0] != after[0] or "target" not in before[0]:
        raise ValueError("CSV header/target invalid")
    if len(before) != len(after) or any(len(a) != len(b) for a, b in zip(before, after)):
        raise ValueError("CSV shape changed")
    target_col = before[0].index("target")
    changes = [(i, j) for i, (a, b) in enumerate(zip(before[1:], after[1:]), 1)
               for j, (x, y) in enumerate(zip(a, b)) if x != y]
    if changes != [(1, 0)] or target_col == 0:
        raise ValueError("Mutation changed more than first feature cell")
    float(before[1][0]); float(after[1][0])
    path.write_bytes(result)
    if sum(x != y for x, y in zip(original, result)) != 1:
        raise AssertionError("CSV mutation must change exactly one byte")


def _snapshot(folder: Path) -> dict[str, str]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Snapshot folder not found: {folder}")
    result = {}
    for path in sorted(folder.rglob("*")):
        if path.is_symlink():
            raise ValueError("Symlinks prohibited in E5 fixture")
        if path.is_file():
            result[path.relative_to(folder).as_posix()] = sha256_file(path)
    return result


def verify_positive(positive: Path, report: dict) -> dict[str, str]:
    """Require clean reference bytes before generating scenario directories."""
    expected_model = report["runs"][0]["model_pickle_sha256"]
    expected_dataset = report["dataset"]["sha256"]
    expected_run_id = report["runs"][0]["mlflow_run_id"]
    model = positive / "model"
    dataset = positive / "dataset.csv"
    if sha256_file(model / "model.pkl") != expected_model:
        raise ValueError("E5 positive model checksum mismatch")
    if sha256_file(dataset) != expected_dataset:
        raise ValueError("E5 positive dataset checksum mismatch")
    if _load_yaml(model / "MLmodel").get("run_id") != expected_run_id:
        raise ValueError("E5 positive MLmodel run_id mismatch")
    return _snapshot(positive)


def build_case(case: str, positive: Path, destination: Path, report: dict) -> dict:
    """Build one isolated case and verify its exact mutation scope.

    Only the newly created destination and in-memory report copy can change.
    The caller is responsible for choosing an ignored workspace.
    """
    if case not in ALL_CASES:
        raise ValueError(f"Unknown E5 case: {case}")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite scenario: {destination}")
    source_snapshot = verify_positive(positive, report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(positive, destination, symlinks=False)

    trial_report = copy.deepcopy(report)
    trial_report["runs"][0]["model_artifact"] = str((destination / "model").resolve())
    model = destination / "model"
    mlmodel = model / "MLmodel"
    pickle = model / "model.pkl"
    dataset = destination / "dataset.csv"
    expected_run_id = report["runs"][0]["mlflow_run_id"]
    expected_model_sha = report["runs"][0]["model_pickle_sha256"]
    expected_dataset_sha = report["dataset"]["sha256"]

    if case == "F1":
        mlmodel.unlink()
    elif case in ("F2", "C5"):
        _flip_one_model_byte(pickle)
        if case == "C5":
            trial_report["runs"][0]["model_pickle_sha256"] = sha256_file(pickle)
    elif case in ("F3", "C6"):
        _mutate_first_feature_digit(dataset)
        if case == "C6":
            trial_report["dataset"]["sha256"] = sha256_file(dataset)
    elif case in ("F4", "C3"):
        original = mlmodel.read_text(encoding="utf-8")
        if case == "F4":
            replacement = _replace_top_level_run_id(
                original, expected_run_id, _fake_run_id(expected_run_id))
            expected_metadata = _load_yaml(mlmodel).copy()
            expected_metadata["run_id"] = _fake_run_id(expected_run_id)
        else:
            replacement = _drop_top_level_run_id(original, expected_run_id)
            expected_metadata = _load_yaml(mlmodel).copy()
            expected_metadata.pop("run_id")
        mlmodel.write_text(replacement, encoding="utf-8")
        if _load_yaml(mlmodel) != expected_metadata:
            raise AssertionError("Unexpected metadata mutation")
    elif case == "C1":
        mlmodel.write_text("run_id: [unterminated\n", encoding="utf-8")
        try:
            _load_yaml(mlmodel)
        except yaml.YAMLError:
            pass
        else:
            raise AssertionError("C1 YAML must not parse")
    elif case == "C2":
        metadata = _load_yaml(mlmodel)
        if "e5_synthetic_note" in metadata:
            raise ValueError("C2 note already present")
        with mlmodel.open("a", encoding="utf-8") as stream:
            stream.write("\ne5_synthetic_note: added_for_e5_challenge\n")
        augmented = _load_yaml(mlmodel)
        expected_metadata = dict(metadata, e5_synthetic_note="added_for_e5_challenge")
        if augmented != expected_metadata:
            raise AssertionError("C2 changed unexpected metadata")
    elif case == "C4":
        trial_report["runs"][0]["mlflow_run_id"] = _fake_run_id(expected_run_id)

    # Enforce the mutated file set and the reference-report fields, not just
    # the outcome of validators. A coding error in mutation must fail as a
    # construction failure, not count as detection success.
    final_snapshot = _snapshot(destination)
    changed_files = sorted(set(source_snapshot) ^ set(final_snapshot) | {
        name for name in set(source_snapshot) & set(final_snapshot)
        if source_snapshot[name] != final_snapshot[name]
    })
    expected_changed = {
        "P0": [], "F1": ["model/MLmodel"],
        "F2": ["model/model.pkl"], "F3": ["dataset.csv"],
        "F4": ["model/MLmodel"], "C1": ["model/MLmodel"],
        "C2": ["model/MLmodel"], "C3": ["model/MLmodel"],
        "C4": [], "C5": ["model/model.pkl"], "C6": ["dataset.csv"],
    }[case]
    if changed_files != expected_changed:
        raise AssertionError(f"Unexpected E5 changes: {case} {changed_files}")
    if _snapshot(positive) != source_snapshot:
        raise AssertionError("E5 positive control was modified")
    if case not in ("F2", "C5") and sha256_file(pickle) != expected_model_sha:
        raise AssertionError("Unexpected model hash change")
    if case not in ("F3", "C6") and sha256_file(dataset) != expected_dataset_sha:
        raise AssertionError("Unexpected dataset hash change")
    if case not in ("F4", "C4") and trial_report["runs"][0]["mlflow_run_id"] != expected_run_id:
        raise AssertionError("Unexpected reference run ID change")
    if case != "C5" and trial_report["runs"][0]["model_pickle_sha256"] != expected_model_sha:
        raise AssertionError("Unexpected expected model hash change")
    if case != "C6" and trial_report["dataset"]["sha256"] != expected_dataset_sha:
        raise AssertionError("Unexpected expected dataset hash change")
    if case in ("F2", "C5") and sha256_file(pickle) == expected_model_sha:
        raise AssertionError("Model mutation did not occur")
    if case in ("F3", "C6") and sha256_file(dataset) == expected_dataset_sha:
        raise AssertionError("Dataset mutation did not occur")

    return {
        "report": trial_report,
        "changed_files": changed_files,
        "case_files_sha256": final_snapshot,
        "reference_changes": {
            "run_id": trial_report["runs"][0]["mlflow_run_id"] != expected_run_id,
            "model_sha256": trial_report["runs"][0]["model_pickle_sha256"] != expected_model_sha,
            "dataset_sha256": trial_report["dataset"]["sha256"] != expected_dataset_sha,
        },
        "invariants_passed": True,
    }


def classify(action: Callable[[], None], repo_root: Path) -> dict:
    """Return TEST_ERROR for unexpected exceptions, never a false REJECT."""
    try:
        action()
    except EXPECTED_REJECTION_ERRORS as exc:
        return {
            "decision": "REJECT", "exception_type": type(exc).__name__,
            "message": _redact_paths(str(exc), repo_root),
        }
    except Exception as exc:
        return {
            "decision": "TEST_ERROR", "exception_type": type(exc).__name__,
            "message": _redact_paths(str(exc), repo_root),
        }
    return {"decision": "ACCEPT", "exception_type": None, "message": None}


def _redact_paths(message: str, repo_root: Path) -> str:
    """Keep provenance diagnostics usable without absolute local repo paths."""
    return message.replace(str(repo_root.resolve()), "<REPO_ROOT>")


def run_validator(mode: str, report: dict, repo_root: Path, model: Path, dataset: Path) -> None:
    if mode == "V1":
        validate_model_directory(model)
        return
    if mode not in ("V2", "V3"):
        raise ValueError(f"Unknown E5 validator: {mode}")
    checked_model, reference = verify_e1_reference(report, repo_root, dataset)
    if mode == "V3":
        metadata = _load_yaml(checked_model / "MLmodel")
        if metadata.get("run_id") != reference["mlflow_run_id"]:
            raise ValueError("MLmodel run_id mismatch")


def evaluate_case(case: str, destination: Path, trial_report: dict, repo_root: Path) -> dict:
    """Evaluate the three frozen validators; retain expected/observed separately."""
    outcomes = {}
    for index, mode in enumerate(VALIDATORS):
        outcome = classify(
            lambda mode=mode: run_validator(mode, trial_report, repo_root,
                                            destination / "model", destination / "dataset.csv"),
            repo_root,
        )
        outcome["expected"] = EXPECTED_DECISIONS[case][index]
        outcome["matches_expectation"] = outcome["decision"] == outcome["expected"]
        outcomes[mode] = outcome
    return outcomes


def _versions() -> dict[str, str]:
    packages = ("PyYAML", "mlflow", "scikit-learn")
    versions = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def _write_json_new(path: Path, report: dict) -> None:
    """Never overwrite a previous verified measurement."""
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite E5 evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(report, indent=2, sort_keys=False) + "\n").encode("utf-8")
    with path.open("xb") as handle:
        handle.write(encoded)


def execute_e5(
    e1_report: Path, dataset: Path, positive: Path,
    workspace: Path, output: Path, repo_root: Path,
) -> dict:
    """Run each scenario once in new folders after enforcing E1/P0 integrity."""
    root = repo_root.resolve()
    e1_report, dataset = e1_report.resolve(), dataset.resolve()
    positive, workspace, output = positive.resolve(), workspace.resolve(), output.resolve()
    if workspace.exists() or output.exists():
        raise FileExistsError("Refusing to overwrite E5 workspace or report")
    if not positive.is_relative_to(root) or not workspace.is_relative_to(root):
        raise ValueError("E5 working directories must be inside the repository")
    if positive == workspace or positive.is_relative_to(workspace) or workspace.is_relative_to(positive):
        raise ValueError("E5 workspace overlaps positive control")
    if not e1_report.is_relative_to(root) or not dataset.is_relative_to(root):
        raise ValueError("E1 inputs must remain repository-local")
    if not output.is_relative_to(root) or output.is_relative_to(workspace):
        raise ValueError("E5 output must be separate and repository-local")
    if not positive.is_dir():
        raise FileNotFoundError("E5 positive control missing")
    # The full E1-based evaluation must not accidentally stage its workspaces.
    subprocess.run(["git", "check-ignore", "-q", str(workspace.relative_to(root))],
                   cwd=root, check=True)
    source_report_sha = sha256_file(e1_report)
    source_dataset_sha = sha256_file(dataset)
    source_report = json.loads(e1_report.read_text(encoding="utf-8"))
    source_model, source_reference = verify_e1_reference(source_report, root, dataset)
    source_model_sha = sha256_file(source_model / "model.pkl")
    source_mlmodel_sha = sha256_file(source_model / "MLmodel")
    source_model_snapshot = _snapshot(source_model)
    positive_before = verify_positive(positive, source_report)
    if _snapshot(positive / "model") != source_model_snapshot:
        raise ValueError("E5 positive control model directory differs from E1")

    workspace.mkdir(parents=True, exist_ok=False)
    rows = []
    for case in ALL_CASES:
        destination = workspace / case
        built = build_case(case, positive, destination, source_report)
        decisions = evaluate_case(case, destination, built["report"], root)
        rows.append({
            "case": case,
            "suite": "exploratory_pilot_replication" if case in PILOT_CASES else "prospective_challenge",
            "description": CASE_DESCRIPTION[case],
            "scenario_directory": destination.relative_to(root).as_posix(),
            "changed_files": built["changed_files"],
            "case_files_sha256": built["case_files_sha256"],
            "reference_changes": built["reference_changes"],
            "invariants_passed": built["invariants_passed"],
            "validators": decisions,
        })

    # No source E1 or E5 positive-control bytes may change during evaluation.
    if (sha256_file(e1_report) != source_report_sha
            or sha256_file(dataset) != source_dataset_sha
            or sha256_file(source_model / "model.pkl") != source_model_sha
            or sha256_file(source_model / "MLmodel") != source_mlmodel_sha
            or _snapshot(source_model) != source_model_snapshot
            or _snapshot(positive) != positive_before):
        raise AssertionError("E5 detected unintended mutation of its inputs")

    counts = {}
    for suite, keys in (("pilot_faults", PILOT_CASES[1:]),
                        ("challenge", CHALLENGE_CASES)):
        counts[suite] = {
            mode: {
                "rejected": sum(
                    row["validators"][mode]["decision"] == "REJECT"
                    for row in rows if row["case"] in keys
                ),
                "test_errors": sum(
                    row["validators"][mode]["decision"] == "TEST_ERROR"
                    for row in rows if row["case"] in keys
                ),
                "total": len(keys),
            }
            for mode in VALIDATORS
        }
    mismatches = [
        {"case": row["case"], "validator": mode}
        for row in rows for mode in VALIDATORS
        if not row["validators"][mode]["matches_expectation"]
    ]
    report = {
        "protocol": "E5-controlled-provenance-fault-injection-v0.1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "deterministic controlled local artifact/data fault detection; no model loading",
        "pilot_is_exploratory": True,
        "challenge_expected_decisions_frozen_before_real_run": True,
        "reference": {
            "canonical_run_id": source_reference["mlflow_run_id"],
            "model_pickle_sha256": source_reference["model_pickle_sha256"],
            "dataset_sha256": source_report["dataset"]["sha256"],
            "original_report_sha256": source_report_sha,
            "original_mlmodel_sha256": source_mlmodel_sha,
        },
        "environment": _versions(),
        "control": {
            "intact_source_and_positive_control_after_run": True,
            "positive_control_files_sha256": positive_before,
        },
        "cases": rows,
        "summary": {"by_suite": counts, "unexpected_outcomes": mismatches},
        "completed": True,
    }
    _write_json_new(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e1-report", type=Path, default=DEFAULT_E1_REPORT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--positive-control", type=Path, default=DEFAULT_POSITIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workspace", type=Path, default=None,
                        help="New ignored E5 run directory; auto-created with unique name if omitted")
    args = parser.parse_args()
    if args.workspace is None:
        E5_WORKSPACE.mkdir(parents=True, exist_ok=True)
        name = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
        workspace = E5_WORKSPACE / name
    else:
        workspace = args.workspace
    report = execute_e5(args.e1_report, args.dataset, args.positive_control,
                        workspace, args.output, ROOT)
    print("E5 report:", args.output)
    print("Cases:", len(report["cases"]))
    print("Pilot / challenge:", len(PILOT_CASES), "/", len(CHALLENGE_CASES))
    print("Expected outcome deviations:", len(report["summary"]["unexpected_outcomes"]))
    print(json.dumps(report["summary"]["by_suite"], indent=2))
    if report["summary"]["unexpected_outcomes"]:
        raise SystemExit("E5: unexpected outcomes need review; evidence preserved")


if __name__ == "__main__":
    main()
