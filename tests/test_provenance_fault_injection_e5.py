"""E5 deterministic synthetic fault tests: no E1 artifacts, Docker or model loading."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest
import yaml

from experiments import provenance_fault_injection_e5 as e5


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def synthetic_e1(tmp_path):
    """A five-run E1-format report and tiny disposable MLflow-like artifact."""
    root = tmp_path / "repo"
    model = root / "fixture/model"
    model.mkdir(parents=True)
    (model / "model.pkl").write_bytes(b"synthetic_e5_pickle_bytes_do_not_load")
    canonical_run = "a" * 32
    (model / "MLmodel").write_text(
        f"artifact_path: model\nrun_id: {canonical_run}\nflavors:\n  python_function: {{}}\n",
        encoding="utf-8",
    )
    dataset = root / "data/processed/fixture.csv"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("f1,f2,target\n12.31,4,0\n7.5,9,1\n", encoding="utf-8")
    sha_model = file_hash(model / "model.pkl")
    report = {
        "summary": {"all_successful": True, "successful_runs": 5,
                    "model_pickle_hashes_identical": True},
        "runs": [
            {"status": "FINISHED", "mlflow_run_id": canonical_run if i == 0 else f"{i:032x}",
             "model_artifact": str(model), "model_pickle_sha256": sha_model}
            for i in range(5)
        ],
        "dataset": {"sha256": file_hash(dataset)},
    }
    report_file = root / "results/reproducibility_e1_verified.json"
    report_file.parent.mkdir(parents=True)
    report_file.write_text(json.dumps(report), encoding="utf-8")
    positive = root / ".research_artifacts/e5/positive_control"
    positive.mkdir(parents=True)
    (positive / "model").mkdir()
    (positive / "model/model.pkl").write_bytes((model / "model.pkl").read_bytes())
    (positive / "model/MLmodel").write_bytes((model / "MLmodel").read_bytes())
    (positive / "dataset.csv").write_bytes(dataset.read_bytes())
    return {"root": root, "model": model, "dataset": dataset,
            "positive": positive, "report": report, "report_file": report_file}


def test_frozen_expected_decisions_cover_every_declared_case():
    assert e5.ALL_CASES == e5.PILOT_CASES + e5.CHALLENGE_CASES
    assert set(e5.EXPECTED_DECISIONS) == set(e5.ALL_CASES)
    assert set(e5.CASE_DESCRIPTION) == set(e5.ALL_CASES)
    assert e5.VALIDATORS == ("V1", "V2", "V3")
    assert len(e5.CHALLENGE_CASES) == 6


@pytest.mark.parametrize("case", e5.ALL_CASES)
def test_synthetic_scenarios_match_frozen_validator_outcomes(synthetic_e1, case):
    f = synthetic_e1
    before = e5._snapshot(f["positive"])
    source_report = copy.deepcopy(f["report"])
    destination = f["root"] / ".research_artifacts/e5/faults" / case
    built = e5.build_case(case, f["positive"], destination, f["report"])
    results = e5.evaluate_case(case, destination, built["report"], f["root"])
    assert [results[mode]["decision"] for mode in e5.VALIDATORS] == list(e5.EXPECTED_DECISIONS[case])
    assert all(results[mode]["matches_expectation"] for mode in e5.VALIDATORS)
    assert built["invariants_passed"] is True
    assert e5._snapshot(f["positive"]) == before
    assert f["report"] == source_report
    assert file_hash(f["model"] / "model.pkl") == f["report"]["runs"][0]["model_pickle_sha256"]
    assert file_hash(f["dataset"]) == f["report"]["dataset"]["sha256"]


@pytest.mark.parametrize("case", ["C1", "C2", "C3", "C4", "C5", "C6"])
def test_challenge_mutations_match_declared_scope(synthetic_e1, case):
    f = synthetic_e1
    destination = f["root"] / ".research_artifacts/e5/scopes" / case
    built = e5.build_case(case, f["positive"], destination, f["report"])
    if case in ("C1", "C2", "C3"):
        assert built["changed_files"] == ["model/MLmodel"]
    elif case in ("C5",):
        assert built["changed_files"] == ["model/model.pkl"]
        assert built["reference_changes"] == {
            "run_id": False, "model_sha256": True, "dataset_sha256": False,
        }
    elif case in ("C6",):
        assert built["changed_files"] == ["dataset.csv"]
        assert built["reference_changes"] == {
            "run_id": False, "model_sha256": False, "dataset_sha256": True,
        }
    else:
        assert built["changed_files"] == []
        assert built["reference_changes"] == {
            "run_id": True, "model_sha256": False, "dataset_sha256": False,
        }
    if case == "C1":
        with pytest.raises(yaml.YAMLError):
            e5._load_yaml(destination / "model/MLmodel")
    if case == "C2":
        data = e5._load_yaml(destination / "model/MLmodel")
        assert data["run_id"] == f["report"]["runs"][0]["mlflow_run_id"]
        assert data["e5_synthetic_note"] == "added_for_e5_challenge"
    if case == "C3":
        assert "run_id" not in e5._load_yaml(destination / "model/MLmodel")


def test_no_overwrite_and_unknown_case_rejected(synthetic_e1):
    f = synthetic_e1
    destination = f["root"] / ".research_artifacts/e5/scenarios/P0"
    e5.build_case("P0", f["positive"], destination, f["report"])
    with pytest.raises(FileExistsError, match="overwrite"):
        e5.build_case("P0", f["positive"], destination, f["report"])
    with pytest.raises(ValueError, match="Unknown E5 case"):
        e5.build_case("Z9", f["positive"], destination, f["report"])


def test_positive_checksum_tamper_is_rejected_before_construction(synthetic_e1):
    f = synthetic_e1
    (f["positive"] / "model/model.pkl").write_bytes(b"tampered")
    destination = f["root"] / ".research_artifacts/e5/faults/P0"
    with pytest.raises(ValueError, match="positive model checksum"):
        e5.build_case("P0", f["positive"], destination, f["report"])
    assert not destination.exists()


def test_unexpected_exception_is_test_error_not_successful_rejection(synthetic_e1):
    f = synthetic_e1
    def fail_unexpected():
        raise AssertionError("unexpected test harness error")
    assert e5.classify(fail_unexpected, f["root"])["decision"] == "TEST_ERROR"
    assert e5.classify(lambda: (_ for _ in ()).throw(ValueError("expected")), f["root"])["decision"] == "REJECT"
    assert e5.classify(lambda: None, f["root"])["decision"] == "ACCEPT"


def test_exception_messages_redact_repository_absolute_path(synthetic_e1):
    f = synthetic_e1
    def fail():
        raise FileNotFoundError(str(f["root"] / "secret-name"))
    result = e5.classify(fail, f["root"])
    assert str(f["root"]) not in result["message"]
    assert "<REPO_ROOT>" in result["message"]


def test_full_synthetic_run_isolation_reporting_and_no_reuse(synthetic_e1):
    f = synthetic_e1
    root = f["root"]
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / ".gitignore").write_text(".research_artifacts/\n", encoding="utf-8")
    workspace = root / ".research_artifacts/e5/runs/synthetic_01"
    output = root / "results/provenance_fault_injection_e5_verified.json"
    report_sha_before = file_hash(f["report_file"])
    source_sha_before = e5._snapshot(f["model"])
    result = e5.execute_e5(f["report_file"], f["dataset"], f["positive"],
                           workspace, output, root)
    assert result["completed"] is True
    assert len(result["cases"]) == 11
    assert result["summary"]["unexpected_outcomes"] == []
    assert result["summary"]["by_suite"]["pilot_faults"]["V1"]["rejected"] == 1
    assert result["summary"]["by_suite"]["pilot_faults"]["V2"]["rejected"] == 3
    assert result["summary"]["by_suite"]["pilot_faults"]["V3"]["rejected"] == 4
    assert result["summary"]["by_suite"]["challenge"]["V1"]["rejected"] == 0
    assert result["summary"]["by_suite"]["challenge"]["V2"]["rejected"] == 0
    assert result["summary"]["by_suite"]["challenge"]["V3"]["rejected"] == 3
    assert file_hash(f["report_file"]) == report_sha_before
    assert e5._snapshot(f["model"]) == source_sha_before
    assert all("/Users/" not in json.dumps(row) for row in result["cases"])
    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8"))["completed"] is True
    with pytest.raises(FileExistsError, match="overwrite"):
        e5.execute_e5(f["report_file"], f["dataset"], f["positive"], workspace, output, root)


def test_full_run_rejects_unignored_workspace(synthetic_e1):
    f = synthetic_e1
    root = f["root"]
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    workspace = root / "unignored"
    with pytest.raises(subprocess.CalledProcessError):
        e5.execute_e5(f["report_file"], f["dataset"], f["positive"],
                      workspace, root / "results/e5.json", root)
    assert not workspace.exists()
