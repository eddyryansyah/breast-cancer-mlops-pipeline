"""Pure unit tests: E3 never starts or restarts Docker during pytest."""

import json
from pathlib import Path
from types import SimpleNamespace
import urllib.error

import pytest

from experiments.serving_reliability_e3 import (
    E2_IMAGE, E3_CONTAINER, INVOCATIONS, encode_frame, inspect_container,
    invalid_payloads, latency_summary, request, validate_local_endpoint,
    wait_for_ready, write_report,
)


def test_local_endpoint_accepts_only_canonical_url():
    assert validate_local_endpoint(INVOCATIONS) == INVOCATIONS


@pytest.mark.parametrize("url", [
    "https://127.0.0.1:5001/invocations",
    "http://localhost:5001/invocations",
    "http://127.0.0.1:5002/invocations",
    "http://127.0.0.1:5001/ping",
    "http://127.0.0.1:5001/invocations?debug=true",
    "http://127.0.0.1:5001/invocations#fragment",
    "http://evil.example/invocations",
    "http://user:pass@127.0.0.1:5001/invocations",
    "file:///tmp/invocations",
])
def test_rejects_nonlocal_url(url):
        with pytest.raises(ValueError, match="only permits"):
            validate_local_endpoint(url)


def test_latency_statistics():
    result = latency_summary([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result["n"] == 5
    assert result["median_ms"] == 3
    assert result["p95_ms"] == pytest.approx(4.8)
    assert result["population_std_ms"] == pytest.approx(2 ** 0.5)


@pytest.mark.parametrize("values", [[], [-1], [float("inf")], [float("nan")]])
def test_rejects_bad_latency_samples(values):
        with pytest.raises(ValueError):
            latency_summary(values)


def test_atomic_write(tmp_path):
    path = tmp_path / "nested" / "report.json"
    write_report({"ok": True}, path)
    assert json.loads(path.read_text()) == {"ok": True}
    assert not path.with_name(path.name + ".tmp").exists()


class _Frame:
    columns = SimpleNamespace(tolist=lambda: ["a", "b"])
    values = SimpleNamespace(tolist=lambda: [[1, 2], [3, 4]])


def test_encode_frame_has_schema():
    actual = json.loads(encode_frame(_Frame()))
    assert actual == {"dataframe_split": {"columns": ["a", "b"], "data": [[1, 2], [3, 4]]}}


def test_invalid_payload_names_unique():
    probes = invalid_payloads(_Frame())
    assert len(probes) == len({name for name, _ in probes}) == 5
    assert any(data == b"[]" for _, data in probes)


def _docker_details(image_id="sha256:expected", running=True, port="5001"):
    return [{"Name": f"/{E3_CONTAINER}", "State": {"Running": running},
             "Config": {"Image": E2_IMAGE}, "Image": image_id,
             "NetworkSettings": {"Ports": {"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": port}]}}}]


def test_container_identity_matching(monkeypatch):
    def fake_run(*args, **kwargs):
        assert args[0] == ["docker", "inspect", E3_CONTAINER]
        return SimpleNamespace(stdout=json.dumps(_docker_details()))
    monkeypatch.setattr("experiments.serving_reliability_e3.subprocess.run", fake_run)
    info = inspect_container(E3_CONTAINER, E2_IMAGE, "sha256:expected")
    assert info["image_id"] == "sha256:expected"
    assert info["host_port"] == 5001


@pytest.mark.parametrize("override", [
    {"image_id": "sha256:other"}, {"running": False}, {"port": "5002"},
])
def test_container_rejects_mismatched_identity(monkeypatch, override):
        monkeypatch.setattr("experiments.serving_reliability_e3.subprocess.run", lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(_docker_details(**override))))
        with pytest.raises(ValueError, match="does not match"):
            inspect_container(E3_CONTAINER, E2_IMAGE, "sha256:expected")


def test_container_rejects_unapproved_name_before_call():
    with pytest.raises(ValueError, match="dedicated"):
        inspect_container("production", E2_IMAGE, "sha256:expected")


def test_readiness_returns_when_healthy(monkeypatch):
    calls = iter([False, True])
    monkeypatch.setattr("experiments.serving_reliability_e3.healthy", lambda url: next(calls))
    monkeypatch.setattr("experiments.serving_reliability_e3.time.sleep", lambda interval: None)
    assert wait_for_ready(INVOCATIONS, timeout=5, interval=0.01) >= 0


def test_readiness_requires_positive_timeout():
    with pytest.raises(ValueError):
        wait_for_ready(INVOCATIONS, timeout=0)


def test_request_rejects_unsupported_method():
    with pytest.raises(ValueError):
        request("DELETE", INVOCATIONS)


def test_request_captures_http_error(monkeypatch):
    from io import BytesIO
    fake_error = urllib.error.HTTPError(INVOCATIONS, 400, "Bad request", {}, BytesIO(b'{"error":"invalid"}'))
    def throw(*args, **kwargs):
        raise fake_error
    monkeypatch.setattr("experiments.serving_reliability_e3.urllib.request.urlopen", throw)
    status, body = request("POST", INVOCATIONS, b"[]")
    assert status == 400 and "invalid" in body


def test_no_restart_without_explicit_opt_in(tmp_path):
    from experiments.serving_reliability_e3 import run_reliability
    with pytest.raises(ValueError, match="--enable-restart"):
        run_reliability(
            e1_report=tmp_path / "e1.json",
            e2_report=tmp_path / "e2.json",
            dataset=tmp_path / "data.csv", output=tmp_path / "report.json",
            url=INVOCATIONS, container=E3_CONTAINER, image=E2_IMAGE,
            valid_requests=5, restarts=1, enable_restart=False,
            readiness_timeout=60,
        )


def test_no_nonlocal_run_before_loading_data(tmp_path):
    from experiments.serving_reliability_e3 import run_reliability
    with pytest.raises(ValueError, match="only permits"):
        run_reliability(
            e1_report=tmp_path / "e1.json",
            e2_report=tmp_path / "e2.json",
            dataset=tmp_path / "data.csv", output=tmp_path / "report.json",
            url="https://example.com/invocations",
            container=E3_CONTAINER, image=E2_IMAGE,
            valid_requests=5, restarts=0, enable_restart=False,
            readiness_timeout=60,
        )


def test_no_unapproved_container_before_loading_data(tmp_path):
    from experiments.serving_reliability_e3 import run_reliability
    with pytest.raises(ValueError, match="Dedicated E3"):
        run_reliability(
            e1_report=tmp_path / "e1.json",
            e2_report=tmp_path / "e2.json",
            dataset=tmp_path / "data.csv", output=tmp_path / "report.json",
            url=INVOCATIONS, container="production", image=E2_IMAGE,
            valid_requests=5, restarts=0, enable_restart=False,
            readiness_timeout=60,
        )
