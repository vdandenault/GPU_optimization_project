"""Focused checks for hardware-pilot accounting, persistence and watchdogs."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("hardware_pilot", ROOT / "scripts/run_hardware_pilot.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_energy_clips_to_exact_arrival_and_drain_bounds():
    samples = [{"monotonic_s": t, "power.draw": p} for t, p in [(0, 10), (2, 30), (4, 50)]]
    assert pilot.integrate_energy(samples, 1, 3, 3) == pytest.approx(60)


@pytest.mark.parametrize("start,end,gap", [(-1, 3, 3), (1, 5, 3), (1, 3, 1)])
def test_energy_rejects_missing_brackets_and_stale_gaps(start, end, gap):
    samples = [{"monotonic_s": t, "power.draw": 10} for t in [0, 2, 4]]
    with pytest.raises(ValueError):
        pilot.integrate_energy(samples, start, end, gap)


def test_checkpoint_retains_previous_and_rejects_corrupt_current(tmp_path):
    p = tmp_path / "progress.json"
    pilot.write_json(p, {"status": "running"})
    first = p.read_bytes()
    pilot.write_json(p, {"status": "complete"})
    assert p.with_name("progress.json.previous").read_bytes() == first
    broken = json.loads(p.read_text())
    broken["payload"]["status"] = "tampered"
    p.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="corrupt checkpoint"):
        pilot.write_json(p, {"status": "replacement"})
    assert p.with_name("progress.json.previous").read_bytes() == first


@pytest.mark.parametrize("change", [
    {"trace_rows": [-1]}, {"trace_rows": [0, 0]}, {"max_model_len": 1024},
    {"model_revision": "main"}, {"trial_timeout_s": 3600}, {"temperature_stop_c": 90},
    {"trace_sha256": "0" * 64},
    {"sampling_backend": "unknown"},
])
def test_invalid_configuration_cannot_silently_change_protocol(tmp_path, change):
    config = json.loads(pilot.DEFAULT_CONFIG.read_text())
    config.update(change)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        pilot.load_config(path)


@pytest.mark.parametrize("backend,value", [("auto", None), ("pytorch", "0")])
def test_sampler_backend_is_an_explicit_recordable_environment_override(backend, value):
    overrides = pilot.serving_environment({"sampling_backend": backend}, "GPU-fixture")
    assert overrides.get("VLLM_USE_FLASHINFER_SAMPLER") == value
    assert overrides["CUDA_VISIBLE_DEVICES"] == "GPU-fixture"


@contextmanager
def completion_server(*, wrong_count=False, incomplete=False):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert payload["ignore_eos"] and payload["return_token_ids"]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunks = [
                {"choices": [{"token_ids": [100], "finish_reason": None}]},
                {"choices": [{"token_ids": [101], "finish_reason": "length"}]},
                {"choices": [], "usage": {"prompt_tokens": len(payload["prompt"]),
                                          "completion_tokens": 3 if wrong_count else 2}},
            ]
            for chunk in chunks:
                self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                self.wfile.flush()
            if not incomplete:
                self.wfile.write(b"data: [DONE]\n\n")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("wrong_count,incomplete,expected", [(False, False, "complete"),
                                                            (True, False, "failed"),
                                                            (False, True, "failed")])
def test_stream_requires_done_exact_usage_and_exact_token_ids(tmp_path, wrong_count, incomplete, expected):
    requests = pilot.Journal(tmp_path / "requests.jsonl")
    stream = pilot.Journal(tmp_path / "stream.jsonl")
    row = {"trace_row": 2, "arrival_s": 0, "prompt": [100, 200], "prompt_tokens": 2, "output_tokens": 2}
    try:
        with completion_server(wrong_count=wrong_count, incomplete=incomplete) as base:
            result = pilot.request_one(row, base, "test", time.monotonic(),
                                       {"seed": 42, "request_timeout_s": 2}, requests, stream, threading.Event())
        assert result["status"] == expected
        records = [json.loads(s) for s in (tmp_path / "requests.jsonl").read_text().splitlines()]
        assert [r["event"] for r in records] == ["scheduled", "dispatched", "outcome"]
        if expected == "complete":
            assert result["streamed_tokens"] == 2
            assert result["e2e_s"] >= result["ttft_s"] >= 0
    finally:
        requests.close()
        stream.close()


@pytest.mark.parametrize("failure", ["temperature", "stale"])
def test_watchdog_aborts_on_measured_temperature_or_failed_telemetry(tmp_path, monkeypatch, failure):
    aborted = threading.Event()
    reasons = []

    def abort(reason):
        reasons.append(reason)
        aborted.set()

    def query():
        if failure == "stale":
            raise RuntimeError("NVML unavailable")
        return {"query_start_s": time.monotonic(), "monotonic_s": time.monotonic(), "temperature.gpu": 78}

    monkeypatch.setattr(pilot.GPUWatch, "query", staticmethod(query))
    watch = pilot.GPUWatch(tmp_path, {"telemetry_period_s": 0.01, "telemetry_stale_s": 0.1,
                                     "temperature_stop_c": 77}, abort)
    watch.start()
    try:
        assert aborted.wait(2)
        assert reasons == ["measured_gpu_temperature_limit" if failure == "temperature" else "stale_gpu_telemetry"]
    finally:
        watch.close()


def test_full_storage_immediately_aborts_instead_of_disabling_watchdog(tmp_path, monkeypatch):
    aborted = threading.Event()
    reasons = []

    def abort(reason):
        reasons.append(reason)
        aborted.set()

    def full_disk(event):
        raise OSError(122, "Disk quota exceeded")

    watch = pilot.GPUWatch(tmp_path, {"telemetry_period_s": 0.01, "telemetry_stale_s": 5,
                                     "temperature_stop_c": 77}, abort)
    monkeypatch.setattr(watch, "query", lambda: {"temperature.gpu": 28})
    monkeypatch.setattr(watch.journal, "add", full_disk)
    watch.start()
    try:
        assert aborted.wait(1)
        assert reasons[0].startswith("telemetry_recording_failure:")
    finally:
        watch.close()


@pytest.mark.parametrize("abort_logging_full", [False, True])
def test_full_runner_retains_auditable_records_and_cleans_owned_server(tmp_path, monkeypatch, abort_logging_full):
    """Exercise real subprocess/HTTP/journal lifecycles without a GPU or model."""
    for name in ["scripts", "runtime/serving", "data/inference_serving", "configs/hardware"]:
        (tmp_path / name).mkdir(parents=True)
    for name in ["uv.lock", "pyproject.toml", ".python-version", "runtime/serving/uv.lock",
                 "runtime/serving/pyproject.toml", "runtime/serving/.python-version",
                 "scripts/setup_pilot_runtime.py"]:
        (tmp_path / name).write_text("fixture\n")
    shutil.copyfile(ROOT / "data/inference_serving/azure_code_evaluation.csv",
                    tmp_path / "data/inference_serving/azure_code_evaluation.csv")
    config = json.loads(pilot.DEFAULT_CONFIG.read_text())
    config["telemetry_period_s"] = 0.02
    config_path = tmp_path / "configs/hardware/blackwell-qualification.json"
    config_path.write_text(json.dumps(config))
    runner_copy = tmp_path / "scripts/run_hardware_pilot.py"
    runner_copy.write_text("# retained source fixture\n")
    interpreter = tmp_path / "fake-serving-python"
    interpreter.write_text("#!" + sys.executable + "\n" + '''
import csv, json, os, sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
args = sys.argv[1:]
def arg(name): return args[args.index(name) + 1]
if "--probe-output" in args:
    c = json.loads(Path(arg("--config")).read_text())
    with Path(c["trace"]).open() as f: rows = list(csv.DictReader(f))
    start = datetime.fromisoformat(rows[0]["TIMESTAMP"])
    work = []
    for i in c["trace_rows"]:
        r = rows[i]
        work.append({"trace_row": i, "arrival_s": (datetime.fromisoformat(r["TIMESTAMP"]) - start).total_seconds(),
                     "prompt_tokens": int(r["ContextTokens"]), "output_tokens": int(r["GeneratedTokens"]),
                     "prompt": [100] * int(r["ContextTokens"])})
    Path(arg("--probe-output")).write_text(json.dumps({"payload": {"workload": work, "packages": []}}))
else:
    Path("server-pid").write_text(str(os.getpid()))
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200); self.end_headers()
            self.wfile.write(json.dumps({"data": [{"id": arg("--served-model-name")}]}).encode())
        def do_POST(self):
            p = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200); self.end_headers()
            chunks = [{"choices": [{"token_ids": [101] * p["max_tokens"], "finish_reason": "length"}]},
                      {"choices": [], "usage": {"prompt_tokens": len(p["prompt"]), "completion_tokens": p["max_tokens"]}}]
            for chunk in chunks:
                self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\\n\\n")
            self.wfile.write(b"data: [DONE]\\n\\n")
        def log_message(self, *a): pass
    ThreadingHTTPServer(("127.0.0.1", int(arg("--port"))), Handler).serve_forever()
''')
    interpreter.chmod(0o700)
    venv_interpreter = tmp_path / "serving-venv/bin/python"
    venv_interpreter.parent.mkdir(parents=True)
    venv_interpreter.symlink_to(interpreter)
    monkeypatch.setattr(pilot, "ROOT", tmp_path)
    monkeypatch.setattr(pilot, "__file__", str(runner_copy))

    def fake_command(args, timeout=15):
        if args[0] == "findmnt":
            return json.dumps({"filesystems": [{"source": "test[/networkvolumes/test]", "options": "rw", "target": str(tmp_path)}]})
        if args[0] == "git":
            return "fixture-commit" if "rev-parse" in args else ""
        if args[0] == "nvidia-smi":
            return "" if "--query-compute-apps=pid" in args else "fixture GPU settings"
        return "fixture system inventory"

    queries = 0

    def telemetry():
        nonlocal queries
        queries += 1
        return {"uuid": "GPU-fixture", "name": config["expected_gpu_name"], "power.limit": 145.0,
                "power.draw": 20.0, "temperature.gpu": 78.0 if abort_logging_full and queries > 1 else 28.0,
                "query_start_s": time.monotonic(), "monotonic_s": time.monotonic()}

    monkeypatch.setattr(pilot, "command", fake_command)
    monkeypatch.setattr(pilot.GPUWatch, "query", staticmethod(telemetry))
    original_add = pilot.Journal.add

    def add_with_abort_log_failure(self, event):
        if abort_logging_full and event.get("event") == "abort":
            raise OSError(122, "Disk quota exceeded")
        return original_add(self, event)

    monkeypatch.setattr(pilot.Journal, "add", add_with_abort_log_failure)
    code = pilot.run(SimpleNamespace(config=config_path, serving_python=str(venv_interpreter), provider_record={}))
    assert code == (1 if abort_logging_full else 0)
    directory, = (tmp_path / "runs").glob("qualification-*")
    summary = json.loads((directory / "summary.json").read_text())["payload"]
    assert summary["requests_expected"] == 4
    if abort_logging_full:
        assert "measured_gpu_temperature_limit" in summary["invalid_reasons"]
        assert any(r.startswith("event_recording_failure:") for r in summary["invalid_reasons"])
        assert summary["requests_complete"] == 0
    else:
        assert summary["requests_complete"] == 4
    assert summary["benchmark_measurements_accepted"] is False
    manifest = json.loads((directory / "manifest.json").read_text())["payload"]
    assert manifest["serving_python"] == str(venv_interpreter)
    progress = json.loads((directory / "progress.json").read_text())["payload"]
    if not abort_logging_full:
        assert progress["next_trial_id"] is None
        assert progress["completed_trials"] == [{"trial_id": "0001", "status": "complete"}]
    if (tmp_path / "server-pid").exists():
        pid = int((tmp_path / "server-pid").read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    audit_spec = importlib.util.spec_from_file_location("pilot_audit", ROOT / "scripts/audit_hardware_pilot.py")
    auditor = importlib.util.module_from_spec(audit_spec)
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    audit_spec.loader.exec_module(auditor)
    result = auditor.audit(directory)
    assert result["audit_passed"], result
    assert result["request_outcomes"] == (0 if abort_logging_full else 4)
