#!/usr/bin/env python3
"""Bounded default-settings qualification, deliberately not a benchmark.

The client uses only the standard library. A separate locked serving Python
owns CUDA, tokenization and vLLM. Every invocation is a new attempt; interrupted
trials are retained and never resumed or spliced into another attempt.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/hardware/blackwell-qualification-torch-sampler.json"


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write_json(path, value):
    """Atomic durable JSON with the previous valid checkpoint retained."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = {"schema_version": 1, "payload": value,
            "integrity_sha256": hashlib.sha256(canonical(value)).hexdigest()}
    with tmp.open("wb") as f:
        f.write(canonical(data) + b"\n")
        f.flush()
        os.fsync(f.fileno())
    if path.exists():
        old = json.loads(path.read_text())
        if hashlib.sha256(canonical(old["payload"])).hexdigest() != old["integrity_sha256"]:
            raise ValueError(f"Refusing to replace corrupt checkpoint: {path}")
        previous_tmp = path.with_name(path.name + ".previous.tmp")
        shutil.copyfile(path, previous_tmp)
        with previous_tmp.open("rb") as f:
            os.fsync(f.fileno())
        previous_tmp.replace(path.with_name(path.name + ".previous"))
    tmp.replace(path)
    if hasattr(os, "O_DIRECTORY"):
        fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


class Journal:
    def __init__(self, path):
        self.file = Path(path).open("x", buffering=1)
        self.lock = threading.Lock()

    def add(self, event):
        with self.lock:
            self.file.write(json.dumps(event, allow_nan=False) + "\n")
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.file.close()


def command(args, timeout=15):
    p = subprocess.run(args, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=timeout)
    if p.returncode:
        raise RuntimeError(f"{args[0]} exited {p.returncode}: {p.stderr.strip()}")
    return p.stdout.strip()


def load_config(path):
    c = json.loads(Path(path).read_text())
    if c.get("sampling_backend", "auto") not in ["auto", "pytorch"]:
        raise ValueError("sampling_backend must be auto or pytorch")
    if c["purpose"] != "development_default_settings_baseline":
        raise ValueError("This runner only supports the development default-settings baseline")
    if c["gpu_settings_policy"] != "observe_only_no_clock_or_power_writes":
        raise ValueError("GPU settings changes are not implemented or qualified")
    if not (1 <= len(c["trace_rows"]) <= 32) or len(set(c["trace_rows"])) != len(c["trace_rows"]):
        raise ValueError("Select one to thirty-two distinct development trace rows")
    for key in ["model_revision", "tokenizer_revision"]:
        if len(c[key]) != 40 or any(x not in "0123456789abcdef" for x in c[key]):
            raise ValueError(f"{key} must be an immutable commit")
    for key in ["arrival_scale", "telemetry_period_s", "telemetry_stale_s",
                "startup_timeout_s", "trial_timeout_s", "request_timeout_s",
                "max_dispatch_lateness_s", "temperature_stop_c"]:
        if not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if c["trial_timeout_s"] > 300 or c["startup_timeout_s"] > 1200:
        raise ValueError("Qualification bounds exceeded")
    if c["temperature_stop_c"] > 77 or c["telemetry_stale_s"] > 5:
        raise ValueError("Qualification watchdog bounds exceeded")
    if c["enable_prefix_caching"] or not c["enforce_eager"]:
        raise ValueError("This pilot freezes eager execution and disables prefix caching")
    if c["max_num_seqs"] != 4:
        raise ValueError("Development baseline freezes serving concurrency at four")
    trace = (ROOT / c["trace"]).resolve()
    if not trace.is_relative_to(ROOT / "data"):
        raise ValueError("Trace must be inside repository data/")
    if digest(trace) != c["trace_sha256"]:
        raise ValueError("Workload trace checksum differs from frozen configuration")
    with trace.open(newline="") as f:
        rows = list(csv.DictReader(f))
    selected = []
    if any(type(i) is not int or i < 0 or i >= len(rows) for i in c["trace_rows"]):
        raise ValueError("Trace row index out of range")
    start = datetime.fromisoformat(rows[c["trace_rows"][0]]["TIMESTAMP"])
    for idx in c["trace_rows"]:
        if type(idx) is not int or idx < 0 or idx >= len(rows):
            raise ValueError("Trace row index out of range")
        row = rows[idx]
        prompt, output = int(row["ContextTokens"]), int(row["GeneratedTokens"])
        arrival = (datetime.fromisoformat(row["TIMESTAMP"]) - start).total_seconds() * c["arrival_scale"]
        if prompt <= 0 or output <= 0 or prompt + output > c["max_model_len"]:
            raise ValueError(f"Row {idx} exceeds context; lengths must not be silently truncated")
        if arrival < 0 or arrival >= c["trial_timeout_s"]:
            raise ValueError("Arrival outside trial bounds")
        selected.append({"trace_row": idx, "source_timestamp": row["TIMESTAMP"],
                         "arrival_s": arrival, "prompt_tokens": prompt, "output_tokens": output})
    if selected != sorted(selected, key=lambda x: x["arrival_s"]):
        raise ValueError("Trace rows must be chronological")
    return c, selected


def serving_environment(config, gpu_uuid):
    """Non-secret environment overrides, retained verbatim in the manifest."""
    overrides = dict(HF_HOME=str(ROOT / "runs/cache/huggingface"),
                     XDG_CACHE_HOME=str(ROOT / "runs/cache/serving"),
                     TRITON_CACHE_DIR=str(ROOT / "runs/cache/triton"),
                     TORCHINDUCTOR_CACHE_DIR=str(ROOT / "runs/cache/torchinductor"),
                     VLLM_NO_USAGE_STATS="1", HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1",
                     CUDA_VISIBLE_DEVICES=gpu_uuid, PYTHONUNBUFFERED="1",
                     TMPDIR=str(ROOT / "runs/cache/tmp"))
    if config.get("sampling_backend", "auto") == "pytorch":
        overrides["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    return overrides


def probe(config_path, output):
    """Only invoked by the isolated serving Python, before server startup."""
    import importlib.metadata
    import torch
    from transformers import AutoTokenizer

    c, rows = load_config(config_path)
    if importlib.metadata.version("vllm") != c["vllm_version"]:
        raise ValueError("Installed vLLM differs from the frozen configuration")
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.manual_seed(c["seed"])
    x = torch.ones((512, 512), device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    assert torch.isfinite(y).all().item() and torch.allclose(y, torch.full_like(y, 512))
    logits = torch.arange(1024, device="cuda", dtype=torch.float32).repeat(4, 1)
    top_values, top_ids = torch.topk(logits, 10, dim=-1)
    sampled = torch.multinomial(torch.softmax(top_values, dim=-1), 1)
    assert sampled.shape == (4, 1) and torch.isfinite(top_values).all().item()
    assert (top_ids >= 0).all().item()
    torch.cuda.synchronize()
    tokenizer = AutoTokenizer.from_pretrained(c["model"], revision=c["tokenizer_revision"],
                                              trust_remote_code=False)
    low, high = c["synthetic_token_id_range"]
    vocab = set(tokenizer.get_vocab().values()) - set(tokenizer.all_special_ids)
    candidates = sorted(vocab.intersection(range(low, high)))
    if not candidates:
        raise ValueError("No valid synthetic token IDs")
    rng = random.Random(c["seed"])
    for row in rows:
        row["prompt"] = [rng.choice(candidates) for _ in range(row["prompt_tokens"])]
        row["prompt_sha256"] = hashlib.sha256(canonical(row["prompt"])).hexdigest()
    packages = sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())
    write_json(output, {"python": sys.version, "packages": packages,
                        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(0),
                        "compute_capability": list(torch.cuda.get_device_capability(0)),
                        "sampler_gpu_ops_smoke": "passed",
                        "sampling_backend_requested": c.get("sampling_backend", "auto"),
                        "local_cuda_home": os.environ.get("CUDA_HOME"),
                        "nvcc_version": command(["nvcc", "--version"]) if shutil.which("nvcc") else None,
                        "tokenizer_class": type(tokenizer).__name__, "workload": rows})


class GPUWatch:
    """Independent polling and monitoring threads; no learned thermal model."""
    FIELDS = ["uuid", "name", "driver_version", "power.draw", "power.limit",
              "clocks.current.graphics", "clocks.current.sm", "clocks.current.memory",
              "temperature.gpu", "utilization.gpu", "memory.used"]

    def __init__(self, directory, config, abort):
        self.c, self.abort = config, abort
        self.journal = Journal(directory / "telemetry.jsonl")
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.samples = []
        self.error = None
        self.started = time.monotonic()
        self.poller = threading.Thread(target=self.poll, daemon=True)
        self.monitor = threading.Thread(target=self.watch, daemon=True)

    @classmethod
    def query(cls):
        begin = time.monotonic()
        raw = command(["nvidia-smi", "--query-gpu=" + ",".join(cls.FIELDS),
                       "--format=csv,noheader,nounits"], timeout=2)
        rows = list(csv.reader(raw.splitlines()))
        if len(rows) != 1 or len(rows[0]) != len(cls.FIELDS):
            raise ValueError("Expected telemetry from exactly one GPU")
        d = dict(zip(cls.FIELDS, [v.strip() for v in rows[0]]))
        for field in cls.FIELDS[3:]:
            d[field] = float(d[field])
            if not math.isfinite(d[field]) or d[field] < 0:
                raise ValueError(f"Invalid telemetry: {field}")
        d.update(utc=utc(), monotonic_s=time.monotonic(), query_start_s=begin)
        return d

    def start(self):
        self.poller.start()
        self.monitor.start()

    def poll(self):
        while not self.stop.is_set():
            t = time.monotonic()
            try:
                d = self.query()
                self.journal.add(d)
                with self.lock:
                    self.latest = d
                    self.samples.append(d)
            except Exception as e:
                try:
                    self.journal.add({"utc": utc(), "monotonic_s": time.monotonic(), "error": str(e)})
                except OSError as logging_error:
                    self.error = f"telemetry_recording_failure: {logging_error}"
                    self.abort(self.error)
                    return
            self.stop.wait(max(0, self.c["telemetry_period_s"] - (time.monotonic() - t)))

    def watch(self):
        while not self.stop.wait(0.1):
            with self.lock:
                d = self.latest
            if time.monotonic() - (d["query_start_s"] if d else self.started) > self.c["telemetry_stale_s"]:
                self.error = "stale_gpu_telemetry"
            elif d and d["temperature.gpu"] >= self.c["temperature_stop_c"]:
                self.error = "measured_gpu_temperature_limit"
            if self.error:
                self.abort(self.error)
                return

    def close(self):
        self.stop.set()
        self.poller.join(timeout=4)
        self.monitor.join(timeout=2)
        if self.poller.is_alive() or self.monitor.is_alive():
            raise RuntimeError("Telemetry threads failed to stop")
        self.journal.close()


def integrate_energy(samples, start, end, max_gap):
    """Require brackets and integrate linearly to exact request-window bounds."""
    points = sorted((s["monotonic_s"], s["power.draw"]) for s in samples)
    if not points or start >= end or points[0][0] > start or points[-1][0] < end:
        raise ValueError("Missing telemetry brackets")
    energy = 0.0
    for (a, pa), (b, pb) in zip(points, points[1:]):
        lo, hi = max(start, a), min(end, b)
        if hi <= lo:
            continue
        if b - a > max_gap:
            raise ValueError("Telemetry gap exceeds validity limit")
        plo = pa + (pb - pa) * (lo - a) / (b - a)
        phi = pa + (pb - pa) * (hi - a) / (b - a)
        energy += (hi - lo) * (plo + phi) / 2
    if not math.isfinite(energy) or energy <= 0:
        raise ValueError("Nonfinite or nonpositive energy")
    return energy


def request_one(row, base, model, start, config, requests, events, stop):
    rec = {k: v for k, v in row.items() if k != "prompt"}
    scheduled = start + row["arrival_s"]
    rec.update(scheduled_monotonic_s=scheduled, status="not_dispatched")
    requests.add({**rec, "event": "scheduled"})
    response = None
    try:
        if stop.wait(max(0, scheduled - time.monotonic())):
            raise RuntimeError("Pilot stopped before dispatch")
        sent = time.monotonic()
        rec.update(dispatch_monotonic_s=sent, dispatch_utc=utc(), dispatch_lateness_s=max(0, sent - scheduled))
        requests.add({**rec, "event": "dispatched"})
        payload = {"model": model, "prompt": row["prompt"], "max_tokens": row["output_tokens"],
                   "min_tokens": row["output_tokens"], "ignore_eos": True, "temperature": 0,
                   "seed": config["seed"], "stream": True, "return_token_ids": True,
                   "stream_options": {"include_usage": True}, "add_special_tokens": False}
        req = urllib.request.Request(base + "/v1/completions", data=canonical(payload),
                                     headers={"Content-Type": "application/json"})
        response = urllib.request.urlopen(req, timeout=config["request_timeout_s"])
        first, last, token_count, usage, done, finish = None, None, 0, None, False, None
        for line in response:
            now = time.monotonic()
            if stop.is_set() or now - sent > config["request_timeout_s"]:
                raise TimeoutError("Request cancelled or exceeded total deadline")
            text = line.decode().strip()
            if not text.startswith("data:"):
                continue
            data = text[5:].strip()
            events.add({"trace_row": row["trace_row"], "monotonic_s": now, "data": data})
            if data == "[DONE]":
                done = True
                break
            chunk = json.loads(data)
            if chunk.get("error"):
                raise ValueError(str(chunk["error"]))
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                ids = choice.get("token_ids") or []
                if ids:
                    first = now if first is None else first
                    last = now
                    token_count += len(ids)
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
        ended = time.monotonic()
        if not done or first is None or usage is None or finish != "length":
            raise ValueError("Incomplete stream, missing usage/token IDs, or unexpected finish reason")
        if usage["prompt_tokens"] != row["prompt_tokens"] or usage["completion_tokens"] != row["output_tokens"]:
            raise ValueError("Server token accounting differs from source trace")
        if token_count != row["output_tokens"]:
            raise ValueError("Stream token count differs from source trace")
        rec.update(status="complete", completion_monotonic_s=ended, completion_utc=utc(), usage=usage,
                   streamed_tokens=token_count, ttft_s=first - sent, e2e_s=ended - sent,
                   client_stream_tpot_s=(last - first) / (token_count - 1) if token_count > 1 else None)
    except Exception as e:
        rec.update(status="failed" if "dispatch_monotonic_s" in rec else "not_dispatched",
                   error=str(e), completion_monotonic_s=time.monotonic(), completion_utc=utc())
    finally:
        if response is not None:
            response.close()
        requests.add({**rec, "event": "outcome"})
    return rec


def run(args):
    c, rows = load_config(args.config)
    run_id = "development-baseline-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
    directory = ROOT / "runs" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    trial = directory / "trials" / "0001"
    trial.mkdir(parents=True)
    print(f"Run directory: {directory}", flush=True)
    events = Journal(directory / "events.jsonl")
    requests = Journal(trial / "requests.jsonl")
    stream = Journal(trial / "stream.jsonl")
    stop = threading.Event()
    process_lock = threading.RLock()
    owned = []
    reasons, results = [], []
    watch = None
    started = time.monotonic()
    previous_handlers = {}
    state = {"run_id": run_id, "status": "preflight", "completed_trials": [], "next_trial_id": "0001"}

    def checkpoint(status):
        state.update(status=status, updated_utc=utc(), invalid_reasons=list(reasons))
        write_json(directory / "progress.json", state)
        events.add({"utc": utc(), "event": "checkpoint", "status": status})

    def abort(reason):
        first_abort = not stop.is_set()
        # Stopping computation must not depend on writable storage.
        stop.set()
        if first_abort:
            reasons.append(reason)
        with process_lock:
            for p in owned:
                if p.poll() is None:
                    try:
                        os.killpg(p.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
        if first_abort:
            try:
                events.add({"utc": utc(), "event": "abort", "reason": reason})
            except OSError as e:
                reasons.append(f"event_recording_failure: {e}")

    def interrupted(sig, frame):
        # Signal handlers must not acquire journal locks or perform file I/O.
        # Main/worker loops observe this flag and perform normal cleanup.
        reasons.append(f"signal_{sig}")
        stop.set()

    def child(cmd, log_path, timeout, env):
        with log_path.open("xb", buffering=0) as f:
            with process_lock:
                if stop.is_set():
                    raise RuntimeError("Aborted before child startup")
                p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
                owned.append(p)
            deadline = time.monotonic() + timeout
            while p.poll() is None:
                if stop.wait(0.1) or time.monotonic() > deadline:
                    abort("child_deadline_or_watchdog")
                    raise TimeoutError(f"Child failed/stopped; inspect {log_path.name}")
            if p.returncode:
                raise RuntimeError(f"Child exited {p.returncode}; inspect {log_path.name}")

    try:
        checkpoint("preflight")
        for sig in [signal.SIGINT, signal.SIGTERM]:
            previous_handlers[sig] = signal.signal(sig, interrupted)
        mount = json.loads(command(["findmnt", "-J", "-T", str(directory), "-o", "TARGET,SOURCE,FSTYPE,OPTIONS"]))["filesystems"][0]
        if "/networkvolumes/" not in mount["source"] or "rw" not in mount["options"].split(","):
            raise ValueError("Runner requires a verified writable RunPod network-volume mount")
        initial = GPUWatch.query()
        if initial["name"] != c["expected_gpu_name"]:
            raise ValueError("GPU differs from frozen pilot configuration")
        if initial["temperature.gpu"] >= c["temperature_stop_c"]:
            raise ValueError("GPU already exceeds pilot temperature limit")
        active = command(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"])
        if active and active != "No running processes found":
            raise ValueError("Existing GPU compute process; refusing to disturb another workload")
        # A venv's bin/python normally symlinks to a base interpreter. Preserve
        # its invocation path: resolving it bypasses pyvenv.cfg and site-packages.
        serving_python = Path(args.serving_python).expanduser().absolute()
        if not serving_python.is_file():
            raise ValueError("Serving interpreter missing; install runtime/serving first")
        files = [Path(args.config).resolve(), Path(__file__).resolve(), ROOT / c["trace"],
                 ROOT / "uv.lock", ROOT / "pyproject.toml", ROOT / ".python-version",
                 ROOT / "runtime/serving/pyproject.toml", ROOT / "runtime/serving/uv.lock",
                 ROOT / "runtime/serving/.python-version", ROOT / "scripts/setup_pilot_runtime.py"]
        hashes = {str(p.relative_to(ROOT)): digest(p) for p in files}
        diff = command(["git", "diff", "--binary", "HEAD"])
        (directory / "working-tree.patch").write_text(diff)
        manifest = {"run_id": run_id, "created_utc": utc(), "purpose": c["purpose"],
                    "git_commit": command(["git", "rev-parse", "HEAD"]),
                    "git_status": command(["git", "status", "--porcelain=v1"]),
                    "command": [sys.executable, *sys.argv], "client_python": sys.version,
                    "input_sha256": hashes, "dirty_diff_sha256": digest(directory / "working-tree.patch"),
                    "initial_gpu": initial, "storage": mount, "provider": args.provider_record,
                    "image_provenance_verified": bool(args.provider_record.get("image_digest_verified", False)),
                    "qualification_only": True, "benchmark_measurements_accepted": False,
                    "gpu_controls": "unchanged; no management setters invoked",
                    "configuration": c, "source_trace_rows": rows,
                    "serving_python": str(serving_python),
                    "os_release": (Path("/etc/os-release").read_text() if Path("/etc/os-release").exists() else None),
                    "kernel": command(["uname", "-a"]),
                    "cpu": command(["lscpu"]),
                    "memory": command(["cat", "/proc/meminfo"]),
                    "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                    "image_provenance_note": "Container digest unavailable unless supplied; runtime inventory and source/package hashes retained. Not an immutable-image benchmark.",
                    "clock_policy": "host defaults; requested fixed clocks not qualified",
                    "latency_targets": None, "confirmation_split": "unused"}
        write_json(directory / "manifest.json", manifest)
        # Keep source snapshots even when these files have not been committed.
        for p in files:
            dst = directory / "source" / p.relative_to(ROOT)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dst)
        (directory / "gpu-initial.txt").write_text(command(["nvidia-smi", "-q"]))
        (directory / "supported-clocks.txt").write_text(command(["nvidia-smi", "-q", "-d", "SUPPORTED_CLOCKS"]))
        events.add({"utc": utc(), "event": "manifest_saved", "manifest_sha256": digest(directory / "manifest.json")})
        env = os.environ.copy()
        # An automatic configuration must not inherit an unrecorded sampler override.
        env.pop("VLLM_USE_FLASHINFER_SAMPLER", None)
        overrides = serving_environment(c, initial["uuid"])
        env.update(overrides)
        manifest["serving_environment_overrides"] = overrides
        write_json(directory / "manifest.json", manifest)
        # Keep temp/compiled artifacts on the verified persistent mount too.
        temp = ROOT / "runs/cache/tmp"
        temp.mkdir(parents=True, exist_ok=True)
        env["TMPDIR"] = str(temp)
        watch = GPUWatch(directory, c, abort)
        watch.start()
        checkpoint("serving_probe")
        child([str(serving_python), str(Path(__file__).resolve()), "--config", str(Path(args.config).resolve()),
               "--probe-output", str(directory / "serving-probe.json")], directory / "serving-probe.log", 180, env)
        info = json.loads((directory / "serving-probe.json").read_text())["payload"]
        manifest["serving_environment"] = {k: v for k, v in info.items() if k != "workload"}
        manifest["workload_sha256"] = hashlib.sha256(canonical(info["workload"])).hexdigest()
        write_json(directory / "manifest.json", manifest)
        write_json(directory / "workload.json", info["workload"])
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        model_alias = run_id
        server_cmd = [str(serving_python), "-m", "vllm.entrypoints.openai.api_server", "--model", c["model"],
                      "--revision", c["model_revision"], "--tokenizer-revision", c["tokenizer_revision"],
                      "--served-model-name", model_alias, "--host", "127.0.0.1", "--port", str(port),
                      "--dtype", c["dtype"], "--max-model-len", str(c["max_model_len"]),
                      "--max-num-batched-tokens", str(c["max_num_batched_tokens"]),
                      "--max-num-seqs", str(c["max_num_seqs"]), "--gpu-memory-utilization", str(c["gpu_memory_utilization"]),
                      "--seed", str(c["seed"]), "--tensor-parallel-size", "1", "--generation-config", "vllm",
                      "--scheduling-policy", c["scheduling_policy"], "--enable-chunked-prefill",
                      "--no-enable-prefix-caching", "--no-async-scheduling", "--enforce-eager"]
        manifest["server_command"] = server_cmd
        write_json(directory / "manifest.json", manifest)
        checkpoint("server_startup")
        with (directory / "server.log").open("xb", buffering=0) as server_log:
            with process_lock:
                if stop.is_set():
                    raise RuntimeError("Aborted before server startup")
                server = subprocess.Popen(server_cmd, cwd=ROOT, env=env, stdout=server_log,
                                          stderr=subprocess.STDOUT, start_new_session=True)
                owned.append(server)
            base = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + c["startup_timeout_s"]
            ready = False
            while not stop.is_set() and time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"vLLM exited {server.returncode}; inspect server.log")
                try:
                    with urllib.request.urlopen(base + "/v1/models", timeout=1) as response:
                        models = json.load(response)
                    ready = any(m["id"] == model_alias for m in models["data"])
                    if ready:
                        break
                except (OSError, ValueError):
                    pass
                stop.wait(0.5)
            if not ready or stop.is_set():
                raise TimeoutError("Server startup deadline/watchdog")
            checkpoint("trial_running")
            trial_start = time.monotonic() + 1.0
            events.add({"utc": utc(), "event": "trial_start", "scheduled_start_monotonic_s": trial_start})
            executor = ThreadPoolExecutor(max_workers=len(rows))
            futures = [executor.submit(request_one, row, base, model_alias, trial_start, c, requests, stream, stop)
                       for row in info["workload"]]
            try:
                while not all(f.done() for f in futures):
                    if stop.wait(0.1) or time.monotonic() > trial_start + c["trial_timeout_s"]:
                        abort("trial_deadline_or_watchdog")
                        break
                # Killing the owned server cancels blocked HTTP readers on abort.
                if stop.is_set():
                    for p in owned:
                        if p.poll() is None:
                            try:
                                os.killpg(p.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                results = [f.result(timeout=10) for f in futures]
            finally:
                executor.shutdown(wait=True, cancel_futures=True)
            trial_end = max(r["completion_monotonic_s"] for r in results)
            # Collect a fresh sample after final drainage to bracket energy.
            end_wait = time.monotonic() + c["telemetry_stale_s"]
            while time.monotonic() < end_wait:
                with watch.lock:
                    latest = watch.latest
                if latest and latest["monotonic_s"] > trial_end:
                    break
                if stop.wait(0.1):
                    break
            if len(results) != len(rows) or any(r["status"] != "complete" for r in results):
                reasons.append("request_accounting_incomplete")
            if any(r.get("dispatch_lateness_s", math.inf) > c["max_dispatch_lateness_s"] for r in results):
                reasons.append("dispatch_lateness")
            energy = None
            try:
                with watch.lock:
                    samples = list(watch.samples)
                energy = integrate_energy(samples, trial_start, trial_end, c["telemetry_stale_s"])
            except ValueError as e:
                reasons.append(str(e))
            write_json(trial / "outcome.json", {"status": "invalid" if reasons else "complete",
                       "qualification_only": True, "invalid_reasons": list(reasons), "requests": results,
                       "energy_j_diagnostic": energy, "start_monotonic_s": trial_start,
                       "end_monotonic_s": trial_end,
                       "timing_note": "TTFT/TPOT use client receipt of token-ID chunks, not GPU token generation timestamps"})
            state["completed_trials"] = [{"trial_id": "0001", "status": "invalid" if reasons else "complete"}]
            state["next_trial_id"] = None
            checkpoint("trial_finished_cleanup_pending")
    except BaseException as e:
        abort(f"{type(e).__name__}: {e}")
        if not (trial / "outcome.json").exists():
            write_json(trial / "outcome.json", {"status": "interrupted" if any(r.startswith("signal_") for r in reasons) else "failed",
                       "invalid_reasons": list(reasons), "results_collected": results,
                       "partial_records": "requests.jsonl and stream.jsonl", "qualification_only": True})
    finally:
        # Stop only process groups created by this runner. Never stop the Pod.
        with process_lock:
            processes = list(owned)
        for p in processes:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for p in processes:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                p.wait(timeout=10)
        if watch:
            try:
                watch.close()
            except Exception as e:
                reasons.append(str(e))
        try:
            final_gpu = GPUWatch.query()
            write_json(directory / "gpu-final.json", final_gpu)
            if "initial" in locals() and (final_gpu["uuid"] != initial["uuid"] or final_gpu["power.limit"] != initial["power.limit"]):
                reasons.append("gpu_identity_or_power_limit_changed")
        except Exception as e:
            reasons.append(f"final_gpu_readback: {e}")
        # Cleanup/readback failures must also invalidate an otherwise complete trial.
        if reasons and (trial / "outcome.json").exists():
            outcome = json.loads((trial / "outcome.json").read_text())["payload"]
            if outcome["status"] == "complete":
                outcome["status"] = "invalid"
            outcome["invalid_reasons"] = list(reasons)
            write_json(trial / "outcome.json", outcome)
        if reasons:
            for completed in state["completed_trials"]:
                completed["status"] = "invalid"
        status = "failed" if reasons else "complete"
        checkpoint(status)
        elapsed = time.monotonic() - started
        rate = args.provider_record.get("hourly_rate_usd")
        summary = {"run_id": run_id, "status": status, "qualification_only": True,
                   "benchmark_measurements_accepted": False, "invalid_reasons": reasons,
                   "requests_expected": len(rows), "requests_complete": sum(r["status"] == "complete" for r in results),
                   "elapsed_s": elapsed, "runner_compute_cost_usd": elapsed / 3600 * rate if rate is not None else None,
                   "total_session_cost_usd": None,
                   "cost_note": "Runner compute only; installation, downloads before launch, storage, transfer and prior Pod time excluded. Total session bill must be reconciled separately.",
                   "retained_storage": str(directory), "finished_utc": utc()}
        write_json(directory / "summary.json", summary)
        events.add({"utc": utc(), "event": "finished", "status": status})
        for journal in [events, requests, stream]:
            journal.close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        # Verify all retained record bytes, including raw logs, after closing writers.
        records = {str(p.relative_to(directory)): digest(p) for p in sorted(directory.rglob("*")) if p.is_file()}
        write_json(directory / "checksums.json", records)
        for name, expected in records.items():
            if digest(directory / name) != expected:
                raise RuntimeError(f"Post-write checksum mismatch: {name}")
        print(json.dumps(summary, indent=2), flush=True)
    return 1 if reasons else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--serving-python", default=str(ROOT / "runtime/serving/.venv/bin/python"))
    parser.add_argument("--provider-record", type=Path, help="Optional JSON with Pod/DC/volume IDs, rate and verified image digest")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--probe-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.probe_output:
        probe(args.config, args.probe_output)
        return 0
    c, rows = load_config(args.config)
    if args.validate_only:
        print(json.dumps({"configuration": c, "selected_trace_rows": rows}, indent=2))
        return 0
    args.provider_record = json.loads(args.provider_record.read_text()) if args.provider_record else {}
    rate = args.provider_record.get("hourly_rate_usd")
    if rate is not None and (type(rate) not in [int, float] or not math.isfinite(rate) or rate < 0):
        parser.error("hourly_rate_usd must be finite and nonnegative")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
