#!/usr/bin/env python3
"""Install the separate locked serving environment with bounded setup records."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import uuid

from run_hardware_pilot import ROOT, command, digest, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--uv", type=Path, default=ROOT / "runs/tools/uv-0.12.7/uv")
    p.add_argument("--timeout-s", type=int, default=600)
    p.add_argument("--link-mode", choices=["copy", "hardlink"], default="hardlink",
                   help="Hardlinks require cache and runtime on the same qualified filesystem")
    p.add_argument("--archive-partial-environment", action="store_true",
                   help="Preserve an interrupted installation before creating a clean environment")
    a = p.parse_args()
    if not 30 <= a.timeout_s <= 900:
        p.error("Setup deadline must be between 30 and 900 seconds")
    if sys.platform != "linux":
        p.error("The serving runtime is Linux x86_64 only")
    version = command([str(a.uv), "--version"])
    if not version.startswith("uv 0.12.7 ") and version != "uv 0.12.7":
        p.error("Use the project's uv 0.12.7 installation")
    mount = json.loads(command(["findmnt", "-J", "-T", str(ROOT), "-o", "TARGET,SOURCE,FSTYPE,OPTIONS"]))["filesystems"][0]
    if "/networkvolumes/" not in mount["source"]:
        p.error("Setup must reside on a verified RunPod network volume")
    run_id = "serving-setup-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
    directory = ROOT / "runs" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    archived_environment = None
    environment = ROOT / "runtime/serving/.venv"
    if a.archive_partial_environment and environment.exists():
        archived_environment = ROOT / "runs" / (run_id + "-partial-environment")
        environment.rename(archived_environment)
    cmd = [str(a.uv.resolve()), "sync", "--project", "runtime/serving", "--locked", "--no-dev"]
    env = os.environ.copy()
    env.update(UV_CACHE_DIR=str(ROOT / "runs/cache/uv"),
               UV_PYTHON_INSTALL_DIR=str(ROOT / "runs/tools/python"), UV_LINK_MODE=a.link_mode)
    manifest = {"run_id": run_id, "status": "installing", "created_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": command(["git", "rev-parse", "HEAD"]), "uv": version,
                "command": cmd, "deadline_s": a.timeout_s, "storage": mount,
                "link_mode": a.link_mode, "archived_partial_environment": str(archived_environment) if archived_environment else None,
                "input_sha256": {str(f.relative_to(ROOT)): digest(f) for f in [
                    ROOT / "runtime/serving/pyproject.toml", ROOT / "runtime/serving/uv.lock",
                    Path(__file__).resolve(), ROOT / "scripts/run_hardware_pilot.py"]}}
    write_json(directory / "progress.json", manifest)
    for name in manifest["input_sha256"]:
        destination = directory / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    print(f"Setup directory: {directory}", flush=True)
    start = time.monotonic()
    process = None
    try:
        with (directory / "install.log").open("xb", buffering=0) as log:
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            process.wait(timeout=a.timeout_s)
        if process.returncode:
            raise RuntimeError(f"uv sync exited {process.returncode}; inspect install.log")
        installed = command([str(a.uv), "pip", "freeze", "--python", str(ROOT / "runtime/serving/.venv/bin/python")], timeout=30)
        (directory / "installed-packages.txt").write_text(installed + "\n")
        actual_python = command([str(ROOT / "runtime/serving/.venv/bin/python"), "--version"])
        if actual_python != "Python 3.12.14":
            raise ValueError("Unexpected serving Python")
        manifest.update(status="complete", serving_python=actual_python)
    except BaseException as e:
        manifest.update(status="failed", error=f"{type(e).__name__}: {e}")
    finally:
        if process:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        manifest.update(elapsed_s=time.monotonic() - start, finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(directory / "progress.json", manifest)
        hashes = {str(f.relative_to(directory)): digest(f) for f in directory.rglob("*") if f.is_file()}
        write_json(directory / "checksums.json", hashes)
        print(json.dumps(manifest, indent=2), flush=True)
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
