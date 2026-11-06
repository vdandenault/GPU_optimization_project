#!/usr/bin/env python3
"""Audit retained pilot bytes, request accounting and recorded failure status."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from run_hardware_pilot import canonical, digest
import hashlib


def read_record(path):
    d = json.loads(path.read_text())
    if d["schema_version"] != 1 or hashlib.sha256(canonical(d["payload"])).hexdigest() != d["integrity_sha256"]:
        raise ValueError(f"Integrity/schema failure: {path}")
    return d["payload"]


def audit(directory):
    errors = []
    checksums = read_record(directory / "checksums.json")
    actual = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    if actual != set(checksums) | {"checksums.json"}:
        errors.append("File inventory differs from checksum inventory")
    for name, expected in checksums.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file() or digest(path) != expected:
            errors.append(f"File checksum/path failure: {name}")
    summary = read_record(directory / "summary.json")
    progress = read_record(directory / "progress.json")
    if progress["status"] != summary["status"]:
        errors.append("Checkpoint and summary statuses differ")
    if not summary["qualification_only"] or summary["benchmark_measurements_accepted"]:
        errors.append("Qualification falsely promoted to benchmark")
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = read_record(manifest_path)
        for name, expected in manifest["input_sha256"].items():
            path = directory / "source" / name
            if not path.is_file() or digest(path) != expected:
                errors.append(f"Source snapshot differs from manifest: {name}")
    outcomes = []
    request_file = directory / "trials/0001/requests.jsonl"
    for line in request_file.read_text().splitlines():
        rec = json.loads(line)
        if rec["event"] == "outcome":
            outcomes.append(rec)
    if len({r["trace_row"] for r in outcomes}) != len(outcomes):
        errors.append("Duplicate request outcome")
    complete = [r for r in outcomes if r["status"] == "complete"]
    streams = defaultdict(list)
    for line in (directory / "trials/0001/stream.jsonl").read_text().splitlines():
        rec = json.loads(line)
        streams[rec["trace_row"]].append(rec)
    for result in complete:
        chunks = streams[result["trace_row"]]
        if not chunks or chunks[-1]["data"] != "[DONE]":
            errors.append(f"Complete request lacks DONE: {result['trace_row']}")
            continue
        payloads = [json.loads(r["data"]) for r in chunks if r["data"] != "[DONE]"]
        token_count = sum(len(c.get("token_ids") or []) for p in payloads for c in p.get("choices", []))
        usages = [p["usage"] for p in payloads if p.get("usage")]
        if token_count != result["output_tokens"] or not usages or usages[-1] != result["usage"]:
            errors.append(f"Stream/usage accounting differs: {result['trace_row']}")
        if result["usage"]["prompt_tokens"] != result["prompt_tokens"] or result["usage"]["completion_tokens"] != result["output_tokens"]:
            errors.append(f"Source token counts differ: {result['trace_row']}")
        for key in ["ttft_s", "e2e_s", "dispatch_lateness_s"]:
            if not math.isfinite(result[key]) or result[key] < 0:
                errors.append(f"Invalid latency: {result['trace_row']} {key}")
        if result["e2e_s"] < result["ttft_s"]:
            errors.append("TTFT exceeds end-to-end latency")
    if summary["status"] == "complete":
        if len(complete) != summary["requests_expected"] or len(outcomes) != len(complete):
            errors.append("Successful pilot has missing/failed requests")
        if summary["requests_complete"] != len(complete):
            errors.append("Summary request count differs")
        if summary["invalid_reasons"]:
            errors.append("Successful pilot contains invalid reasons")
    elif not summary["invalid_reasons"]:
        errors.append("Failed pilot lacks reason")
    return {"run_id": summary["run_id"], "recorded_status": summary["status"],
            "files_verified": len(checksums), "request_outcomes": len(outcomes),
            "complete_requests": len(complete), "errors": errors,
            "audit_passed": not errors, "qualification_only": True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_directory", type=Path)
    a = p.parse_args()
    result = audit(a.run_directory)
    print(json.dumps(result, indent=2))
    return 0 if result["audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
