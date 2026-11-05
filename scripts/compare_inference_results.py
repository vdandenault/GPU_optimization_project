"""Audit a full simulation against the immutable imported reference outputs."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REL_TOL = 1e-6
ABS_TOL = 1e-8
METRICS = (
    "metrics_mpc.csv", "metrics_open_loop.csv", "metrics_dp.csv", "metrics_fqi.csv",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equal_value(expected, actual) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(expected) is type(actual) and expected == actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isfinite(expected) and math.isfinite(actual) and math.isclose(
            expected, actual, rel_tol=REL_TOL, abs_tol=ABS_TOL
        )
    return type(expected) is type(actual) and expected == actual


def compare_json(expected, actual) -> dict:
    """Compare all leaves, including request timings, clocks, and MPC plans."""
    checked = 0
    exact = 0
    groups = {}

    def difference(path, left, right):
        group = "/".join(path.split("/")[:5])
        entry = groups.setdefault(group, {
            "mismatched_values": 0, "max_absolute_difference": None,
            "first_difference": {"path": path, "reference": left, "generated": right},
        })
        entry["mismatched_values"] += 1
        if type(left) in (int, float) and type(right) in (int, float):
            delta = abs(right - left)
            entry["max_absolute_difference"] = max(
                delta, entry["max_absolute_difference"] or 0.0
            )

    def visit(left, right, path=""):
        nonlocal checked, exact
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(left.keys() | right.keys()):
                child = f"{path}/{key}"
                if key not in left or key not in right:
                    difference(child, "<present>" if key in left else "<missing>",
                               "<present>" if key in right else "<missing>")
                else:
                    visit(left[key], right[key], child)
        elif isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                difference(f"{path}/length", len(left), len(right))
            for index, (x, y) in enumerate(zip(left, right)):
                visit(x, y, f"{path}/{index}")
        else:
            checked += 1
            if type(left) is type(right) and left == right:
                exact += 1
            if not equal_value(left, right):
                difference(path, left, right)

    visit(expected, actual)
    return {
        "checked_leaves": checked,
        "exactly_equal_leaves": exact,
        "mismatched_values": sum(g["mismatched_values"] for g in groups.values()),
        "differences_by_field": groups,
    }


def compare_csv(reference: Path, generated: Path) -> list[dict]:
    with reference.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        expected_rows = list(reader)
    with generated.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or []) != set(fields or []):
            raise ValueError(f"CSV columns differ: {generated.name}")
        actual_rows = list(reader)

    def indexed(rows):
        result = {}
        for index, row in enumerate(rows):
            key = row.get("controller", str(index))
            if key in result:
                raise ValueError(f"Duplicate row key: {key}")
            result[key] = row
        return result

    expected = indexed(expected_rows)
    actual = indexed(actual_rows)
    if expected.keys() != actual.keys():
        raise ValueError(f"CSV row keys differ: {generated.name}")
    comparisons = []
    for key, row in expected.items():
        for metric in fields:
            left, right = row[metric], actual[key][metric]
            delta = ""
            try:
                x, y = float(left), float(right)
            except ValueError:
                matches = left == right
            else:
                matches = equal_value(x, y)
                delta = y - x
            comparisons.append({
                "file": reference.name, "controller": key, "metric": metric,
                "reference": left, "generated": right, "delta": delta,
                "matches": matches,
            })
    return comparisons


def run_comparison(reference: Path, generated: Path, output: Path) -> dict:
    reference_path = reference / "textbook_results.json"
    generated_path = generated / "textbook_results.json"
    expected = json.loads(reference_path.read_text())
    actual = json.loads(generated_path.read_text())
    if expected["metadata"]["artifact_protocol"] != "full" or actual["metadata"]["artifact_protocol"] != "full":
        raise ValueError("Both artifacts must use the full protocol; quick runs cannot qualify.")
    trajectory = compare_json(expected, actual)
    comparisons = [row for name in METRICS for row in compare_csv(reference / name, generated / name)]
    mismatches = [row for row in comparisons if not row["matches"]]
    imports = json.loads((ROOT / "docs/reference/import_manifest.json").read_text())
    preserved = {}
    for entry in imports["files"]:
        if entry["path"].startswith(("data/", "artifacts/", "code/")):
            preserved[entry["path"]] = digest(ROOT / entry["path"]) == entry["sha256_at_import"]
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_passed": not mismatches and not trajectory["mismatched_values"] and all(preserved.values()),
        "relative_tolerance": REL_TOL, "absolute_tolerance": ABS_TOL,
        "csv_fields_compared": len(comparisons),
        "csv_fields_matching": len(comparisons) - len(mismatches),
        "csv_mismatches": mismatches,
        "json_comparison": trajectory,
        "imported_code_data_and_reference_unchanged": preserved,
        "fqi_protocol": actual["fqi"]["protocol"],
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in (
                "numpy", "scipy", "scikit-learn", "matplotlib",
            )},
            "git_head": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "uv_lock_sha256": digest(ROOT / "uv.lock"),
        },
        "artifact_sha256": {
            "reference_json": digest(reference_path), "generated_json": digest(generated_path),
            **{name: digest(generated / name) for name in METRICS},
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (output / "metric_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    print(f"CSV: {len(comparisons) - len(mismatches)}/{len(comparisons)} fields match; "
          f"JSON: {trajectory['mismatched_values']} values outside tolerance.")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / "artifacts/inference_serving")
    parser.add_argument("--generated", type=Path, default=ROOT / "runs/inference_serving")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/simulation_reproduction")
    args = parser.parse_args()
    summary = run_comparison(args.reference, args.generated, args.output)
    if not summary["comparison_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
