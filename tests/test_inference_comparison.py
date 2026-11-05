"""Guard the audit against false matches and reduced-protocol results."""

import csv
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_inference_results import compare_csv, compare_json, run_comparison


def test_clock_action_difference_cannot_be_hidden_by_matching_latency():
    reference = {"mpc": {"clock": [2040.0], "ttft": 11.5}}
    generated = {"mpc": {"clock": [1125.0], "ttft": 11.5}}
    comparison = compare_json(reference, generated)
    assert comparison["checked_leaves"] == 2
    assert comparison["mismatched_values"] == 1
    assert comparison["differences_by_field"]["/mpc/clock/0"]["first_difference"]["generated"] == 1125.0


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_results_never_match(invalid):
    assert compare_json({"value": invalid}, {"value": invalid})["mismatched_values"] == 1


def test_missing_observations_and_boolean_type_changes_fail():
    comparison = compare_json(
        {"completed": True, "requests": [1, 2], "checksum": "abc"},
        {"completed": 1, "requests": [1]},
    )
    assert comparison["mismatched_values"] == 3


def test_csv_comparison_uses_controller_identity_and_reports_energy_difference(tmp_path):
    reference = tmp_path / "reference.csv"
    generated = tmp_path / "generated.csv"
    for path, rows in (
        (reference, [("maximum", 3506), ("mpc", 3341.784)]),
        (generated, [("mpc", 3341.180), ("maximum", 3506)]),
    ):
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["controller", "energy_j"])
            writer.writerows(rows)
    differences = [row for row in compare_csv(reference, generated) if not row["matches"]]
    assert len(differences) == 1
    assert differences[0]["controller"] == "mpc"
    assert differences[0]["delta"] == pytest.approx(-0.604)


def test_quick_protocol_cannot_qualify_as_full_reproduction(tmp_path):
    reference = tmp_path / "reference"
    generated = tmp_path / "generated"
    reference.mkdir()
    generated.mkdir()
    for path, protocol in ((reference, "full"), (generated, "quick")):
        (path / "textbook_results.json").write_text(json.dumps({
            "metadata": {"artifact_protocol": protocol},
        }))
    with pytest.raises(ValueError, match="full protocol"):
        run_comparison(reference, generated, tmp_path / "report")
