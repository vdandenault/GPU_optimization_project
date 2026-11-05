"""Record raw MPC solves and test replay of the reference clock decisions.

This diagnostic leaves the imported controller and quantizer unchanged.
"""

from __future__ import annotations

from bisect import bisect_right
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from build_inference_artifacts import plant_from_profile_manifest
from inference_control import (
    _MPCClockController,
    _select_frequency_experiment_window,
    shift_largest_burst,
)
from inference_serving import (
    chunked_prefill_scheduler,
    load_profile,
    load_workload,
    normalize_offered_load,
    simulate,
)


def main() -> None:
    data = ROOT / "data" / "inference_serving"
    profile = load_profile(data / "l4_profile.csv")
    evaluation, _ = normalize_offered_load(
        load_workload(data / "azure_code_evaluation.csv"), profile
    )
    expected = float(np.mean([request.output_tokens for request in evaluation]))
    nominal, _ = _select_frequency_experiment_window(
        evaluation, profile, expected_output_tokens=expected
    )
    workload = shift_largest_burst(
        nominal, profile, expected_output_tokens=expected
    )
    plant = plant_from_profile_manifest(profile, time_step_s=0.1)
    solves = []

    class ObservedMPC(_MPCClockController):
        def _solve(self, observation, arrival_rate):
            candidate, accepted, elapsed = super()._solve(observation, arrival_rate)
            solves.append({
                "time_s": observation.time_s,
                "raw_clock_mhz": candidate.tolist(),
                "quantized_clock_mhz": [
                    profile.quantize_clock(value, downward=True) for value in candidate
                ],
                "accepted": accepted,
                "solve_time_s": elapsed,
            })
            return candidate, accepted, elapsed

    controller = ObservedMPC(
        workload, plant, horizon_s=10.0,
        control_period_s=1.0, solve_time_limit_s=0.8,
    )
    observed = simulate(
        workload, plant, chunked_prefill_scheduler, controller,
        controller_name="MPC",
        scheduler_name="512-token interleaved chunked-prefill surrogate",
    )
    reference = json.loads(
        (ROOT / "artifacts/inference_serving/textbook_results.json").read_text()
    )["mpc"]["controllers"]["mpc"]
    updates = [
        (time, plan[0])
        for time, plan in zip(
            reference["planned_clock_start_time_s"], reference["planned_clock_mhz"]
        ) if time is not None and plan
    ]
    update_times = [time for time, _ in updates]

    def reference_clock(observation):
        index = bisect_right(update_times, observation.time_s + 1e-12) - 1
        return updates[index][1] if index >= 0 else profile.minimum_clock_mhz

    replay = simulate(
        workload, plant, chunked_prefill_scheduler, reference_clock,
        controller_name="Reference applied-clock replay",
    )
    result = {
        "solves": solves,
        "observed_mpc": observed.as_dict(stride=5),
        "reference_clock_replay": replay.as_dict(stride=5),
    }
    output = ROOT / "runs" / "inference_serving" / "mpc_diagnostic.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Recorded {len(solves)} solves; maximum solve time "
          f"{max(solve['solve_time_s'] for solve in solves):.4f} s.")
    print(f"Observed MPC energy: {observed.metrics.energy_j:.9f} J")
    print(f"Reference clock replay energy: {replay.metrics.energy_j:.9f} J")


if __name__ == "__main__":
    main()
