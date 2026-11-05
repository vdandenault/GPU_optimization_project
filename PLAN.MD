# GPU Inference Optimization Reproduction Plan

## Objective

Reproduce the GPU frequency-control experiment described in
[`mpc-variants-reliability.md`](docs/reference/mpc-variants-reliability.txt), then
evaluate the controller on a real GPU. The controller adjusts GPU clocks as
inference demand changes to reduce energy while managing latency, queueing,
and temperature.

The first milestone is a local command that reproduces the four-controller
simulation and exports metrics and plots using the existing measurements.
This milestone requires no GPU access.

## Data Strategy and Execution Order

Start with the existing workload dataset and recorded measurements. Once the
simulation agrees with the reference, capture fresh GPU measurements on the
available L4 using the same workload, calibrate and validate the model, and
compare controllers on hardware. Evaluate our own traffic afterward.

| Data | Contents | Initial source and use |
|---|---|---|
| Request trace | Arrival times, prompt lengths, output lengths | Recorded Azure dataset, reused for simulation and initial hardware replay |
| GPU measurements | Power, temperature, realized clocks, throughput, latency | Archived L4 measurements for reproduction; fresh measurements from our VM for live calibration and evaluation |

The request trace supplies timing and token counts, rather than original
prompt text. Hardware replay follows the reference's synthetic token-ID
prompts and forced output lengths. Record the replay construction, arrival
schedule, and development/confirmation splits so each controller receives the
same workload.

The first hardware acquisition is a fixed-clock baseline and calibration run
using the recorded workload. Collecting our own traffic follows the reference
reproduction and hardware comparison. This sequence helps distinguish
implementation differences, VM performance differences, and workload effects.

## Available Infrastructure

- Use the existing Google Cloud VM with an NVIDIA L4 for hardware experiments.
- Record the VM project, zone, instance name, operating system, NVIDIA driver,
  and access method during setup.
- The Azure request trace is a public workload dataset used by the reference
  experiment. Replaying it on the Google Cloud VM requires no Azure resources.
- Keep the recorded trace for the initial reproduction; evaluate additional
  workloads separately after establishing agreement with the reference.

## Stage 1: Reproduce the Recorded Simulation

### 1. Define the reproduction target

- Match the reference NVIDIA L4 calibration, Qwen2.5-7B-Instruct workload,
  recorded Azure request trace, and reduced 512-token interleaved
  chunked-prefill scheduler.
- Keep scheduling fixed to isolate the effect of GPU frequency control.
- Record the source revision, configuration, data provenance, and checksums.
- Freeze the reference trace subset and workload transformations for the
  simulation reproduction, and document the separate live replay schedules.

**Deliverable:** A documented experiment configuration and source inventory.

### 2. Set up this repository

The standalone import now includes the simulation modules, artifact builder,
three regression test modules, workload subsets, complete measured profile
bundle, recorded results, archival reference notes, and an import manifest.
All active paths resolve inside this repository. Generated results go to
`runs/inference_serving/`; imported reference results remain under `artifacts/`.
The environment is defined in `pyproject.toml` with its own `uv.lock`.

- Add a Python environment, configuration files, source modules, data
  directories, and experiment commands.
- Bring over the relevant simulator and controller implementation with source
  provenance and applicable attribution.
- Include the measured profile bundle and workload data needed to reproduce
  the experiment, preserving validation metadata and checksums.
- Keep reference data separate from generated results.

**Deliverable:** A reproducible local environment and documented entry points.

### 3. Reproduce the baseline simulation

- Run maximum-clock, reactive-governor, and offline-optimized policies against
  identical requests and initial conditions.
- Match the reference workload preparation and shifted-burst experiment.
- Verify request completion, latency calculations, queue behavior, and energy
  accounting through the final request drain.
- Compare the baseline metrics with the committed reference artifacts.

**Deliverable:** Baseline metrics and trajectories with a reproducibility check.

### 4. Reproduce the MPC controller

- Match the one-second control interval and ten-second prediction horizon.
- Forecast arrivals from the trailing 30-second rate and output work from the
  empirical output-length distribution.
- Match the normalized objective, constraints, and SLSQP settings: at most
  50 iterations and a tolerance of 1e-6.
- Apply only the first planned frequency, rounded downward to a profiled
  requested clock, then observe the state and replan.
- Reject infeasible, non-finite, or late solutions using the 0.8-second solve
  deadline and apply the reactive fallback.
- Record solve times, acceptance decisions, fallback reasons, and planned
  versus applied clocks.

**Deliverable:** MPC simulation with visible solver diagnostics and fallback
behavior.

### 5. Verify the published comparison

- Compare all four controllers on the same shifted request trace.
- Export latency, energy, queue, clock, and fallback metrics and plots.
- Report unfinished requests at the reporting horizon separately from final
  completion, and include post-horizon drain energy.
- Check numerical agreement with the recorded metrics using documented
  tolerances. Report any differences in solver timing or fallback behavior.

Recorded simulation targets:

| Controller | Energy (J) | Mean TTFT (s) | p95 TTFT (s) |
|---|---:|---:|---:|
| Maximum clock | 3,506.0 | 10.93 | 18.41 |
| Reactive governor | 3,363.3 | 12.51 | 20.01 |
| Offline schedule | 3,646.8 | 23.23 | 31.28 |
| MPC | 3,341.8 | 11.53 | 19.01 |

These are simulation outputs calibrated with measured L4 profiles. Every
controller misses the experimental TTFT limit for every request. The recorded
MPC run accepts 67 solves and invokes no fallback; timing-dependent outcomes
may differ on another machine.

**Deliverable:** One local command that generates the four-controller
comparison, metrics, plots, and a report of differences from the reference.

**Completion criterion:** Document agreement with the reference metrics and
resolve material implementation differences before starting GPU measurements.

## Stage 2: Evaluate on a Real GPU

### 6. Prepare a reproducible hardware benchmark

- Use the existing Google Cloud VM with its NVIDIA L4. Verify GPU visibility,
  supported clock and power settings, and permissions to control them.
- Inspect the existing driver, CUDA, and serving environment before changing
  dependencies; record the environment used for each run.
- Define and test handling of solver values near supported clock boundaries.
  The full reproduction exposed a value only 2.5e-12 MHz below 2040 MHz that
  the original downward quantizer maps to 1125 MHz. Preserve the original
  reproduction as a reference when introducing the hardware-controller policy.
- Pin the model revision, serving software, scheduler configuration, and
  workload.
- Prepare replay of the recorded workload with synthetic token-ID prompts and
  forced output lengths. Preserve trace row identities and document any arrival
  rescaling needed for the live development protocol.
- Record requested and realized clocks, board power, temperature, request
  timing, and controller decisions.
- Define trial validity checks, conditioning procedures, independent measured
  temperature watchdogs, and cleanup behavior.
- Define experiment duration, storage locations, and cost scope for the existing
  VM. Restore the prior clock and power settings after each experiment and keep
  workload cleanup separate from the VM lifecycle.

**Deliverable:** A hardware benchmark runner and documented acquisition protocol.

### 7. Capture fixed-clock baselines, calibrate, and validate

- Begin acquisition with fixed-clock runs of the recorded workload, measuring
  latency, completion, energy, temperature, and realized clocks.
- Measure service rates and power across clock settings, including mixed
  prefill/decode workloads, at the intended power cap.
- Use fresh calibration when hardware, power cap, or serving configuration
  differs from the recorded experiment. Keep the original 64.8 W simulation
  calibration separate from the later 40 W live development measurements.
- Collect and validate a calibration on the available L4 VM before live MPC;
  matching the GPU model alone does not establish matching performance or power.
- Fit the model on development measurements and validate on separate whole
  runs before enabling live MPC.
- Preserve raw telemetry, request records, configuration, and checksums in new
  run directories; keep the archived reference calibration unchanged.
- Evaluate forecasts using inputs available when the controller acts.
- Keep measured-temperature watchdogs independent of model predictions. The
  reference thermal models remain unaccepted for online advisory forecasts or
  hard thermal constraints.

**Deliverable:** Fixed-clock baseline results, a fresh measurement bundle, a
validated service/power model, and an explicit thermal-model acceptance decision.

**Completion criterion:** Accept service/power prediction quality on separate
whole runs before using the model to drive live MPC.

### 8. Run a controlled live comparison

- Use development measurements to select a fixed-clock baseline and freeze
  common latency targets before confirmation.
- Compare maximum clock, the selected fixed clock, the reactive governor, and
  MPC including fallback under steady and bursty arrivals.
- Use the recorded workload for this first controller comparison, keeping the
  workload construction consistent with the fixed-clock calibration runs.
- Use disjoint development and confirmation requests, identical requests and
  arrival times within each pair, counterbalanced method order, and repeated
  acquisitions.
- For fidelity to the reference's planned confirmation, use five paired
  repetitions per schedule across three independent acquisitions.
- Measure energy from the first scheduled arrival through final completion.
- Retain invalid trials, failed requests, unfinished work, and fallback events.
- Claim an efficiency improvement only when the compared policies meet both
  frozen latency targets and the required valid-pair criteria.

**Deliverable:** A hardware comparison report showing energy, latency,
variability, validity, and fallback behavior. MPC is not assumed to win.

The reference's broader live fixed-clock/governor/MPC comparison has not reached
confirmation. Completing this stage extends the recorded work.

## Stage 3: Evaluate Our Own Workload

### 9. Capture and evaluate representative traffic

- After completing the reference-workload hardware comparison, collect a
  representative request trace from our intended application, including arrival
  times and prompt/output lengths, with appropriate handling of prompt content.
- Document differences from the reference workload, such as burstiness, token
  lengths, and concurrency.
- Validate the serving model on this workload and recalibrate on development
  data if needed. Keep confirmation runs separate from tuning.
- Repeat the fixed-clock, governor, and MPC comparison with latency targets
  appropriate to the application, retaining the same reporting discipline.

**Deliverable:** A separate report showing whether the energy/latency tradeoffs
carry over to our traffic.

## Reference Implementation and Data

- [Case-study write-up](docs/reference/mpc-variants-reliability.txt)
- [Serving simulator](code/inference_serving.py)
- [Controllers and optimization](code/inference_control.py)
- [Replay and plotting](code/inference_replay.py)
- [Artifact builder](scripts/build_inference_artifacts.py)
- [Simulation reproduction instructions](artifacts/inference_serving/README.md)
- [Data provenance and calibration notes](data/inference_serving/README.md)
- [Recorded MPC metrics](artifacts/inference_serving/metrics_mpc.csv)
- [Live experiment status and protocol notes](docs/reference/live-control-development.txt)

## Progress

The full simulation and comparison completed on 2026-11-05. Baselines,
open-loop, DP, and FQI metrics match exactly. MPC latency, request records,
completion, and fallback diagnostics match; energy differs by -0.604415 J
(-0.0181%) because of one clock-boundary rounding decision. An instrumented
repeat and replay of reference clock decisions isolate the cause. Strict
comparison reports 229/233 CSV fields matching and does not claim an exact
MPC numerical match. All imported code/data/reference checksums are unchanged,
and 66 tests pass. See the [full report](reports/simulation_reproduction/README.md).
Live GPU evaluation and a robust clock-boundary policy remain to be completed.

- [x] Reproduction target documented.
- [x] Repository and environment set up.
- [x] Baseline simulations reproduced.
- [x] MPC reproduced with diagnostics and fallback (rounding difference documented).
- [x] Four-controller simulation report verified (strict differences retained).
- [ ] Hardware benchmark prepared.
- [ ] Recorded workload replayed on the L4 at fixed clocks; fresh measurements captured.
- [ ] Prediction models calibrated and validated.
- [ ] Live comparison completed and reported.
- [ ] Own workload captured and evaluated after the reference comparison.
