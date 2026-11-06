# GPU Inference Optimization Reproduction Plan

## Objective

Reproduce the GPU frequency-control experiment described in
[`mpc-variants-reliability.md`](docs/reference/mpc-variants-reliability.txt), then
evaluate the controller on real GPUs, using L4 as the reproduction reference
and comparing the candidates below on the same workload. The controller adjusts GPU clocks as
inference demand changes to reduce energy while managing latency, queueing,
and temperature.

The first milestone is a local command that reproduces the four-controller
simulation and exports metrics and plots using the existing measurements.
This milestone requires no GPU access.

## Data Strategy and Execution Order

Start with the existing workload dataset and recorded measurements. Once the
simulation agrees with the reference, capture fresh GPU measurements on the
available L4 using the same workload, calibrate and validate the model, and
compare controllers on hardware. Compare the same recorded workload across GPU
candidates using fresh calibration for each device/host. Evaluate our own
traffic afterward.

| Data | Contents | Initial source and use |
|---|---|---|
| Request trace | Arrival times, prompt lengths, output lengths | Recorded Azure dataset, reused for simulation and initial hardware replay |
| GPU measurements | Power, temperature, realized clocks, throughput, latency | Archived L4 measurements for reproduction; fresh measurements from our RunPod L4 for live calibration and evaluation |

The request trace supplies timing and token counts, rather than original
prompt text. Hardware replay follows the reference's synthetic token-ID
prompts and forced output lengths. Record the replay construction, arrival
schedule, and development/confirmation splits so each controller receives the
same workload.

The first hardware acquisition is a fixed-clock baseline and calibration run
using the recorded workload. Collecting our own traffic follows the reference
reproduction and hardware comparison. This sequence helps distinguish
implementation differences, host performance differences, and workload effects.

## Available Infrastructure

- Use single-GPU, on-demand RunPod Secure Cloud Pods for hardware experiments.
  The L4 remains the reproduction reference; RTX A5000, RTX 4090, A40, L40S,
  and A100 PCIe 80 GB are comparison candidates. This replaces the earlier Google Cloud
  VM plan; historical Google Cloud acquisitions remain reference provenance.
- Record the Pod and data-center IDs, storage/volume IDs, GPU UUID, operating
  system, NVIDIA driver, immutable image digest, and access method during setup.
- Use Full SSH for standard file transfers and verified persistent storage for
  checkpoints and results. Follow the version/checkpoint guidelines in
  [AGENTS.md](AGENTS.md) and the [README](README.md#hardware-work).
- The Azure request trace is a public workload dataset used by the reference
  experiment. Replaying it on RunPod requires no Azure resources.
- Keep the recorded trace for the initial reproduction; evaluate additional
  workloads separately after establishing agreement with the reference.

### GPU candidate matrix

RunPod Secure Cloud on-demand listings verified **2026-11-05**; recheck actual
offers at launch. Prices exclude storage and transfers. Use one GPU per trial.

| Candidate | VRAM | Listed USD/hour | Initial purpose |
|---|---:|---:|---|
| [RTX A5000](https://www.runpod.io/gpu-models/rtx-a5000) | 24 GB | $0.27 | Live-stack smoke run, then common-workload measurement if qualified |
| [L4](https://www.runpod.io/gpu-models/l4) | 24 GB | $0.59 | Original hardware reference and primary baseline |
| [RTX 4090](https://www.runpod.io/gpu-models/rtx-4090) | 24 GB | $0.89 | First matched speed/energy/cost comparison against L4 |
| [A40](https://www.runpod.io/gpu-models/a40) | 48 GB | $0.59 | Common workload and larger-memory capacity experiments |
| [L40S](https://www.runpod.io/gpu-models/l40s) | 48 GB | $1.09 | Common workload and larger-memory throughput experiments |
| [A100 PCIe](https://www.runpod.io/gpu-models/a100-pcie) | 80 GB | $1.79 | Common workload and capacity experiments exceeding 48 GB |

Start the primary comparison with L4 and RTX 4090. Screen other candidates with
bounded pilots before committing to full repeats within the experiment budget.
Local simulator smoke checks need no rented GPU. Keep candidate selection,
actual availability, management permissions, and measured outcomes in the run
records; a listed candidate is not yet a qualified platform.

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

## test

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

- Qualify the RunPod L4: verify GPU visibility, telemetry, supported clock and
  power settings, and permissions to control them. GPU management access must
  be verified on the actual Pod before a long acquisition.
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
- Define experiment duration, storage locations, and cost scope for the Pod.
  Freeze the qualified image and package versions. Restore prior GPU settings
  after each experiment and keep workload cleanup separate from Pod lifecycle.
- Implement unique run/trial IDs, manifests, incremental telemetry, atomic
  checkpoints at trial boundaries, and verified artifact backups. Preserve
  interrupted attempts; resume from completed trials with matching inputs.
- Include GPU model/variant, UUID, CPU/RAM allocation, image/driver, telemetry
  capability, and profile identifiers in every run manifest so the same runner
  can qualify the cross-GPU candidates in Stage 3.

**Deliverable:** A hardware benchmark runner and documented acquisition protocol.

### 7. Capture fixed-clock baselines, calibrate, and validate

- Begin acquisition with fixed-clock runs of the recorded workload, measuring
  latency, completion, energy, temperature, and realized clocks.
- Measure service rates and power across clock settings, including mixed
  prefill/decode workloads, at the intended power cap.
- Use fresh calibration when hardware, power cap, or serving configuration
  differs from the recorded experiment. Keep the original 64.8 W simulation
  calibration separate from the later 40 W live development measurements.
- Collect and validate a calibration on the RunPod L4 before live MPC;
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

## Stage 3: Compare GPU Platforms

### 9. Run matched workloads across the candidate GPUs

- Qualify L4 and RTX 4090 first, then RTX A5000, A40, L40S, and A100 PCIe 80 GB
  as budget and availability allow. Use one GPU per trial and retain the exact
  model/variant and host identity.
- Freeze a common-workload suite that fits the intended 24 GB comparison:
  identical model/tokenizer revision, precision, context limit, request rows,
  prompt/output lengths, arrival schedule, scheduler, batching, and concurrency.
  Record KV capacity and memory-allocation settings. Use common latency targets.
- Use a common qualified image where supported. Record required architecture,
  kernel, or image differences rather than silently changing software settings.
  Log CPU/RAM allocation and region to expose host differences in the platform
  results. Preserve all environment and checkpoint rules in AGENTS.md.
- Collect a fresh calibration per GPU/host. Extend the current L4-specific
  measured-profile schema/validator before accepting other devices; retain the
  existing L4 validation and immutable reference bundle.
- Compare qualified default operating settings first. Evaluate maximum-clock,
  fixed-clock, governor, and MPC policies where controls are available, using
  each GPU's supported clock/power settings and its own validated model.
  Record operating limits explicitly; the L4 clock grid and 64.8 W cap are not
  assumed portable. Label hosts without controls as inference-only candidates.
- Pair identical workloads and schedules across GPUs; repeat trials and
  acquisitions for finalists. Preserve conditioning, request drainage, SLO
  failures, OOM, unsupported settings, interrupted trials, and fallback events.
  A workload that fails on one device must retain that outcome without quietly
  changing its model, precision, or batch settings.
- Report mean/p95 TTFT and TPOT, end-to-end latency, output tokens/second,
  completion time, request/SLO failures, memory use, measured GPU joules,
  joules/output token, realized clocks, temperatures, and solver behavior.
  Record telemetry sampling and availability for each device.
- Calculate timed-run compute cost from the actual rental rate and measured
  billable duration. Report total session cost separately, including setup,
  downloads, warm-up/conditioning, retries, storage, and transfers. Derive cost
  per completed workload and output token with overhead allocation stated.
- Show speed/energy/cost ratios against L4 on matched configurations and
  controller improvements against each GPU's own baseline. Rank energy/cost at
  a frozen common latency target and retain devices that miss it in the report.
- Add a separately named capacity suite for longer contexts, larger batches,
  or models requiring 48/80 GB. Keep configuration fixed within each comparison
  and label unsupported/OOM devices. Keep these outcomes separate from the
  common-workload results.

**Deliverable:** A cross-GPU results matrix with measured speed, energy, rental
cost, SLO compliance, memory/capability limits, uncertainty across repeats, and
links to per-device calibration and run records. Distinguish L4 reproduction,
hardware-platform comparisons, and larger-capacity experiments.

## Stage 4: Evaluate Our Own Workload

### 10. Capture and evaluate representative traffic

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

Unit tests are optional. Acceptance focuses on checksums, reference comparisons,
solver diagnostics, request completion, and relevant smoke/hardware measurements.
Future long hardware runners must preserve checkpoints and partial acquisitions
as specified in AGENTS.md.

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
- [ ] Candidate GPU/host capability matrix qualified.
- [ ] Matched-workload L4/RTX 4090 comparison completed.
- [ ] Remaining qualified GPU candidates measured and cross-GPU report published.
- [ ] Larger-memory capacity suite evaluated separately where applicable.
- [ ] Own workload captured and evaluated after the reference comparison.
