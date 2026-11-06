# GPU Inference Frequency Optimization

Standalone reproduction of the NVIDIA L4 inference-serving frequency-control
experiment. This repository contains its own simulator, controllers, request
traces, measured calibration bundle, tests, and recorded reference results.
No sibling repository or GPU is required for the simulation.

## Setup and validation

Use Python **3.12.14** and [uv](https://docs.astral.sh/uv/) **0.12.7**, the
versions used for the local reproduction. Install the locked runtime packages:

```bash
uv sync --locked --no-dev
```

The environment installation may download Python packages. The workload and
calibration data are already included; simulation does not download a model,
contact a cloud provider, or access a GPU.

Unit tests are optional. The usual acceptance checks are data integrity,
reference-metric comparisons, solver diagnostics, and request completion. To
use the existing tests when useful:

```bash
uv sync --locked --group dev
uv run --locked pytest -q
```

## Rebuild the simulation artifacts

For a reduced software smoke run:

```bash
MPLBACKEND=Agg uv run --locked --no-dev python scripts/build_inference_artifacts.py --quick
```

For the full original experiment suite:

```bash
MPLBACKEND=Agg uv run --locked --no-dev python scripts/build_inference_artifacts.py
```

The copied builder includes the related scheduling and fitted-Q-iteration
experiments as well as the four frequency controllers. The full suite is more
expensive than the quick smoke run. Quick results are labeled as such and are
not publication results. Solve times and fallback counts can vary by machine.

Generated JSON, CSV metrics, and SVG figures go to `runs/inference_serving/`.
Each run replaces outputs there. The imported results in
`artifacts/inference_serving/` remain separate and unchanged.

Compare a full run with the recorded reference:

```bash
uv run --locked --no-dev python scripts/compare_inference_results.py
```

The audit writes CSV and JSON evidence under `reports/simulation_reproduction/`
and exits with status 1 if differences remain. The first full run reproduces all
baseline metrics and MPC latency, with a 0.0181% MPC energy difference traced to
clock-boundary rounding. See the [full report](reports/simulation_reproduction/README.md)
for the comparison and diagnostic commands. Live GPU measurements are next.

## Repository contents

- [PLAN.MD](PLAN.MD): reproduction and hardware evaluation milestones.
- [AGENTS.md](AGENTS.md): environment, experiment, checkpoint, and validation guidelines.
- `code/`: serving simulator, frequency controllers, replay and plotting.
- `scripts/`: local simulation artifact builder.
- [data/inference_serving/README.md](data/inference_serving/README.md): included
  workload and measurement bundle, attribution, and limitations.
- [artifacts/inference_serving/README.md](artifacts/inference_serving/README.md):
  recorded comparison results.
- `tests/`: imported simulator, controller, and replay regression tests.
- `reports/`: compact comparison reports and diagnostic evidence to track in Git.
- [docs/reference/README.md](docs/reference/README.md): source provenance and
  archival notes.

## Hardware work

The platform is **single-GPU, on-demand RunPod Secure Cloud Pods**, with the
**L4 as the reference GPU** and the candidates below for comparison. Start with
the L4 and RTX 4090 on the same workload; add the other cards as acquisitions
progress. Connect by SSH, clone this repository, install the pinned
environment, and run the experiments. Choose **Full SSH with a public IP** for
`scp`/SFTP transfers; Basic SSH does not support those transfers.
[RunPod SSH setup](https://docs.runpod.io/pods/configuration/use-ssh).

As checked on **2026-11-05**, RunPod lists L4 on-demand compute at **$0.59/hour
on Secure Cloud** and **$0.44/hour on Community Cloud**, with regional
availability varying. Check the actual offer before launch and budget storage,
setup/download time, and transfers separately.
[L4 pricing](https://www.runpod.io/gpu-models/l4).

Start with fixed-clock replay of the included workload, collect fresh
calibration, and validate it before live MPC. The live runner and RunPod
environment qualification remain planned work. The local simulation does not
provision cloud resources.

### GPU candidates and comparison scope

RunPod Secure Cloud on-demand prices checked **2026-11-05**. The GPU links
are the source listings; actual offers, regions, storage, and transfers must
be recorded for each run.

| GPU | VRAM | Listed USD/hour | Experiment role |
|---|---:|---:|---|
| [RTX A5000](https://www.runpod.io/gpu-models/rtx-a5000) | 24 GB | $0.27 | Lower-cost live-stack smoke runs and a candidate for the common-workload comparison |
| [L4](https://www.runpod.io/gpu-models/l4) | 24 GB | $0.59 | Reference reproduction and primary energy/latency baseline |
| [RTX 4090](https://www.runpod.io/gpu-models/rtx-4090) | 24 GB | $0.89 | Primary comparison with L4 for speed, energy, and cost per completed workload |
| [A40](https://www.runpod.io/gpu-models/a40) | 48 GB | $0.59 | Common workload plus a lower-priced larger-memory candidate |
| [L40S](https://www.runpod.io/gpu-models/l40s) | 48 GB | $1.09 | Common workload and larger-memory throughput experiments |
| [A100 PCIe](https://www.runpod.io/gpu-models/a100-pcie) | 80 GB | $1.79 | Common workload and workloads requiring more than 48 GB; record the exact variant |

Begin with a bounded pilot on each candidate as budget and availability allow.
Use local CPU simulation for simulator installation checks; a rented smoke run
qualifies the live CUDA/serving stack. This list does not deploy six Pods or
assume every host supports GPU management.

For the common-workload comparison, freeze the model/tokenizer revisions,
precision, request trace, prompt/output lengths, arrival schedule, context limit,
batching/concurrency settings, and scheduler. Use one GPU per run and the same
latency targets. Verify a common supported software image; record any required
image/kernel differences explicitly. Equal VRAM does not establish identical
runtime memory use. Record OOM or unsupported configurations rather than
silently reducing a model or batch on one card.

Run larger-model, longer-context, or larger-batch experiments as a separately
named capacity suite on eligible 48/80 GB cards. Compare identical configurations
within that suite and report unsupported or OOM outcomes on smaller cards.

Collect a separate service/power profile, supported clock set, and thermal
baseline for each GPU/host. The current measured-profile validator is L4-specific;
extend it with explicit device provenance and validation before using non-L4
profiles. Preserve the original L4 bundle and validation. Qualify clock/power
controls independently for each candidate; if a host permits inference but
not control, label its measurements as inference-only.

Report per GPU:

- Mean and p95 TTFT/TPOT, end-to-end latency, output tokens/second, completion
  time, SLO violations, failed requests, and peak allocated/reserved memory.
- Measured board energy through final drainage, joules/output token, realized
  clocks, power, temperature, and controller fallback/deadline behavior where
  supported. Identify telemetry sampling differences between devices.
- Actual rental rate, timed-run compute cost, and total session cost including
  setup, downloads, warm-up, conditioning, retries, storage, and transfers.
  Derive cost per completed workload and per output token; keep measured GPU
  energy separate from rental cost.
- Speed, energy, and cost ratios relative to L4 on the matched workload, plus
  each controller's improvement over its own GPU's baseline. Freeze a common
  latency target before ranking efficiency and retain target failures.

Record CPU/RAM allocation, region, host/GPU identity, image digest, driver,
memory/KV allocation, and operating limits as comparison context. First compare
each GPU at its qualified default operating settings; then evaluate clock
policies using its own validated calibration. Share workload and targets across
cards while recording device-specific settings instead of assuming the L4's
clock levels or 64.8 W cap apply to every GPU. See the
[cross-GPU plan](PLAN.MD#stage-3-compare-gpu-platforms).

### GPU control preflight

This experiment needs telemetry and permission to change GPU clocks and power
settings. NVIDIA documents clock locking as requiring root. Qualify these
operations on the actual Pod; CUDA execution or root inside a container alone
does not establish that host GPU management is available. Verify supported
settings, readbacks, and restoration of prior settings with a bounded smoke
run before a long acquisition. If control is unavailable, record that limitation
and use a host that exposes the required controls for the clock experiment.
[NVIDIA management options](https://docs.nvidia.com/deploy/nvidia-smi/index.html).

Record the Pod/data-center IDs, GPU UUID and model, driver, CUDA runtime,
installed software, storage mount, requested settings, and realized telemetry.
Resolve the documented clock-boundary rounding issue before live control.

### Persistence and session workflow

RunPod releases a Pod's GPU when it stops. A later session may need a fresh
Pod; retain the environment specification and run records independently of
GPU allocation. [Pod lifecycle](https://docs.runpod.io/pods/manage-pods).

| Storage | What survives |
|---|---|
| Container disk | Temporary; lost on stop/restart |
| Pod volume disk, normally `/workspace` | Survives stop/restart; deleted with the Pod |
| Network volume | Independent of Pod termination; attach in its data center |

Confirm the actual mounted storage before writing valuable results. Back up
completed runs outside the Pod before termination.
[Storage behavior](https://docs.runpod.io/pods/storage/types).

For repeated sessions, a network volume can retain datasets, model weights,
checkpoints, and run records across fresh Pods. Standard storage is listed at
**$0.07/GB/month for the first TB**, so 100 GB is about **$7/month** while
retained. A volume's location constrains Pod placement and available GPUs.
[Network volumes](https://docs.runpod.io/storage/network-volumes).

Keep caches on the verified persistent mount and verify checkpoint backups
before ending a session. If transferring larger artifacts from Google Cloud
Storage, include applicable retrieval and outbound-transfer charges in the
session budget. The initial reference data is already in Git.
[GCS pricing](https://cloud.google.com/storage/pricing).

### Pin the experiment environment

The local runtime pins are Python 3.12.14, uv 0.12.7, NumPy 2.3.1, SciPy
1.16.1, scikit-learn 1.9.1, and Matplotlib 3.10.3. `uv.lock` pins transitive
dependencies. Use locked installation; change versions deliberately in a new
experiment configuration.

Before the first live acquisition, freeze a tested Linux Pod image by immutable
`@sha256:` digest, including its CUDA, PyTorch, and serving-stack versions.
Record the host driver separately; it is provided by the host. The interactive
template must provide the required SSH access. A RunPod image has **not yet
been qualified or selected** in this project.

The archived serving reference uses `vllm/vllm-openai:v0.28.0`, resolved to:

```text
vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14
```

It uses `Qwen/Qwen2.5-7B-Instruct` at revision
`acbd96531cda22292a3ceaa67e984955d3965282`. Preserve these identifiers for
provenance; validate the chosen interactive RunPod stack before treating it as
equivalent. Keep the Python client/controller environment and serving runtime
versions explicit rather than silently replacing the server's dependencies.
See [the profile manifest](data/inference_serving/profile_manifest.json).

### Experiment records and checkpoints

Use a unique UTC run ID for every acquisition and a separate directory for each
trial or attempt. Keep the following on persistent storage:

```text
runs/<run-id>/
  manifest.json       # Code/config/data hashes, environment, seed, provider IDs
  config.json         # Frozen workload, scheduler, controller, and limits
  progress.json       # Completed trials, next trial, last checkpoint, status
  trials/<trial-id>/  # Telemetry, request timings, solver decisions, outcome
  checkpoints/       # Resume state with hashes and schema/version metadata
  summary.json       # Aggregate results with invalid/unfinished trials retained
```

- Save the manifest and configuration before measurement. Record the Git
  commit and any dirty diff, lockfile and dataset hashes, model revision, image
  digest, package versions, GPU details, and initial conditions.
- Flush telemetry and request records periodically during a trial. At every
  trial boundary, atomically checkpoint progress and its outcome; use a
  temporary file followed by replacement and retain the last valid checkpoint.
- For long simulation or fitting runs, checkpoint meaningful completed stages
  or iterations, including RNG state and algorithm state needed for continuation.
- Resume only when code, configuration, environment, and input hashes match.
  Restart an interrupted hardware trial as a new attempt after conditioning;
  keep partial telemetry and label the earlier attempt interrupted or invalid.
- Preserve the prior GPU settings and restore them during normal cleanup or
  interruption handling. Include request drainage in energy measurement.
- Verify copied file counts and hashes before stopping or terminating the Pod.
  Keep compact summaries and artifact locations/checksums in Git; keep bulky
  telemetry, checkpoints, and weights in persistent storage.

These are requirements for the forthcoming hardware runner. The current
simulation builder writes to the fixed `runs/inference_serving/` directory and
does not implement resumable checkpoints. Archive each completed output under
its run ID before another build replaces it.

### Alternatives if capacity is unavailable

| Option | Use and comparison |
|---|---|
| RunPod Community Cloud | Another L4 capacity pool; recheck the offer and GPU-control access. |
| [Vast.ai](https://docs.vast.ai/guides/instances/connect/ssh) | Compare live L4 offers, host reliability, SSH, storage, and bandwidth costs. |
| [Scaleway L4 VM](https://www.scaleway.com/en/pricing/gpu/) | A full VM alternative; L4-1-24G is listed at €0.79/hour before tax in Paris as checked on 2026-11-05. |
| [AWS EC2 G6](https://aws.amazon.com/ec2/instance-types/g6/) | Another L4 VM option; check regional capacity and [G/VT quotas](https://docs.aws.amazon.com/ec2/latest/instancetypes/ec2-instance-quotas.html), whose published default is zero. |

For VM alternatives, account for retained disks and IPs separately from compute;
follow the provider's documented shutdown/billing behavior, such as
[Scaleway's power-off guidance](https://www.scaleway.com/en/docs/instances/how-to/power-off-instance/).
The RTX 4090 and other GPU models are planned comparison candidates above.
Their measurements form a separate hardware comparison alongside the L4
reproduction, with fresh calibration for each model/host.

If capacity searches become a recurring problem, [SkyPilot](https://docs.skypilot.co/en/v0.10.2/docs/)
provides an optional path to provisioning with failover across configured
infrastructure. Qualify every selected GPU/host and keep its calibration and
measurement records separate.

## Version control

Commit the source, tests, documentation, `pyproject.toml`, `uv.lock`, reference
data under `data/`, and recorded results under `artifacts/`. The imported bundle
is about 10 MB; its largest file is `l4_telemetry.csv` at about 8.3 MB. Normal
Git is sufficient for the current files. The existing PNG diagrams can also be
tracked.

`.gitignore` excludes generated `runs/`, virtual environments, caches, secrets,
ordinary logs, and profiling reports. The archived `data/inference_serving/vllm.log`
has an explicit exception because profile validation checks its digest.

Keep future large or repeated raw acquisitions in object storage, with their
locations, checksums, configurations, and compact result summaries recorded in
Git. Promote a small, fixed reference dataset deliberately when needed.
