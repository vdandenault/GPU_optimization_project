# Resume hardware experiments after RunPod support replies

Last updated: 2026-11-08. The serving qualification and an exploratory
default-settings baseline are complete. The next milestone is to qualify GPU
clock control, then collect fixed-clock measurements. Live MPC is later work.
Follow [AGENTS.md](AGENTS.md), [README.md](README.md), and [PLAN.MD](PLAN.MD).

## Where we stopped

- The four-request qualification completed and passed its audit:
  `runs/qualification-20261108T195948Z-5e261094`.
- The 32-request development baseline completed and passed its audit:
  `runs/development-baseline-20261108T204757Z-302f81e6`.
  Both audits verified 28 retained files, with exact request accounting and no
  reported errors.
- A power-setting permission probe requested the existing 145 W limit and
  failed with `Insufficient Permissions` (exit 4). Evidence is retained at
  `runs/control-permission-20261108T203333Z-25709`.
- A subsequent readback confirmed 145 W, 31 C, and no GPU compute processes.
- The container runs as UID 0 but lacks effective `CAP_SYS_ADMIN`. This is
  consistent with restricted host management; it does not independently prove
  the cause of every driver permission failure.
- **Graphics-clock locking has not been tested.** A denied power setter does
  not establish whether clock locking is supported or permitted.
- A human-support inquiry is pending. The initial automated response did not
  establish that management access can be enabled. No support resolution or
  ticket identifier has been recorded here yet.

Installing a Python package or using `sudo` inside the existing container does
not grant missing host privileges. Do not change dependencies to address this
permission error.

## Last observed environment

These are historical identifiers, not assumptions about a future session.

| Item | Observed value |
|---|---|
| Pod name | `intelligent_apricot_dragon-migration` |
| Pod ID | `8881glr5lzzh5s` |
| Data center | `EU-RO-1` |
| Network volume | `eqs8d57xtv`, mounted at `/workspace` |
| Repository on Pod | `/workspace/GPU_optimization_project` |
| GPU | NVIDIA RTX PRO 4000 Blackwell, 24,467 MiB reported |
| GPU UUID | `GPU-91faca7a-66bd-ac60-2dbd-d51bf92cdf9e` |
| Host driver | `595.91.07` |
| NVIDIA-SMI CUDA compatibility display | `13.2`; not the installed serving CUDA runtime version |
| Container image tag | `runpod/pytorch:1.5.0-cu1281-torch280-ubuntu2404` |
| Template ID | `hywnp7k0yl` |
| Deployed immutable image digest | Unverified |
| Power limits | Default/current 145 W; reported range 100–145 W |
| Maximum reported graphics clock | 3090 MHz; not a qualified fixed-clock setting |
| Actual rental rate | User-confirmed $0.58/hour; not invoice verified |
| Volume usage | User-reported 40 GB of 100 GB before the baseline; recheck |

The user reported no manual or startup-script clock/power changes on this Pod.
That statement is not a readback of the host's clock-lock configuration.

The retained tool is `runs/tools/uv-0.12.7/uv`; the system `uv` was 0.13.4 and
must not silently replace it. Client Python is 3.12.14. The serving runtime is
separate under `runtime/serving/`, locked to vLLM 0.28.0. Its installed inventory
and CUDA/PyTorch versions are retained in each acquisition manifest.

## Baseline to preserve

See the tracked [compact summary](reports/hardware_development/development-baseline-20261108T204757Z-302f81e6/summary.json).
Raw run directories are ignored by Git and live on the network volume; pulling
the repository onto another computer does not copy those records.

| Metric | Development baseline |
|---|---:|
| Completed requests | 32/32 |
| Prompt / output tokens | 81,516 / 709 |
| First scheduled arrival through final completion | 41.664 s |
| Diagnostic GPU board energy | 3274.480 J |
| Average board power over replay | 78.592 W |
| Output tokens/s, including arrival gaps | 17.017 |
| Mean / p95 TTFT | 3.829 / 6.453 s |
| Mean / p95 end-to-end latency | 5.063 / 8.126 s |
| Mean / p95 client-stream TPOT per request | 59.444 / 93.687 ms |
| Board J/output token, including prefill and idle | 4.618 |
| Runner duration, including startup | 773.023 s |
| Estimated runner compute cost | $0.12454; total session cost unknown |

The frozen configuration is
[`configs/hardware/blackwell-default-baseline-dev32.json`](configs/hardware/blackwell-default-baseline-dev32.json),
using trace rows 0–31 as development data with original arrivals. The model and
tokenizer are Qwen/Qwen2.5-7B-Instruct at commit
`acbd96531cda22292a3ceaa67e984955d3965282`. Serving uses FP16, seed 42,
8192-token context, four concurrent sequences, 512-token chunked-prefill
batching, FCFS, eager execution, no prefix caching or asynchronous scheduling,
85% GPU memory allocation, and the explicit PyTorch sampler override.

This is one exploratory acquisition with no explicit workload warm-up, no
verified immutable deployed image, and no frozen latency targets. Client stream
timings are not GPU generation timestamps. It is not an accepted benchmark,
calibrated profile, L4 reproduction, or evidence of energy savings. The 145 W
power limit is not a fixed GPU clock. Do not compare its energy directly with
the archived simulation as if the experiments were matched.

## When support replies

1. Retain their response, date, ticket reference, and exact supported access
   method in a compact repository note. Exclude credentials and private access
   details.
2. Obtain separate answers for programmatic graphics-clock locking, clock
   reset/restoration, and power-limit changes. Confirm allowed ranges, required
   permissions, and whether controls are available during execution.
3. If both controls are available, qualify them on the actual host before any
   sweep. A support statement alone is not a successful hardware preflight.
4. If clocks are available but power setters remain denied, a fixed-power
   clock experiment may be possible. Explicitly revise the protocol, record the
   host power limit, and qualify clocks independently; do not claim power-control
   access.
5. If only a provider-applied static power cap is available, that may support
   separate fixed-power trials but does not enable live clock control.
6. If neither is available, retain this host as inference-only. Ask for a
   specific supported offering before allocating another Pod. Follow the
   candidate matrix and budget scope in PLAN.MD; a different GPU or image alone
   does not guarantee management permission.

## Resume checklist

### 1. Verify the session and retained evidence

Read-only commands to run manually on the Pod, from the repository root:

```bash
git status --short --branch
runs/tools/uv-0.12.7/uv --version
.venv/bin/python --version
nvidia-smi
findmnt -T /workspace
.venv/bin/python scripts/audit_hardware_pilot.py \
  runs/development-baseline-20261108T204757Z-302f81e6
```

Check the actual volume quota in RunPod; shared-filesystem `df` capacity is not
the individual volume quota. Reconfirm Pod/data-center/volume IDs, rate, GPU
UUID, driver, and current settings. Do not reuse old provider records for a new
Pod. Any changed Pod, driver, or image requires environment verification and a
bounded qualification before comparisons. If nothing changed, do not repeat
expensive serving setup merely to reproduce already-audited evidence.

### 2. Freeze the measurement environment

Verify the actual deployed image by immutable digest. Resolving today's digest
of a mutable tag does not prove which image a running Pod used. Record serving
Python, CUDA, PyTorch, vLLM, model/tokenizer revisions, installed packages, and
host driver. Preserve the existing locks and separate client/serving runtimes.
Any changed runtime gets a new qualification record.

### 3. Implement a bounded control preflight

Keep the original observe-only pilot and its results unchanged. The development
baseline runner is also observe-only; neither runner currently implements GPU
setting restoration. Do not insert ad hoc setters into either one.

Before executing control changes, prepare a reviewable preflight that:

- Creates a unique UTC run/trial directory and saves code/diff, config/data/lock
  hashes, environment/provider identity, command, and initial GPU settings.
- Establishes the prior clock-lock state and a supported restoration mechanism;
  idle clocks alone do not reveal lock bounds. Stop if restoration cannot be
  established. Do not blindly reset unknown host settings.
- Checks for other GPU workloads, applies only supported settings, records setter
  return codes, and verifies requested versus realized operation under bounded
  load. Clock and power support are evaluated separately.
- Retains independent measured-temperature and stale-telemetry watchdogs,
  incremental flushed records, bounded deadlines, and signal/exception cleanup.
- Restores prior settings on success, failure, or interruption and verifies the
  final readback. Restoration failure invalidates the trial and must be surfaced.
- Writes atomic integrity checkpoints and preserves failed/interrupted attempts.

Do not assume the L4 clock grid or 64.8 W cap applies to this Blackwell GPU.

### 4. Collect fixed-clock development measurements

After preflight and environment qualification pass, freeze the protocol before
launch: supported clock levels, fixed power policy, workload, warm-up and
conditioning, deadlines, repetitions/order, and latency targets. Keep model,
precision, arrivals, batching, concurrency, and scheduler identical across
matched trials. Preserve rows 0–31 as development-only; freeze a disjoint
confirmation split before tuning or confirmation evaluation.

Collect a fresh default-settings baseline within that qualified protocol, then
the selected fixed-clock trials. Retain realized clocks, power, temperature,
memory, request/token accounting, and energy through final drainage. Preserve
OOM, unsupported controls, latency failures, and incomplete requests. Audit each
completed acquisition; report uncertainty across repeats and total session cost
separately from timed-run compute cost.

### 5. Calibrate before enabling live control

Extend the L4-specific profile validation with explicit device/host provenance
before accepting a Blackwell profile. Fit service/power models on development
measurements and validate on separate whole runs. Resolve clock-boundary
rounding for the hardware policy while preserving the original simulation and
its documented MPC difference. Keep measured-temperature safety independent of
the archived thermal fit. Only then evaluate reactive control and MPC, including
solver timing, fallback, and frozen latency-target checks.

## Persistence and collaboration

Preserve all existing successful, failed, and interrupted attempts. Before Pod
termination or migration, verify retained artifact locations, counts, and
checksums and back up records outside the Pod. Do not delete the network volume
or assume Git contains raw results. No stop, termination, migration, or new
resource allocation is authorized by this document.

The user runs VM terminal commands manually. Do not execute shell commands on
their behalf. Review diffs before committing or pushing; obtain explicit approval
before any assistant-created commit. Keep raw runs, caches, environments, model
weights, and credentials out of Git. Compact reports, source, configs, and this
handoff document are suitable for review and tracking.
