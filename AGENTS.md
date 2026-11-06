# Project guidelines

## Scope and current state

- Keep this repository independent of sibling folders. Use repository-relative
  paths and preserve data/source attribution.
- Read README.md and PLAN.MD before changing experiment behavior.
- Use single-GPU, on-demand RunPod Secure Cloud Pods. L4 is the reproduction
  reference; RTX A5000, RTX 4090, A40, L40S, and A100 PCIe 80 GB are comparison
  candidates. Follow the candidate matrix and staged acquisition plan in
  PLAN.MD. Historical Google Cloud records describe the reference experiment.
- Simulation Steps 3–5 are complete with a documented MPC rounding difference.
  Preserve the report and original artifacts when making later improvements.

## Versions and environment

- Use Python 3.12.14, uv 0.12.7, and `uv sync --locked --no-dev` for runtime
  setup. `uv.lock` is authoritative for Python packages; keep direct pins and
  the lockfile consistent. Do not upgrade dependencies as incidental cleanup.
- Keep the client/controller environment distinct from the serving runtime.
  Before measured hardware runs, qualify and freeze a Linux image by immutable
  digest and record CUDA, PyTorch, vLLM, Python, and host-driver versions.
- Pin the model and tokenizer revisions, workload subset, preprocessing,
  scheduler, controller configuration, and random seeds.
- Record actual installed versions and GPU identity for every acquisition.
  A fresh Pod or changed driver/image requires environment verification and
  calibration checks before comparing its results with previous acquisitions.
- The archived vLLM image/model identifiers are in the profile manifest. They
  are provenance, not evidence that an interactive RunPod image is qualified.

## Validation policy

- Unit tests are optional, not a required gate. Do not add tests automatically
  for documentation, data copying, or routine experiment wiring.
- Prioritize input checksums, reference comparisons, finite/complete outputs,
  request accounting, solver timing/fallback diagnostics, and workload replay.
- Use a bounded smoke run to qualify a new runtime before long measurements.
  Run only validation relevant to the change; avoid unnecessary repetitions.
- Preserve differences and failed or unfinished trials in reports. The strict
  comparator currently exits 1 for the documented MPC boundary-rounding
  difference. Do not loosen tolerances or rewrite references to obtain a pass.
- Keep simulated calibration results distinct from measured hardware results.
  Freeze development/confirmation splits and latency targets before confirmation.
- Cross-GPU comparisons use identical workload/model/precision and batching
  settings, with fresh per-device/host calibration and explicit operating limits.
  Preserve OOM, unsupported controls, and SLO failures. Keep larger-capacity
  configurations in a separate suite and report actual session cost alongside
  energy. Extend the L4-specific profile validation before accepting other GPUs.

## Run records and checkpoints

- Give each acquisition a unique UTC run ID; give each trial/retry a distinct
  directory. Never overwrite previous measured results.
- Before starting, save a manifest with Git commit and dirty diff, full command,
  config/data/lockfile hashes, seeds, image digest, installed versions, model
  revision, Pod/data-center/volume IDs, GPU UUID, and initial GPU settings.
- Write telemetry and request records incrementally and flush periodically.
  Checkpoint completed trials immediately with status and the next trial ID.
- Write checkpoints atomically with a schema/version and integrity hash. Retain
  the last valid checkpoint; include RNG/optimizer/controller state when a
  simulation or fitting stage supports continuation.
- Resume only with matching code, configuration, environment, and data. Treat
  an interrupted hardware trial as a separate failed/interrupted attempt and
  rerun it after conditioning; do not splice partial measurements into a valid
  uninterrupted trial.
- Preserve raw telemetry, requested/realized clocks, control decisions, solver
  status/timing, request timestamps, energy through final drainage, and invalid
  trial reasons. Derive summaries from these retained records.
- Checkpointing is required when implementing long hardware runners. The current
  simulator builder is not resumable and replaces its fixed output directory;
  archive a completed build under a unique run ID before running it again.

## RunPod execution and persistence

- Verify the actual persistent mount; `/workspace` alone does not identify its
  storage type. Use an attached network volume or external artifact storage for
  results that must survive Pod termination.
- Keep data, model caches, checkpoints, and manifests on verified persistent
  storage. A network volume is tied to its data center; plan GPU placement
  accordingly. Do not assume a stopped Pod reserves its GPU.
- Qualify telemetry, supported clock/power settings, write permissions, and
  readbacks on the actual host. Container root or CUDA availability alone does
  not establish management capability.
- Apply only supported, qualified GPU settings. Preserve prior settings and
  restore them on exit/interruption. Keep independent measured-temperature and
  stale-telemetry watchdogs; the archived thermal model is not accepted for live
  safety constraints.
- Resolve and validate clock-boundary handling before live MPC. Preserve the
  unchanged reproduction for comparison.
- Follow the user's authorized resource, budget, and runtime scope. Documentation
  edits do not imply deploying Pods or allocating storage.
- Before ending a session, copy results to retained storage, verify counts and
  checksums, and record artifact locations. Cleanup must preserve results and
  use the intended Pod/volume IDs.

## Git and data ownership

- Track code, documentation, configs, lockfiles, small fixed reference data, and
  compact reports. Keep imported `data/` and reference `artifacts/` immutable.
- Keep new raw runs, large checkpoints, model weights, caches, and credentials
  out of Git. Track remote artifact locations and SHA-256 hashes in summaries.
- Retain the archived `data/inference_serving/vllm.log`; the reference profile
  loader verifies its checksum. Other logs remain ignored.
- Do not commit credentials, private SSH keys, tokens, or environment secrets.
