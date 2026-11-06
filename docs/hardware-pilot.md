# Bounded RunPod environment qualification

The pilot checks the serving stack on the existing RTX PRO 4000 Blackwell Pod.
It is an environment qualification attempt, not an L4 reproduction, a calibrated
GPU profile, a controller comparison, or a confirmation measurement. The
original simulation, measured L4 bundle, and rounding discrepancy remain intact.
The workload is the CC BY 4.0 Azure 2023 code trace; its authors, source,
license and checksum provenance remain in the [data attribution](../data/inference_serving/README.md).

The client uses the standard library in the project's Python 3.12.14 environment.
`runtime/serving/pyproject.toml` and its separate `uv.lock` define the Linux
x86_64 serving environment with vLLM 0.28.0. `uv sync --locked --no-dev` installs
that environment independently of the client and the Pod template's PyTorch.
The actual serving Python, PyTorch, CUDA runtime, tokenizer and complete package
inventory are recorded before replay. Do not infer serving compatibility from
the template's successful CUDA smoke check.

## Frozen pilot scope

`configs/hardware/blackwell-qualification-torch-sampler.json` fixes:

- Qwen/Qwen2.5-7B-Instruct and tokenizer commit
  `acbd96531cda22292a3ceaa67e984955d3965282`.
- Azure evaluation trace rows `[0, 1, 2, 3]`, with original token lengths and
  arrival offsets relative to row 0. No clipping or arrival dilation occurs.
  All four rows belong to development; confirmation rows remain unused.
- Synthetic prompt IDs sampled with Python seed 42 from valid, non-special
  tokenizer IDs in `[100, 20000)`. Full prompts and their hashes are retained.
- Forced output lengths using `min_tokens`, `max_tokens`, and `ignore_eos`;
  streamed token IDs and final usage must agree exactly with the source rows.
- FP16, one GPU, an 8192-token context, four maximum concurrent sequences,
  512-token chunked-prefill batching, FCFS, no prefix caching, no asynchronous
  scheduling, eager execution, and 85% vLLM GPU memory allocation.
- FlashInfer sampling disabled with the supported
  `VLLM_USE_FLASHINFER_SAMPLER=0` override. The initial automatic sampler in
  `blackwell-qualification.json` failed during engine warm-up with
  `FlashInfer requires GPUs with sm75 or higher` on this SM120 device. That
  configuration and failed acquisition remain separate. The new configuration
  selects vLLM's alternative sampling path explicitly; it does not patch packages
  or change the locked dependencies. Environment overrides are in the manifest.
- Existing host GPU settings. The runner issues no clock/power setters and
  records this as unqualified default-settings operation. OOM/unsupported
  configurations are failures, not reasons to silently alter the protocol.
- A 180-second probe deadline, 900-second server startup/download deadline,
  180-second replay deadline, and 120-second per-request deadline. The separate
  package-installation deadline is 600 seconds. No automatic retries occur.

## Commands on the Pod

From the repository on the verified RunPod network volume:

```bash
cd /workspace/GPU_optimization_project
runs/tools/uv-0.12.7/uv sync --locked --no-dev
.venv/bin/python scripts/run_hardware_pilot.py --validate-only
.venv/bin/python scripts/setup_pilot_runtime.py --timeout-s 600
.venv/bin/python scripts/run_hardware_pilot.py
```

The default config is the named Torch-sampler qualification config. Pass
`--config configs/hardware/blackwell-qualification.json` only to reproduce the
original automatic-sampler attempt; its operational failure is retained.

Audit the printed run directory after the runner exits:

```bash
.venv/bin/python scripts/audit_hardware_pilot.py runs/qualification-<UTC>-<id>
```

The audit verifies file counts/hashes, manifest source snapshots, terminal
checkpoint consistency, and successful-request accounting against the retained
raw token-ID streams. An audit can pass for a faithfully recorded failed attempt;
its `recorded_status` remains failed.

The last command downloads the pinned model as needed. Hugging Face, compiler,
temporary-file and uv caches reside under ignored `runs/` on the network volume.
Server access binds only to localhost and does not add exposed Pod ports.
Setup uses hardlinks between the cache and runtime on the same persistent
filesystem. Qualify hardlink support first; `--link-mode copy` is available when
needed. For an interrupted installation, use a new setup attempt with
`--archive-partial-environment` to retain the incomplete environment and install
into a fresh directory. This is package-installation recovery, not continuation
of a hardware trial. Never start a second setup while the first remains active.
Once the replacement runtime is verified, the incomplete package environment
may be removed to reclaim storage, with the deletion recorded separately. Retain
the setup manifests/logs, completed runtime, model cache, and all acquisition
records. The initial Pod hit its volume quota during model loading; the shared
filesystem's `df` capacity did not describe the individual volume's quota.

Optional `--provider-record runs/provider.json` accepts a JSON object with the
Pod ID, data-center ID, volume ID, image digest, verification evidence, and actual
hourly rate. Only mark `image_digest_verified` true when verified against the
deployed image. Missing metadata remains explicit. Do not store API tokens or
credentials in this record. The current user-approved qualification can proceed
without a digest; it is labeled as missing immutable-image provenance. Measured
comparisons still require a qualified immutable image under the project policy.
The runner intentionally cannot promote a smoke attempt to a benchmark.

## Records and failure handling

Each invocation creates a unique `runs/qualification-<UTC>-<id>/` directory and
`trials/0001/`. Setup attempts have separate `runs/serving-setup-*` directories.
The acquisition manifest includes the command, source snapshots and hashes,
Git commit and dirty diff, configuration, model/tokenizer revisions, runtime
inventory, OS/kernel/CPU/RAM, GPU UUID/driver/settings, mount details, and supplied
provider metadata. Uncommitted runner sources are copied and hashed too.

`telemetry.jsonl`, `events.jsonl`, trial request records and raw SSE responses
are flushed and fsynced incrementally. The manifest precedes CUDA probing and
inference; the serving inventory and exact server command are added before
starting the server. Checkpoints contain a schema version and payload SHA-256,
use atomic replacement and retain their previous valid version. The runner
does not resume a partial trial: every retry must be a new invocation/directory.

Two independent threads poll measured GPU telemetry and enforce stale-sample
and measured-temperature limits. The pilot stops at 77 C or after five seconds
without fresh usable telemetry. Nonfinite or missing telemetry is never accepted
as a valid sample. A watchdog or deadline terminates only process groups created
by the runner. Cleanup does not terminate the Pod or delete the volume. Initial
and final GPU readbacks are retained; the runner never changes GPU settings.
Failure to persist telemetry also stops computation immediately; that stop does
not depend on being able to write an abort event to the full filesystem.

Request outcomes retain failed and undispatched requests. A complete pilot
requires all four requests with exact token accounting, a final `[DONE]`, and
dispatch lateness at most 100 ms. Energy diagnostics use trapezoidal integration
between exact first-arrival and final-response times, with telemetry brackets
and a maximum five-second gap. These are diagnostic board-energy observations,
not accepted benchmark measurements. Client stream TTFT/TPOT do not establish
per-token GPU generation timestamps. There is no SLO acceptance claim.

Every terminal outcome writes a summary, integrity checkpoint, and verified
file-count/hash inventory. Logs and results remain on the verified network
volume even after failure. Before terminating a Pod, copy the entire run and
setup directories to retained storage outside that Pod and verify `checksums.json`.
The network-volume location itself is recorded as the retained artifact location.
No automatic Pod or storage lifecycle action is taken.

If a rate is supplied, the summary estimates runner-duration compute cost only.
Total session cost, including package installation, earlier Pod time, downloads,
warm-up, storage, transfer and retries, remains unknown until reconciled with
provider billing; never label the runner-only estimate as total cost.

## Remaining work before controlled measurements

Freeze a qualified immutable image; verify clock/power management and exact
restoration on the actual host; build trial conditioning and fixed-clock replay;
extend profile provenance/validation for this GPU; collect fresh service/power
calibration with a held-out validation split; and resolve the live clock-boundary
policy before MPC. Do not use the archived L4 thermal fit for live safety.

Serving protocol references: [vLLM 0.28.0 installation](https://docs.vllm.ai/en/v0.28.0/getting_started/installation/gpu/),
[engine arguments](https://docs.vllm.ai/en/v0.28.0/configuration/engine_args/), and
[completion request fields](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/entrypoints/openai/completion/protocol.py).
The [vLLM sampler implementation](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/sample/ops/topk_topp_sampler.py)
documents the environment override. An [upstream FlashInfer issue](https://github.com/flashinfer-ai/flashinfer/issues/3675)
describes this error on SM120 with a local toolkit below CUDA 12.9; PyTorch's
CUDA runtime version alone does not establish the local JIT toolkit version.
