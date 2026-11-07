# RTX PRO 4000 environment qualification — 2026-11-06

The runner reported successful completion for
`qualification-20261106T161054Z-fe7908f8` at
`2026-11-06T16:21:15.506708+00:00`: four requests expected, four completed,
and no invalidity reasons. This is environment qualification only;
`benchmark_measurements_accepted` remains false.

The result was recovered from the completed SSH process output on 2026-11-07.
The final raw artifact bundle has **not yet been copied back or independently
audited**. Direct SSH refused connections and RunPod's proxy returned
`container not found`. Successful runner status is not a substitute for that
pending artifact audit.

## Scope and runtime

- GPU: NVIDIA RTX PRO 4000 Blackwell; host driver 595.91.07.
- Serving environment: Python 3.12.14, vLLM 0.28.0, PyTorch 2.13.0+cu130,
  CUDA runtime 13.0. The template's local CUDA compiler was 12.8.93.
- Model/tokenizer: Qwen/Qwen2.5-7B-Instruct at
  `acbd96531cda22292a3ceaa67e984955d3965282`.
- Configuration: `configs/hardware/blackwell-qualification-torch-sampler.json`.
  Four original Azure development rows, FP16, unchanged default GPU settings,
  512-token chunked prefill, and the supported
  `VLLM_USE_FLASHINFER_SAMPLER=0` override.
- Container digest unavailable; runtime inventory and source/lockfile hashes
  were retained by the runner. No immutable-image benchmark claim is made.
- Focused software validation: 23 checks passed before the final attempt.

The completed runner took 620.673309 seconds, including startup and cleanup.
Its compute estimate was $0.099997 at the screenshot's $0.58/hour rate. This
excludes other setup attempts, development/idle Pod time, storage and transfers;
total session cost has not been reconciled with provider billing.

## Retained attempts

| Attempt | Recorded outcome |
|---|---|
| `serving-setup-20261106T152914Z-e533b6cc` | Copy-mode installation timed out after 600 seconds |
| `serving-setup-20261106T154035Z-4fe041ce` | Hardlink installation completed in 26 seconds |
| `qualification-20261106T154143Z-c6e053ae` | Launcher resolved the venv interpreter symlink incorrectly; stopped before requests; fixed with a regression check |
| `qualification-20261106T154244Z-495a8479` | Storage quota exhausted during initialization; telemetry watchdog stopped the server; zero requests completed |
| `qualification-20261106T155830Z-5f91a8be` | Weights loaded, but automatic FlashInfer sampler failed during warm-up; zero requests completed |
| `qualification-20261106T161054Z-fe7908f8` | Runner reported complete, 4/4 requests, no invalidity reasons; independent artifact audit pending |

The first five bundles were copied to
`runs/remote_artifacts/gq6l3t6ss5eanh/` and their retained file hashes verified.
The three failed qualification bundles passed the independent record audit
while retaining their failed status. The incomplete dependency environment
from the first install was removed to reclaim 13 GB; acquisition records,
setup logs, the working runtime and model cache were preserved. The storage
recovery observation is in ignored `runs/storage-recovery-20261106T155827Z-34330006.json`.

## Pending retrieval

Provider fields observed from `/etc/environment`:
Pod `gq6l3t6ss5eanh`, data center `EU-RO-1`, network volume `eqs8d57xtv`.
The mount source was
`mfs#euro-3.runpod.net:9421[/networkvolumes/eqs8d57xtv]`.

The final bundle's recorded retained location is:

```text
/workspace/GPU_optimization_project/runs/qualification-20261106T161054Z-fe7908f8/
```

After restoring access to this volume, copy the complete final bundle and run:

```bash
.venv/bin/python scripts/audit_hardware_pilot.py \
  runs/qualification-20261106T161054Z-fe7908f8
```

Verify file inventory and hashes on the independent copy as well. Until then,
request latency, energy diagnostics and final GPU cleanup readbacks have not
been independently checked. Fixed-clock acquisition, fresh calibration,
clock-control qualification and live MPC remain future work. The simulation
reference and its documented rounding difference are unchanged.
