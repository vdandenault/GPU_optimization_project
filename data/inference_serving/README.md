# Inference-serving data

`azure_code_animation.csv` and `azure_code_evaluation.csv` are unmodified row
subsets of the Azure 2023 LLM inference code trace. The first file contains the
first 20 requests. The second contains requests arriving during the first five
minutes, relative to the first timestamp. Timestamps and token counts are kept
exactly as published.

Source: [Azure Public Dataset, Azure LLM Inference Dataset
2023](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md)

License: CC BY 4.0. The trace was retrieved on 2026-09-01. The SHA-256 digest of
the complete downloaded code trace was
`54e9a6d2a4bd06ba1e060304b900abbc74cbea53de96506e60fe5bb4f2277fb6`.

The dataset's requested attribution is Pratyush Patel, Esha Choukse, Chaojie
Zhang, Aashaka Shah, Inigo Goiri, Saeed Maleki, and Ricardo Bianchini,
["Splitwise: Efficient generative LLM inference using phase
splitting"](https://www.microsoft.com/en-us/research/publication/splitwise-efficient-generative-llm-inference-using-phase-splitting/),
ISCA 2024.

`l4_profile.csv` is the deterministic aggregate of a completed NVIDIA L4
measurement run. The experiment used Qwen2.5-7B-Instruct at the revision and
vLLM image digest recorded in `profile_manifest.json`. It requested five GPU
clocks and retained all five in `l4_profile_all_requested.csv`; the modeled
profile uses the maximum-cardinality subset for which realized clock, prefill
rate, and decode rate are strictly increasing. The raw request observations,
100 ms GPU telemetry, requested-versus-realized clocks, conditions,
repetitions, checksums, and provenance are retained alongside the aggregate.

The measurements also expose limitations that matter for interpretation. The
upper requested clocks collapsed to a narrow realized-clock plateau under the
power cap, and recorded telemetry occasionally exceeded the requested 64.8 W
limit. Three very short batches required a one-sample telemetry fallback. The
one-state thermal fit has only `R^2 = 0.00201`, so it is a deliberately reduced
control model rather than evidence of a well-identified thermal plant. Consult
the manifest and raw telemetry before making hardware-performance claims.

No textbook build downloads the trace, contacts GCP, or rewrites these files.

## Local bundle and usage

All files needed by the copied serving simulator are present in this directory.
The measured profile loader validates the manifest, checksums, schemas, clock
selection, and raw measurement counts before accepting the calibration.
`vllm.log` is part of that checksummed bundle and must be retained.

| File | Contents |
|---|---|
| `azure_code_animation.csv` | 20 recorded requests |
| `azure_code_evaluation.csv` | 781 recorded requests from the first five minutes |
| `l4_profile.csv` | Four modeled requested-clock settings |
| `l4_profile_all_requested.csv` | All five measured requested-clock settings |
| `l4_profile_raw.csv` | 1,950 raw request observations |
| `l4_telemetry.csv` | 64,737 GPU telemetry samples |
| `profile_manifest.json` | Measurement provenance and integrity metadata |
| `vllm.log` | Original serving log, required by the manifest checksums |

The traces contain timestamps and token counts, not original prompt text.
Replay on the L4 VM will construct synthetic token-ID prompts and force output
lengths, following the reference protocol.

The initial reproduction uses these archived measurements. Live control needs
fresh measurements on the available L4 VM. Keep new acquisitions in separate
run directories and leave the reference bundle unchanged.

Full historical source notes are archived in
[original-data-notes.txt](../../docs/reference/original-data-notes.txt).
Hardware runners and separate thermal-identification datasets are not required
for the initial simulation bundle. See the [root README](../../README.md) for
local setup and experiment commands.
