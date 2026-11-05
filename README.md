# GPU Inference Frequency Optimization

Standalone reproduction of the NVIDIA L4 inference-serving frequency-control
experiment. This repository contains its own simulator, controllers, request
traces, measured calibration bundle, tests, and recorded reference results.
No sibling repository or GPU is required for the simulation.

## Setup and validation

Use Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked
uv run pytest -q
```

The environment installation may download Python packages. The workload and
calibration data are already included; simulation does not download a model,
contact a cloud provider, or access a GPU.

## Rebuild the simulation artifacts

For a reduced software smoke run:

```bash
MPLBACKEND=Agg uv run python scripts/build_inference_artifacts.py --quick
```

For the full original experiment suite:

```bash
MPLBACKEND=Agg uv run python scripts/build_inference_artifacts.py
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
uv run --locked python scripts/compare_inference_results.py
```

The audit writes CSV and JSON evidence under `reports/simulation_reproduction/`
and exits with status 1 if differences remain. The first full run reproduces all
baseline metrics and MPC latency, with a 0.0181% MPC energy difference traced to
clock-boundary rounding. See the [full report](reports/simulation_reproduction/README.md)
for the comparison and diagnostic commands. Live GPU measurements are next.

## Repository contents

- [PLAN.MD](PLAN.MD): reproduction and hardware evaluation milestones.
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

The next hardware target is the existing Google Cloud VM with an NVIDIA L4.
Start with fixed-clock replay of the included workload, then collect fresh
calibration before live MPC. The live benchmark runner and VM access setup
remain planned work; the simulation bundle does not provision or modify a VM.

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
