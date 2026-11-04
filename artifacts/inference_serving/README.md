# Recorded reference results

These are imported results from the source experiment, retained for comparison:

- `textbook_results.json`: recorded trajectories and experiment metadata.
- `metrics_mpc.csv`: maximum-clock, reactive, offline, and MPC comparison.
- `metrics_open_loop.csv`: original open-loop comparison.
- `metrics_dp.csv` and `metrics_fqi.csv`: related scheduling experiments.

The results are simulations calibrated with measured NVIDIA L4 profiles.
Their existence does not establish a successful reproduction on this machine
or a live MPC result on the available VM.

The local builder writes new results to `runs/inference_serving/`, preserving
these reference files. See the [root README](../../README.md) for commands and
the [import manifest](../../docs/reference/import_manifest.json) for checksums.
