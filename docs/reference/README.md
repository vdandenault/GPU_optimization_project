# Reference provenance

The simulation code, tests, dataset subsets, calibration, and recorded results
were copied from `https://github.com/pierrelux/rlbook`, source revision
`ce6014fe75aee4906793c96de87ed179c989b99e`, on 2026-11-04.
The source project metadata credits Pierre-Luc Bacon. Existing source content
and dataset attribution are retained; this import does not assign new licenses.

[import_manifest.json](import_manifest.json) records every imported file's
original repository path and SHA-256 digest at import time. The files were
copied from a local working tree, so the individual digests identify the exact
bytes independently of the recorded commit.

The local artifact builder changes the output location to
`runs/inference_serving/` and places figures inside that directory to preserve
the imported reference outputs. Other imported Python files are initially
unchanged. The standalone environment has its own dependency lockfile.

## Archived source notes

- [Original chapter](mpc-variants-reliability.txt)
- [Original data notes](original-data-notes.txt)
- [Original reproduction notes](original-reproduction-notes.txt)
- [Historical live development status](live-control-development.txt)

These are verbatim archival text snapshots. Book syntax, historical file paths,
cloud launch commands, and budget authorizations inside them describe the source
project. They are not local entry points or authorizations for this project.
Use the root README and PLAN.MD for local instructions.

The import contains the initial serving simulation's complete measured profile
bundle and workload subsets. Separate thermal-identification acquisitions and
historical cloud launch infrastructure are outside this initial bundle; their
results are discussed in the archived notes. They are not required by the
copied simulation or tests. Fresh live measurements will be stored in this
project when hardware evaluation begins.
