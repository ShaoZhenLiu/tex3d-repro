# Frozen-texture mechanism measurements

The protocol fixes a single deployed RefCloth and runs no optimization.

* `source_direction.py` measures native versus differentiable feature displacement at 32 freshly archived Spatial states, with exact native and synthesis A/A checks. The deployed PNG is primary; the original floating theta is a sensitivity check.
* `evaluate.py` runs four paired native camera-input interventions on all 100 pi05-Object episode keys. Actual sensor-exposure MjData is copied; selected camera appearances are rerendered without changing live physics or query noise. Plain and Both must match direct native deployment for a complete prescribed trajectory.
* `control.py` admits workers only after host-memory and GPU checks. GPU 0--3 only; at most two workers/card. An unexplained worker error stops new admissions for inspection while preserving live workers and existing observations.
* `summarize.py` accepts exactly 400 outcomes, verifies consumed camera hashes and paired initial exposures, reports all six paired contrasts, and archives raw features, simulator snapshots, images, episode traces and execution records with a SHA-256 receipt.

Run through `run.sh <exact-commit>` from the synchronized repository on the authorized 41 host. The existing fixed source and target environments and renderer assets are read-only dependencies.

`analysis_archived.py` and `load_feature_cache.py` independently recompute directions from the complete original Fig.1d float32 feature archive using NumPy. Specify `--source <t1_fix_directory> --out <new_output_directory> --protocol <protocol.json>`. This archive is kept separate from the new canonical-state measurement; the two trajectories must not be described as bitwise identical.
