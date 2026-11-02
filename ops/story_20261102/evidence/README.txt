Frozen-source view-response diagnostics, 2026-11-02

The pre-measurement protocol is ../source_protocol.json. Worker outputs
record acquisition commit 353596eea013b4f62d7c55eb9d18d69fae13f05f.
Tasks 0,2,6,8; initial state 21; steps 10,50; eight frozen Spatial states.
Seven physical cloth textures plus main-only and wrist-only image swaps.
No texture retraining and no target-model gradients. Workers used GPUs
6 and 0 on 4090-48G; GPU 4 was excluded. Both completed successfully.

worker0_results.json / worker1_results.json retain all normalized action
vectors, state metadata, texture hashes, and attention summaries.
calibration.json independently confirms exact hook-on/off identity for
all eight Plain actions. The original SDPA kernel supplies model outputs;
attention is reconstructed from its actual rotated inputs and exact mask.
Maximum relative attention-output reconstruction RMS: 0.0017327278619632125.

summary.json, attention_all_conditions.csv, and view_interventions.csv
are the complete derived summaries. Raw per-state PNGs and attention
arrays remain in experiments/refcloth_story/20261102/worker0 and worker1
on the experiment server, and in the delivered local paper data archive.

Source cloth attention density medians, main/wrist: RefCloth 1.271/0.705,
Plain 0.678/0.376. RefCloth exceeds the three-random-texture mean in 8/8
main states and 7/8 wrist states. Native reference also has elevated
attention; its main-view median exceeds RefCloth. Attention is not a
semantic gripper-recognition test.

The wrist symmetric projection share of the full normalized pose-action
change averages 85.0% at step 10 and 40.2% at step 50. This is a signed
attribution in a frozen source-input experiment, not target closed-loop
causality or a claim that the wrist always dominates. Pose-action RMS
uses the first five actions and six pose coordinates, excluding gripper.

../source_attention.py includes acquisition and --calibration-root mode.
Run in the existing tex3d-oft-mj330 environment, with
ops/support_20261101 in PYTHONPATH and the pinned source helpers.
The script takes --repo --assets --out --tasks; --out must be fresh.
The full manuscript was not added to this public code repository.
