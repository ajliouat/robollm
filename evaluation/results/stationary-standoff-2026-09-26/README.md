# Stationary standoff study — 26 September 2026

Source and protocol were frozen at **75c6573f543d567699a723769464bb232bd84ec4** before either controller ran on seeds 260928000–260928099. Both executions used that clean revision; nothing was tuned after their outcomes. This is a new fixed-goal task, not a before/after comparison to the earlier 35/100 moving-center result.

## Contract and comparison

A fixed goal lies 100 mm above the requested object's initial top surface. Success requires the EE site strictly within 20 mm for one second (501 samples spanning 500 physics intervals), with no forbidden generated moving-robot contact and no all-object surface-displacement bound above 5 mm at any recorded sample. The bound includes translation and rotation. The horizon is ten simulated seconds, with 20 Hz joint commands and 500 Hz independent checks. See the [method](../../STATIONARY_STUDY.md) and [machine-readable protocol](../../stationary_protocol.json).

Both controllers have identical complete initial simulator/model/object states and goals. Direct DLS uses the old gain-8 Cartesian law. The candidate uses a finite multi-start IK search, discrete collision screening and a bounded quintic joint-position reference, with planner seed 1729. No-plan failures stay in the denominator.

| Controller | Successes | Marginal Wilson 95% interval | Other outcomes |
|---|---:|---:|---|
| Direct DLS | 29/100 | 21.01–38.54% | 71 forbidden-contact failures |
| Collision-aware joint plan | 88/100 | 80.19–93.00% | 12 bounded searches without an accepted path |

Paired outcomes: 29 both succeed, 59 candidate-only, 12 both fail, zero direct-only. The descriptive difference is +59 percentage points. Marginal intervals are not a paired-effect interval.

All 200 episodes and 252,950 physics intervals are retained. None started within the goal tolerance. The largest object-surface displacement bound across all observed samples is 0.117437 mm (below 0.118 mm), against the 5 mm limit. The candidate's 12 search failures do not prove that those goals are unreachable. Every actual candidate execution completed the sampled holding contract in this set; this does not establish universal collision freedom or hardware safety.

## Limits and computational context

This is hover/standoff reaching, without an orientation target, physical grasp or placement. Both controls use privileged simulator state; neither measures visual grounding, a VLM or a learned policy. Ideal robot-only gravity compensation is unchanged. The simulator's collision filters omit some body pairs and all checks are discrete.

A failed search performs no physics steps after reset. Contact failures and successful holds stop early, so observation durations differ. Successful direct episodes take a median 1.358 simulated seconds; successful planned episodes take 5.366 seconds. These are different success subsets, not a controlled speed comparison. Candidate initial planning took a median 0.118 seconds and maximum 0.213 seconds on this local run; these are incidental CPU observations, not a performance benchmark or real-time guarantee. Planning computation does not advance simulation time and the compared controllers do not use equal compute.

## Complete evidence, divided into modest files

- `metadata.json.gz` retains every original report field except the episode array, including all initial scenes, full model snapshots, protocol, implementation hashes, provenance and CPU timing records.
- Four `episodes-*.json.gz` files retain ordered consecutive groups of 50 complete episodes, including every physics sample, control command, result and planner counter. No trace field is dropped.
- `manifest.json` records file sizes/hashes, ordering and the full report's semantic digest.
- `episodes.csv` and `summary.json` make the outcomes inspectable without loading all traces.
- `replay.json` records exact equality and source/runtime identity, and both original compressed-file hashes.

Run `python reconstruct.py` in this directory to verify every shard and the reconstructed complete report. Optionally use `--output /path/to/new-report.json.gz` to produce one file. The two original complete run files remain local under ignored `runs/`; the public shards reconstruct all fields of the first one exactly.

The full second execution reproduced every semantic field exactly. Digest: `58bea917391adc80ef6513debd1f7d7b1112cff1f2c8a13cf2cf89134f31ae43`. Excluded fields are top-level timestamps, elapsed/CPU timings and provenance; provenance was separately identical. Planner work counters, geometry, decisions and outcomes are included. This checks repeatability on Python 3.11.14, NumPy 2.4.2, MuJoCo 3.14.0 and Gymnasium 1.3.0 on macOS ARM64, not another 100 independent samples or cross-platform identity.

Independent trace validation recomputed all 253,150 rows (including the 200 initial samples), the fixed goal geometry, all-object pose bounds, contact classification, safety-first scoring, paired initial snapshots and the full denominators. Every successful hold contains 501 qualifying samples spanning exactly one second. The closest candidate entry to the strict 20 mm boundary is only about 3.17 micrometres inside; sampled success is not a robustness margin or continuous-time guarantee.

## Reproduce the execution

```bash
git switch --detach 75c6573f543d567699a723769464bb232bd84ec4
python -m pip install numpy==2.4.2 mujoco==3.14.0 gymnasium==1.3.0 pytest==9.1.1
python -m evaluation.stationary_reach --split held-out --output runs/reproduced-stationary.json.gz
```

Use a clean checkout and a fresh output path. The reserved scenes are now observed; replaying them is reproduction, not a new held-out evaluation. No GPU, renderer or checkpoint is needed. [Focused source CI](https://github.com/ajliouat/robollm/actions/runs/36243007801) passed 293 tests and three development simulation smokes. The branch is not merged and the broader training/rendering suite is not certified.

Implementation, regression checks and editorial preparation used Codex assistance. Inspect the source, full trajectories and declared limits to assess the demonstrated scope.
