# Paired reaching-system repair — 26 September 2026

The implementation and [protocol](../../control_repair_protocol.json) were frozen at **9c35fccea85bfc9d7ffc98b0d751c9e5dd7e5c6d** before either system saw the 100 reserved scenes. Both executions used that clean revision. No code or parameter was adjusted after observing these results.

## What changed

Corrected two penetrating collision capsules while preserving the robot inertials and kinematics, selected a collision-free home with aligned servo targets, and enabled ideal robot-only body gravity support. The target-following policy and strict post-action 3 cm success criterion are unchanged. This measures the entire geometry/reset/support bundle, not the isolated effect of one component. See [method and reproduction](../../CONTROL_REPAIR.md).

## All 100 paired scenes

| System | Successes | Wilson 95% interval | Mean initial distance | Mean final distance | Mean steps |
|---|---:|---:|---:|---:|---:|
| Frozen original | 0/100 | 0–3.70% | 52.46 cm | 32.65 cm | 200 |
| Repaired bundle | 35/100 | 26.36–44.75% | 40.67 cm | 13.01 cm | 139.27 |

35 pairs succeed only after the repair, 65 fail in both systems, none succeeds only in the original. The descriptive paired difference is +35 percentage points. Marginal Wilson intervals are not an interval for that paired difference. There are 200 episodes and 33,927 recorded control steps, with no initially satisfied scene, omitted scene or aborted episode.

**This is limited reaching performance.** Success is instantaneous proximity to the live object center, without orientation, dwell, grasp or stable placement. All objects retain gravity and contacts. In the repaired system the target moves more than 3 cm at some point in 42/100 episodes, including 6/35 successful ones; the maximum target displacement is 1.792 m, in a failed episode. Contacts can move or eject objects. No outcome was excluded on this basis. The metric is insufficient for safe or stable manipulation.

The reset change also makes the end effector closer on average initially (40.67 cm versus 52.46 cm). Matching object scenes does not imply matching robot states. Ideal gravity compensation is a simulator assumption; there is no calibrated hardware, learned-policy or VLM result here.

## Artifacts

- `episodes.json.gz`: all 200 full initial snapshots and model parameters, live geometry/actions, joint/actuation/force/contact traces and outcomes.
- `episodes.csv`: one compact outcome row per episode.
- `summary.json`: frozen protocol, runtime, manifests, aggregate and target-motion counts.
- `replay.json`: exact comparison record, provenance and compressed file hashes.
- `development-diagnostics.json.gz`: eight combinations of collision geometry, home/servo reset and gravity support on development seeds 17 and 18. These continue for 200 steps and always use object index 1; they are diagnostics, not evaluation outcomes.

The complete second execution reproduced every semantic field exactly on Python 3.11.14, NumPy 2.4.2, MuJoCo 3.14.0 and Gymnasium 1.3.0 on macOS ARM64. Semantic SHA-256: `c4a1d3ba7e6b06ad9d4791b9ae9d30eb65a0c61e5ad99b10345d34465f5d5581`. Only timestamps, duration and provenance are excluded from that digest; provenance was separately equal. The duplicate replay remains local; the first full trace and comparison record are published. Replay is not another independent sample.

[Focused source CI](https://github.com/ajliouat/robollm/actions/runs/36238199770) passed 230 tests and both development simulation smokes. This is a review-branch delivery, not a merged main release or a pass of the full training/rendering suite.

## Reproduce the recorded run

```bash
git switch --detach 9c35fccea85bfc9d7ffc98b0d751c9e5dd7e5c6d
python -m pip install numpy==2.4.2 mujoco==3.14.0 gymnasium==1.3.0 pytest==9.1.1
python -m evaluation.control_repair --split held-out --output runs/reproduced-repair.json.gz
```

Use a clean checkout and a fresh output filename. The reproduction of already observed scenes is not a new held-out study. A future improvement needs a separately frozen safety/hold metric and fresh seeds.

## Next question

Can collision-aware reaching attain and hold a target without displacing it or striking the table/other objects? First classify the retained failed/contact-heavy trajectories on development data, then define a stricter stationary-target/dwell contract before changing the controller. Do not use these 100 observed scenes for tuning and call the next score held out.

Implementation, diagnostics and editorial preparation used Codex assistance. The archived code and measurements define the demonstrated scope.
