# Named-target proximity study — 26 September 2026

**Result: 0/100 successes for each of four controls.** The target-identity repair
passes its regressions, but the current controller does not solve this reaching
protocol. This is a negative motor-control result, not a learned robotics or VLM
benchmark.

| Control | Successes | Wilson 95% interval | Mean final target distance |
|---|---:|---:|---:|
| Grounded scripted | 0/100 | 0–3.6995% | 0.331325 m |
| First-object action ablation | 0/100 | 0–3.6995% | 0.352365 m |
| Random XYZ, open gripper | 0/100 | 0–3.6995% | 0.711441 m |
| Zero XYZ, open gripper | 0/100 | 0–3.6995% | 0.678147 m |

All 400 episodes reach the 200-step limit; none starts satisfied. All 100
grounded/ablation pairs fail. The smaller final distance is descriptive and does
not establish improved task success. The historical one-object MoveTo benchmark
uses a different task and is not a comparable before/after baseline.

## Retained evidence

- [episodes.json.gz](episodes.json.gz): complete report, protocol, initial scenes
  and 80,000 post-step records with actions and live geometry (7.62 MB compressed).
- [episodes.csv](episodes.csv): readable one-row-per-episode outcomes and geometry
  summaries. No episodes are removed.
- [summary.json](summary.json): provenance, aggregate results, artifact hashes,
  selected seeds and post-hoc descriptive diagnostics.
- [replay.json](replay.json): exact replay comparison, both run metadata records
  and compressed/semantic hashes. The duplicate second trace remains local in
  ignored `runs/`; the first complete trace is retained here. The replay is not
  an independent sample.

Source and protocol were frozen at
[`f102fe10dc30bf2aa18cc16ed0dc43594051abbd`](https://github.com/ajliouat/robollm/tree/f102fe10dc30bf2aa18cc16ed0dc43594051abbd)
before evaluation. Both executions recorded that clean revision and the same
runtime: Python 3.11.14, NumPy 2.4.2, MuJoCo 3.14.0, Gymnasium 1.3.0, macOS ARM64;
Torch absent. Each took approximately 49 seconds on the local host. These timings
are informational, not a performance benchmark or a runtime guarantee.

The exact replay matched all report fields except timestamps, elapsed time and
provenance; provenance was compared separately and also matched. Semantic SHA-256:

```text
05a5ab273f5a318ba95fec85725b44be4946fcb3bbd55c1fb623d93065e5f2c7
```

[Focused GitHub validation](https://github.com/ajliouat/robollm/actions/runs/36229451051)
passed 180 tests plus a development-seed simulation smoke at the source revision.
It is not the full training/rendering CI suite, and this evidence is delivered
on `codex/grounded-control-evaluation`, not merged into `main`.

## What the traces reveal, and what they do not

Post-hoc geometry checks find every grounded run ends short in X, with mean X
error 0.324548 m. X actions saturate at +1 for 99.395% of grounded steps. The
closest grounded approach anywhere in the trajectories is 0.099408 m, outside
the 0.03 m criterion. Across all arms, target movement is below 0.000117 m; moving
objects do not explain these failures.

These diagnostics follow directly from the named columns in the compressed
report: final `target_x - ee_x`, fraction of `action_x == 1`, minimum
`named_target_distance_m`, and maximum `target_displacement_m`. They were inspected
after the frozen comparison, and were not used to tune it.

The traces lack per-step joint, actuator-force and contact data. They cannot
identify whether initialization, gravity compensation, actuator tracking, joint
limits, collision or Jacobian conditioning causes the shortfall. Zero XYZ input
also does not freeze the arm. A separate development experiment should instrument
those quantities before further training. The evaluation seeds are now observed;
any tuning requires a fresh future holdout.

See [the full method and reproduction commands](../../CONTROL_STUDY.md).
