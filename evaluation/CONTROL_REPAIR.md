# Reaching-system repair: frozen protocol

This follow-up investigates the 0/100 result in [the original study](CONTROL_STUDY.md). It compares the correctly grounded original system with a physical-model/reset/gravity-support bundle. It is not an isolated controller comparison, learned policy, grasping study or hardware validation.

## Development diagnosis

On development seeds 17 and 18, the original reset embeds link1 in the table/base and folds link2 through link4. The link1/table overlap is 10 mm; link2/link4 overlap is approximately 22.6 mm. The initial positional Jacobian has full row rank, so this is not an initial singularity. With the original robot weight uncompensated, an incremental position command also drifts under gravity.

The repair trims only the collision capsules (link1 lower endpoint 0 → 0.03 m; link2 upper endpoint 0.30 → 0.28 m), preserving the original robot masses, inertia and kinematic link lengths exactly. A shared collision-free home and matching actuator targets replace the folded reset. Ideal MuJoCo body gravity compensation is enabled only on robot bodies. Objects retain normal gravity and all collision handling remains active.

[MuJoCo describes body gravity compensation](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body) as an upward force proportional to body weight. This is a simulation assumption, not a validated physical servo. The gain-8 reaching policy, DLS inverse kinematics, action scaling, 20 Hz control rate, horizon and strict 3 cm criterion are unchanged.

Twenty-six focused physical checks cover initial contacts, exact original inertial properties, reset/keyframe agreement, ten-second zero-command hold, falling objects, retained self-collision detection, and nearby forward-kinematics-derived targets. The PlaceEnv subclass also aligns its closed-finger servo targets with its reset pose. Passing those checks does not establish whole-workspace reachability.

`evaluation.motor_diagnostics` crosses the two collider edits, new home/actuator reset and gravity support on seeds 17 and 18. All eight combinations keep contacts and object gravity active. These diagnostic trajectories continue for the full 200 steps, even after proximity, and always target object index 1: their minima are not the evaluation scores below. Raw joint states, forces, contacts and Jacobian singular values are retained. Two development scenes do not establish the isolated effect of each repair across the workspace.

## Reserved comparison

The machine-readable [protocol](control_repair_protocol.json) fixes 100 fresh seeds, 260927000–260927099. The earlier 260926000–260926099 seeds are now observed evidence and are excluded. Development uses 17, 18 and 101–110. Commit the implementation and this protocol before running either system on reserved scenes; real held-out execution requires a clean Git revision.

Each fresh scene runs both `frozen_v1` and `repaired_system`. Ordered object identities, initial poses/velocities, physical properties, table/floor and simulation settings must match. Arm configuration and geometry differ by design; full snapshots and model hashes retain those differences. The home change alters initial target distances, so the result measures this complete system bundle.

The target alternates object indices 1 and 2, with exact named-object grounding. Success means the EE site first enters a strict 3 cm sphere around the **current** target body center after an action. Stop at that hit or 200 control steps (10 simulated seconds). There is no settling step, grasp, orientation or dwell requirement. Initially satisfied scenes earn no success credit. All 100 scenes remain in each denominator, with no filtering or resampling. Target movement is retained: hitting or moving an object can affect this proximity metric.

Report both integer success counts, marginal Wilson 95% intervals and paired outcomes. Marginal intervals are not a confidence interval for the paired difference. Runtime failures retain explicitly aborted partial evidence, not an aggregate score.

The original environment, spawner and XML are vendored byte-for-byte from `f102fe10dc30bf2aa18cc16ed0dc43594051abbd` under `frozen_v1/`, verified against file hashes and loaded without changing the live environment imports. The original v1 runner now defaults to that snapshot, preserving its physics after the default environment is repaired.

## Reproduce

Use Python 3.11 and the recorded versions:

```bash
python -m pip install numpy==2.4.2 mujoco==3.14.0 gymnasium==1.3.0 pytest==9.1.1
python -m pytest tests/test_arm_control.py tests/test_control_repair.py -q
python -m evaluation.motor_diagnostics --output runs/development-diagnostics.json.gz
python -m evaluation.control_repair --split development --output runs/repair-development.json.gz
# Only from the frozen clean revision; use a different output path for replay.
python -m evaluation.control_repair --split held-out --output runs/repair-evaluation.json.gz
```

No renderer, trained checkpoint or GPU is required. Exact replay on one recorded runtime checks repeatability; it does not add independent scenes or promise cross-platform bitwise identity. The focused GitHub job runs development scenes only. The wider training/rendering suite and main-branch release remain separate.

## Results

No reserved-scene outcome had been inspected when this source/protocol document was frozen. Results and replay evidence will be retained separately after evaluation; do not infer performance from development smoke tests.

After the source freeze, the [retained result and replay](results/control-repair-2026-09-26/README.md) measured 35/100 for the repaired bundle versus 0/100 for the frozen original. The original no-outcome statement above documents the pre-evaluation freeze.
