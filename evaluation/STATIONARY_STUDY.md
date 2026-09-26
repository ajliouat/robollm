# Stationary standoff reaching and holding

The earlier [physical-system repair](CONTROL_REPAIR.md) improved live-object proximity to 35/100, but some successful trajectories moved the target. This study asks a different question: can the end effector approach a fixed point above an object and remain there without disturbing the scene? It does not re-score the old trajectories or claim that the two task scores are comparable.

## A physically meaningful, frozen goal

For the requested named object, freeze a goal at its initial body X/Y and 100 mm above the highest point of its compiled initial geometry. The goal never follows the object. Objects retain gravity and normal contacts. No settling phase, orientation target, grasp or placement is introduced.

The existing EE site is a reference point, not a collision shape. The fully open hand has a maximum finger-corner distance of about 71.56 mm from that point. The 100 mm standoff leaves about 3.44 mm of conservative hand clearance after accounting for the 20 mm positional tolerance and 5 mm object-motion bound. This local hand-envelope argument does not guarantee clearance for the rest of the arm; those contacts must be checked throughout execution. The task is a hover/standoff primitive, not physical contact with the object.

Success requires strict distance below 20 mm for 500 elapsed physics intervals: 501 qualifying samples spanning one second. Any exit at or beyond 20 mm resets the dwell. Both controllers start from exactly the same repaired robot/object/model state and command open fingers. The horizon is 5,000 physics intervals (10 simulated seconds), including holding. A scene initially inside the radius is recorded and still needs the full subsequent second. All scenes remain in the denominator.

## Safety is part of scoring

A separate diagnostic MjData copies the live state after each 2 ms physics step and refreshes derived geometry there. It never updates the live integration state to obtain a measurement. [MuJoCo contact generation](https://mujoco.readthedocs.io/en/stable/computation/index.html#collision-detection) uses collision masks and body-pair filters; those settings are unchanged.

Any generated contact with signed distance at or below zero involving a moving robot body fails the episode, including self-contact and moving-link/base contact. Normal object/table, object/object and static mounting contacts are allowed, subject to the object-motion bound. Adjacent-body and other excluded pairs are not independently certified.

For every object, bound the displacement of any material point by its body translation plus `2 × enclosing radius × sin(shortest quaternion angle / 2)`. This conservative bound includes rotation, with equivalent quaternion signs handled identically. A value greater than 5 mm at any sample fails permanently. Safety is checked before success, including at the end of the holding interval. These are checks at discrete 500 Hz samples, not continuous-time or hardware safety guarantees.

## Paired controllers

- **Direct DLS:** the original gain-8 clipped Cartesian command follows this fixed standoff goal, using the existing damped Jacobian mapping to seven position-servo targets.
- **Collision-aware joint plan:** a bounded privileged-state planner searches up to 64 IK starts, each with 250 iterations, and screens endpoints and straight joint-space paths on a private model with 3 mm robot geom margins. It chooses the shortest accepted joint displacement and executes a quintic position profile; it also screens the current-state-to-command connector. The planner uses seed 1729 in every scene. All search/profile constants are recorded in the protocol and source.

Planning checks sample joint paths at at most 0.02 rad per joint. They do not replace the independent 500 Hz execution monitor. If the finite search finds no acceptable path, or the runtime connector guard fails, the episode counts as a failure. No-plan means this bounded planner did not find a path, not proof that the goal is unreachable. Planning time does not advance the simulator; CPU durations are recorded separately. This is not an equal-compute or real-time comparison. Both controllers use privileged state; no VLM, learned policy or perception is evaluated.

## Development, freeze and evidence

Development is restricted to seeds 17, 18 and 101–110. The JSON [protocol](stationary_protocol.json) reserves 260928000–260928099, with requested object index 1 or 2 alternating by original split-list index. Previous evaluation ranges are observed and excluded. Full initial integration/model/goal snapshots must match within each pair.

Finalize and commit source, all numerical settings and the protocol before either controller sees a reserved scene. Require a clean recorded revision before and after evaluation. Unexpected numerical values, simulator warnings, time reset, parity or runtime errors retain an aborted partial report without aggregates; no scene is silently omitted. Ordinary safety, horizon and no-plan failures remain in the complete denominator.

Report integer successes, marginal Wilson 95% intervals, paired outcomes and descriptive difference. The marginal intervals are not a paired-effect interval. Preserve per-physics-step EE/object/contact measurements, per-control-step joint commands, exact scenes, planner work counters and explicit runtime. Early stopping means episode observation durations differ. Exact replay on one runtime checks repeatability, not another independent sample or cross-platform identity.

```bash
python -m pip install numpy==2.4.2 mujoco==3.14.0 gymnasium==1.3.0 pytest==9.1.1
python -m pytest tests/test_stationary_contract.py tests/test_stationary_policy.py tests/test_stationary_study.py -q
python -m evaluation.stationary_reach --seeds 17 18 --output runs/stationary-smoke.json.gz
# Only after the source/protocol freeze, from a clean checkout.
python -m evaluation.stationary_reach --split held-out --output runs/stationary-evaluation.json.gz
```

No held-out result was inspected when this method was prepared. Results will be retained separately after the freeze. The focused CI runs development scenes only; training, rendering, main-branch release and physical hardware remain separate scopes.
