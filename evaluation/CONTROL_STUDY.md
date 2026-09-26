# Grounded reaching: an interface and control study

This study asks whether the end effector reaches the **requested object** in a
three-object scene. It separates an object-identity bug from the limitations of
the existing motor controller. It does not evaluate a VLM, learned policy,
grasping, stable placement or transfer to a physical robot.

## The failure being investigated

The executor previously grounded a target and then replaced its position with
`env.object_pos(0)` when assembling policy input. Pick/place success checks also
read object zero. A correct grounding result therefore did not imply that the
controller acted on, or measured, the selected object.

The repaired executor retains a grounded name and resolves its current position
at each step. Pick/place additionally retain the selected source across the
sequence. Their lift, closure and release checks are still simulation heuristics,
not proof of a secure physical grasp. A failed primitive prevents later steps
from proceeding, and the original environment horizon is restored even when
execution raises an exception.

This repair operates on named scene objects. Abstract destinations such as
`goal`, `stack_position` or `red_bin` are not generally grounded by this
environment. It does not make arbitrary planner-generated pick/place sequences
executable.

## Frozen comparison

The machine-readable specification is [control_protocol.json](control_protocol.json).
Before running its evaluation seeds, the code and protocol are committed.
Development checks use seeds 17 and 18. Evaluation uses all 100 seeds from
260926000 through 260926099, with no parameter selection on those scenes.
These are held out from this repair's development, not an independent benchmark
or a randomly sampled set of real-world tasks.

Each scene has three objects. The requested target alternates between indices
one and two, so object zero is always a distractor. Four policies receive the
same initial scene for each seed:

1. **Grounded scripted:** the existing proportional reaching controller, gain 8,
   follows the named object's current position.
2. **First-object action ablation:** the same controller follows object zero;
   success is still scored against the requested target. This isolates incorrect
   action targeting, not every behavior of the old executor.
3. **Random XYZ, open gripper:** seeded random translation actions, with the
   gripper held open to avoid confounding the comparison with finger commands.
4. **Zero XYZ, open gripper:** zero translation commands. This checks for passive
   success caused by scene or arm dynamics; it does not freeze the simulator.

All four use the same executor, success predicate and stopping rule. The limit
is 200 control steps at 20 Hz (10 simulated seconds). The gain, scene generator,
robot model and threshold are unchanged. No renderer, GPU, Torch or model
download is required.

## What success means

Success means that the end-effector site becomes strictly closer than 3 cm to
the requested object's **current body center** on a post-action step. The
predicate must not already hold at reset. Already-satisfied episodes are retained
in the denominator, separately flagged, and receive no credited reaching success.
The executor stops at the first qualifying step or the horizon/episode boundary.

This is instantaneous proximity. It requires no particular orientation, grasp,
contact, dwell time or stationary object. There is no settling phase before
measurement. Objects can move or fall, so traces retain their positions and
displacement. Reaching a displaced object is not equivalent to reaching an
unchanged tabletop goal. These limitations apply even to a nonzero success rate.

## Evidence and interpretation

The artifact retains every seed, scene identity, target identity, success flag,
step count, initial/final/minimum distance, and per-step actions and geometry.
It includes the protocol, code revision, dirty status and runtime versions.
Paired scene hashes must match across policies. The original historical benchmark
files are never overwritten.

Report integer successes out of all 100 scenes and Wilson 95% endpoints for each
arm. Paired counts distinguish scenes solved by both compared policies, only one,
or neither. The difference between their success rates is descriptive; marginal
Wilson intervals are not an interval or significance test for that paired
difference. A deterministic replay checks artifact integrity, not independent
replication or additional statistical evidence.

The new `Grounded control checks` workflow runs focused headless regression tests
and a development-seed smoke on GitHub. It is separate from the existing full
CI suite. Passing this job does not establish that all training, rendering or
optional-model tests pass, or that a branch has been merged into `main`.

## Reproduce

Use the code revision recorded in the result, then install the pinned CPU runtime:

```bash
python -m pip install numpy==2.4.2 mujoco==3.14.0 gymnasium==1.3.0 pytest==9.1.1
python -m pytest tests/test_v104.py tests/test_v106.py tests/test_v107.py \
  tests/test_benchmark_reproducibility.py tests/test_pipeline_grounding.py \
  tests/test_grounded_control.py
python -m evaluation.grounded_control --seeds 17 18 --output runs/control-smoke.json
```

## Measured result and exact replay

The [retained report](results/grounded-control-2026-09-26/README.md) contains
400 episodes and 80,000 steps. Every arm produced 0/100 successes, with Wilson
95% interval [0%, 3.6995%]. The grounded controller ended 0.331325 m from the
target on average. Correcting target identity did not solve motor control.

The frozen source revision is `f102fe10dc30bf2aa18cc16ed0dc43594051abbd`.
To reproduce the evaluation from an existing clone, fetch the review branch,
switch to that revision with a clean working tree, install the pinned runtime
above, then run:

```bash
git fetch origin codex/grounded-control-evaluation
git switch --detach f102fe10dc30bf2aa18cc16ed0dc43594051abbd
python -m evaluation.grounded_control --split held-out --output runs/control-first.json.gz
python -m evaluation.grounded_control --split held-out --output runs/control-replay.json.gz
python - <<'PY'
import gzip, json
from evaluation.grounded_control import semantic_payload, semantic_digest
with gzip.open('runs/control-first.json.gz', 'rt') as file:
    first = json.load(file)
with gzip.open('runs/control-replay.json.gz', 'rt') as file:
    replay = json.load(file)
assert first['provenance'] == replay['provenance']
assert semantic_payload(first) == semantic_payload(replay)
print(semantic_digest(first))
PY
```

Use fresh filenames: existing artifacts are protected. Keep outputs in ignored
`runs/` until both executions complete, so the second run retains a clean source
checkout. On the recorded runtime, both executions produced semantic SHA-256
`05a5ab273f5a318ba95fec85725b44be4946fcb3bbd55c1fb623d93065e5f2c7`.
Numerical replay is checked on that runtime; bitwise equivalence across different
platforms or simulator versions is not assumed. Repeated execution is not an
additional independent sample.
