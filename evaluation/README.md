# Evaluation protocol and evidence

For the new named-target, multi-object control study, see
[Grounded reaching](CONTROL_STUDY.md). It exercises the hierarchical executor
with one reaching primitive. The subsequent [physical-model repair study](CONTROL_REPAIR.md)
compares the frozen original system with corrected geometry, initialization and
ideal gravity support on a fresh, separately reserved scene set. The next
[stationary standoff study](STATIONARY_STUDY.md) defines a different task: hold
a fixed goal above the object for one second without forbidden simulator
contacts or more than 5 mm of object-surface motion. Its scores must not be
compared directly with the older moving-center proximity scores. The general benchmark described below continues
to measure each environment's own success flag; its metric was not silently
redefined to match that study.

The benchmark evaluates environment/policy pairs in MuJoCo. The default suite
contains eight configurations: random and scripted policies for PickPlace and
MoveTo, and random policies for ColorPick, Stack, Sort and ComplexLanguage.
It does **not** evaluate a learned VLM, visual grounding or the hierarchical
executor. A run that completes is not evidence of successful manipulation.

## Small CPU reproduction

The evaluator needs Python 3.10+, NumPy, Gymnasium and MuJoCo. Torch is needed
only when evaluating SAC checkpoints; no model download, GPU or renderer is
needed for the default benchmark. In an isolated environment:

```bash
python -m pip install numpy gymnasium mujoco pytest
python -m pytest tests/test_benchmark_reproducibility.py -q
python -m evaluation.benchmark --episodes 3 --seed 42 --output runs/benchmark-smoke
```

Three episodes per configuration are a functional smoke check, not an estimate
of useful performance. For a performance study, choose and report a suitable
sample size and seed range before inspecting results, compare matching reset
seeds, and retain all outcomes. CPU time depends on the host and MuJoCo version.
New runs default to the ignored `runs/benchmark/` directory, so they do not
overwrite the checked-in historical archive.

## What is seeded and measured

- Episode `i` calls `env.reset(seed=seed+i)` and
  `env.action_space.seed(seed+i)`. Both are necessary: resetting the environment
  does not seed its action sampler. Scripted policies are reset each episode;
  optional SAC checkpoints use deterministic actions.
- Custom stochastic policies must seed their own randomness. The evaluator
  does not promise bitwise reproducibility across platforms, MuJoCo versions or
  arbitrary policy implementations.
- An episode runs until the environment returns `terminated` or `truncated`.
  Default project environments limit episodes to 200 steps. Success is the
  final `info["success"]` value; reward is the undiscounted sum of step rewards.
  An absent success flag counts as false. Non-finite rewards and invalid/empty
  evaluation requests fail rather than generating misleading measurements.
- Return/length standard deviations use `ddof=0`. They describe the sampled
  episode distribution; they are not confidence intervals for the mean.
- `initial_success` records whether the reset state already met the environment
  success condition (or null when no flag was provided). It does not alter the
  existing metric, but makes trivial starts visible. In particular, L5 tests
  final geometric predicates, not execution of every verb in an instruction.

## Interpreting Wilson intervals

For success count `k` out of `n > 0`, `p=k/n` and `z=1.96`, compute:

```text
denominator = 1 + z²/n
centre = (p + z²/(2n)) / denominator
spread = z × sqrt((p(1-p) + z²/(4n))/n) / denominator
interval = [max(0, centre-spread), min(1, centre+spread)]
```

The Wilson interval is not symmetric about the observed rate. For 0/100,
report **0%, 95% CI [0%, 3.70%]**, never `0% ± 1.85%`. For 20/100, report
**20%, 95% CI [13.34%, 28.88%]**. The interval describes episode outcome
uncertainty under the sampled protocol; it does not establish generalization
to other instructions, scenes, training seeds or real robots.

The `ci95` field remains a deprecated half-width for existing readers. New
consumers should use `ci95_low` and `ci95_high`, not infer endpoints from `ci95`.
Differences in negative shaped returns should be expressed as absolute return
differences; ratios such as “2.6× better” are not meaningful performance claims.

The existing CLI threshold checks compare point estimates with historical
reference values, including zero thresholds. Their PASS/FAIL labels are not
statistical acceptance tests or evidence of useful performance; especially do
not interpret them that way for a three-episode smoke run.

## Output format (schema version 2)

| File | Evidence |
|---|---|
| `benchmark_results.json` | Aggregate metrics, integer success counts, Wilson endpoints and every episode outcome; protocol, Git revision/dirty flag, Python/package/platform versions, optional checkpoint SHA-256 hashes. |
| `benchmark_results.csv` | Flat aggregate metrics, without embedding episode lists. |
| `benchmark_episodes.csv` | One row per episode: environment/policy labels, seeds, initial/final success, return, length, termination and truncation. |

All three payloads are serialized and staged before replacing existing output
files. Serialization or staging-write failures preserve the previous bundle;
individual replacements are atomic, but a process interruption during the
replacement sequence is not covered by a multi-file transaction. Use a fresh
output directory for each evidence-bearing run.

A dirty checkout is explicitly marked and is not an exact code identifier.
For shared results, commit the implementation first, run from that clean
revision, and retain the output and dependency versions. Output records avoid
machine usernames and absolute checkpoint paths. Timing fields naturally vary
between repeated runs; episode records should replay for the same code,
runtime and supported policy.

## Historical records and present limitations

See [the archive notes](results/README.md) before citing the original results.
Those files lack raw episodes, complete provenance and seeded random actions;
they are historical observations, not a newly reproduced baseline.

The tested language planner is a keyword-based `MockVLM`, and `SimGrounder`
uses privileged simulator object metadata. Unit-test coverage for these
components is not learned VLM or visual-grounding accuracy. The real VLM
wrapper is optional and was not evaluated by this benchmark; DINOv2 grounding
is a TODO. The hierarchical executor currently uses scripted policies and
handles `move_to`, `pick` and `place`. The named-target repair and its narrower
reaching evaluation are documented separately; the L1–L5 task success predicates
and physical manipulation limits remain. Inspecting parser support or executing
an integration test is not proof of successful end-to-end instruction following.
