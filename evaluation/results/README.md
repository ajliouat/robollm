# Historical result archive

The existing JSON/CSV numeric records are preserved without modification.
They predate the reproducibility/reporting repair and must not be presented
as fresh measurements from the current implementation.

`benchmark_results.json` has timestamp `2026-02-22 14:48:51` and 800 episodes
(100 for each of eight environment/policy configurations). It records the
observed rates and returns, but not raw episode outcomes, exact executing
revision, runtime versions or random action seeds. The old runner reset the
environment with a seed but did not seed `action_space.sample()`. Its `ci95`
is a Wilson half-width around the adjusted Wilson centre, **not** an error
bar about the observed success rate.

Intervals below are recalculated from the recorded count/rate and `n=100`;
this is a statistical correction of archived values, not a rerun:

| Historical observation | Count | Correct 95% Wilson interval |
|---|---:|---:|
| Zero-success configurations | 0/100 | [0.00%, 3.70%] |
| Scripted MoveTo | 20/100 | [13.34%, 28.88%] |
| Random L5 Language | 4/100 | [1.57%, 9.84%] |

The historical L1 mean returns are -142.37 (random) and -53.17 (scripted);
MoveTo returns are -715.76 and -260.33. These shaped-return differences do
not support multiplicative “times better” claims. L5 success is a final-state
predicate that can already hold initially; it is not evidence of random-policy
language understanding.

`primitive_comparison.json` is a separate 25-episode comparison, with different
returns. Do not combine its returns with the 100-episode benchmark as though
they came from one experiment. `pick_training_summary.json` describes a 50K
step run and an interrupted 130K/200K step run; an early reported 6.7% success
was not sustained. It does not establish successful trained manipulation or
the proposed 500K-step/T4 training budget.

Use the [current protocol](../README.md) for new evaluations and write them to
a separate directory. Historical results have deliberately not been relabeled
with a current commit, regenerated from imagined episodes, or overwritten.
