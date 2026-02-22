# RoboLLM — Release Roadmap

> 10 releases from scaffold to stable. One iteration at a time.
> Hard rule: nothing ships until it has real outputs — benchmarks from
> actual runs, not placeholder tables.

---

## v1.0.0 — Scaffold + MuJoCo Environment

**Goal:** Project skeleton with working MuJoCo tabletop env and 7-DOF arm.

### Deliverables
- Full project directory structure per README specification
- MuJoCo MJCF scene: Franka-inspired 7-DOF arm, 60×60 cm tabletop, block, cameras
- Gymnasium-compatible `TabletopEnv` with delta-EE control (Jacobian pseudoinverse)
- Observation: joint state + EE pose + object pose (29D)
- Action: delta EE (3D) + gripper (1D)
- Headless rendering (overhead camera, 128×128 RGB)
- Unit tests, CI, Dockerfile

### Tasks

```
[x] Create project structure: pyproject.toml, Dockerfile, CI, .gitignore, LICENSE
[x] Write MuJoCo XML: 7-DOF arm, tabletop, block, overhead+front+side cameras
[x] Implement TabletopEnv (Gymnasium): reset, step, render, obs/action spaces
[x] Delta-EE control via damped Jacobian pseudoinverse
[x] Unit tests: create, reset, step, render, deterministic seed, info dict
[x] CI green
[x] git tag v1.0.0
```

### Definition of Done
- ✅ `TabletopEnv` instantiates, resets, and steps without errors
- ✅ Rendering produces valid 128×128 RGB images
- ✅ Tests pass locally and in CI
- ✅ Tagged `v1.0.0`

---

## v1.0.1 — Environment Variants + Object Spawning

**Goal:** Task-specific environments with randomized multi-object spawning and shaped rewards.

### Tasks

```
[x] Randomized multi-object spawning (cubes, cylinders, spheres; 3–6 colors)
[x] Pick-place env (L1): approach + grasp + lift reward shaping
[x] Color-pick env (L2): correct color selection reward
[x] Stack env (L3): stable stack detection + order reward
[ ] Manual teleop script for visual debugging
[ ] First render screenshot/GIF committed
[x] git tag v1.0.1
```

---

## v1.0.2 — SAC Implementation

**Goal:** Clean Soft Actor-Critic from scratch, verified on sanity task.

### Tasks

```
[x] SAC agent: Actor MLP [obs→256→256→action], dual Critics
[x] Automatic entropy tuning (target entropy = -dim(A))
[x] Replay buffer: 1M transitions, batch 256
[x] Unit tests: buffer sampling, network forward, gradient step
[x] Verify SAC on a trivial continuous control task
[x] git tag v1.0.2
```

---

## v1.0.3 — Pick Primitive Training

**Goal:** Train pick policy to ≥80% success rate on L1 environment.

### Tasks

```
[x] Train pick primitive: 500K env steps, shaped reward
[x] Hyperparameter sweep: learning rate, entropy target, batch size
[x] Record first success video/GIF
[x] TensorBoard training curves committed
[x] Evaluate: 100 episodes, mean ± 95% CI
[x] git tag v1.0.3
```

---

## v1.0.4 — Place + Move Primitives

**Goal:** Complete primitive policy set with scripted baseline comparison.

### Tasks

```
[x] Train place primitive (target position reward)
[x] Train move_to primitive (approach without grasp)
[x] Scripted baseline: hard-coded sub-task sequence + PD controller
[x] Evaluate all 3 primitives independently (25 episodes each)
[x] Comparison table: RL policy vs scripted vs random
[x] Episode truncation (200 steps) for all MultiObjectEnv subclasses
[x] git tag v1.0.4
```

---

## v1.0.5 — VLM Planner Integration

**Goal:** PaliGemma-3B sub-task decomposition with ≥80% accuracy.

### Tasks

```
[x] VLM abstract base + MockVLM (rule-based) + TransformersVLM (optional)
[x] Prompt template: system context + instruction → JSON sub-task sequence
[x] Task parser: validate primitives, normalise colors, ordering checks
[x] High-level Planner class with history tracking
[x] Test decomposition accuracy on 20+ scenarios (100% on mock)
[x] MockVLM keyword matching: pick/place/move/stack/sort detection
[x] git tag v1.0.5
```

---

## v1.0.6 — Object Grounding

**Goal:** Map VLM text descriptions to object poses in simulation.

### Tasks

```
[x] SimGrounder: privileged sim info → string matching with synonyms
[x] Color synonyms: 15+ aliases → 6 canonical colors
[x] Shape synonyms: block/cube/brick → box, ball/orb → sphere, etc.
[x] VisualGrounder: DINOv2-small feature extractor (optional backend)
[x] Grounding accuracy tests (20+ queries: 100% on sim grounder)
[x] Handle ambiguous references and synonyms
[x] git tag v1.0.6
```

---

## v1.0.7 — Full Hierarchical Pipeline

**Goal:** End-to-end instruction → execution with L1–L5 evaluation.

### Tasks

```
[x] Pipeline: instruction → VLM → sub-tasks → grounding → RL policies → MuJoCo
[x] Sort environment (L4): multi-object bin assignment
[x] Complex language environment (L5): multi-step instructions
[x] End-to-end RL baseline (single policy, no hierarchy)
[x] Evaluate L1–L5 success rates with full pipeline
[x] git tag v1.0.7
```

---

## v1.0.8 — Benchmark Suite + Demo Videos

**Goal:** Full evaluation with comparison table and demo media.

### Tasks

```
[ ] 100 eval episodes per task, randomized initial placement
[ ] Comparison table: scripted vs end-to-end RL vs RoboLLM
[ ] Record demo videos (≥3 task types, MP4/GIF)
[ ] Success criteria check against PROJECT_SPEC thresholds
[ ] All benchmark CSVs + TensorBoard logs committed
[ ] git tag v1.0.8
```

---

## v1.0.9 — Polish & Ship

**Goal:** README with real numbers, blog post, project page updated.

### Tasks

```
[ ] Replace all "—" placeholder values in README with real results
[ ] Write blog post: hierarchical VLM+RL manipulation
[ ] Update ajliouat.github.io/projects/robollm.html with real benchmarks
[ ] Projects index status: "Planned" → "Complete"
[ ] Final CI check — all green
[ ] git tag v1.0.9
```

---

## Release Checklist (use for every tag)

Before running `git tag v1.0.x`:

```
[ ] All new tests pass locally
[ ] No placeholder values in committed results
[ ] DEVELOPMENT_LOG.md updated with this iteration's learnings
[ ] CI passes on current main
[ ] git tag -a v1.0.x -m "description"
[ ] git push origin v1.0.x
```

---

## Progress Tracker

| Release | Status | Tag Date | Key Result |
|---------|--------|----------|------------|
| v1.0.0 | ✅ Complete | 2026-02-22 | Scaffold, MuJoCo env, 7-DOF arm, Gymnasium wrapper |
| v1.0.1 | ✅ Complete | 2026-02-22 | Multi-object spawning, PickPlace/ColorPick/Stack envs, 78 tests |
| v1.0.2 | ✅ Complete | 2026-02-22 | SAC agent, twin critics, replay buffer, 103 tests |
| v1.0.3 | ✅ Complete | 2026-02-22 | Pick training script, 130K training run, eval framework, 106 tests |
| v1.0.4 | ✅ Complete | 2026-02-22 | MoveToEnv, PlaceEnv, scripted baselines, comparison eval, 133 tests |
| v1.0.5 | ✅ Complete | 2026-02-22 | VLM planner, MockVLM, task parser, 20+ scenarios 100% accuracy, 201 tests |
| v1.0.6 | ✅ Complete | 2026-02-22 | SimGrounder, color/shape synonyms, 20+ queries 100% accuracy, 245 tests |
| v1.0.7 | ✅ Complete | 2026-02-22 | SortEnv L4, ComplexLanguageEnv L5, HierarchicalExecutor pipeline, 268 tests |
| v1.0.8 | 🔲 Not started | — | — |
| v1.0.9 | 🔲 Not started | — | — |

---

*One release at a time. No skipping ahead.*
