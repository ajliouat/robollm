# RoboLLM — Development Log

> Real-time build diary. Updated as work happens — not retroactively.

---

## Status: v1.0.3 COMPLETE — Pick Primitive Training

### Pre-Development Setup (Week 0)
- [x] Install MuJoCo on Mac (via `pip install mujoco`)
- [x] Verify MuJoCo rendering (headless + viewer)
- [ ] Download and test PaliGemma-3B with 4-bit quantization locally
- [x] Set up MuJoCo environment with Franka Panda arm
- [x] Test basic joint control and rendering
- [ ] Benchmark MuJoCo step speed on Mac (should be >10K steps/sec)

---

## v1.0.0 — Scaffold + MuJoCo Environment (2026-02-22)

### What was built
Full project skeleton with working Gymnasium-compatible MuJoCo environment.

**MuJoCo scene (`envs/assets/tabletop_scene.xml`):**
- 7-DOF arm with Franka Panda-inspired kinematics (alternating Z/Y hinge axes)
- Parallel-jaw gripper (2 slide joints)
- 60×60 cm tabletop at 40 cm height
- Red 4 cm cube with free joint (pick-and-place target)
- 3 cameras: overhead, front, side
- Position-controlled PD actuators with per-joint kp tuning
- Home keyframe: arm in L-pose above table

**Environment (`envs/tabletop.py`):**
- Gymnasium-compatible: `reset()`, `step()`, `render()`, `close()`
- Delta end-effector control: action[0:3] = delta (x, y, z), action[3] = gripper
- Jacobian pseudoinverse (damped least squares, λ=0.01) for IK at every control step
- 20 Hz control frequency (25 MuJoCo substeps at dt=0.002)
- 29-D observation: joint pos (7) + vel (7) + gripper (1) + EE pose (7) + object pose (7)
- Randomized block placement on reset (seeded)
- Headless rendering via `mujoco.Renderer` (overhead camera, configurable resolution)

**Project structure:**
- `pyproject.toml` with mujoco, gymnasium, numpy, torch dependencies
- Dockerfile with OSMesa for headless rendering
- GitHub Actions CI: Ubuntu, Python 3.11, OSMesa, pytest
- Skeleton packages: `planner/`, `policies/`, `training/`, `evaluation/`
- 18 unit tests across 6 test classes

### Design decisions
```
1. Delta-EE control (not joint position):
   Matches the PROJECT_SPEC action space (4D). Uses Jacobian pseudoinverse
   for IK — damped version handles singularities smoothly.

2. Position-controlled PD actuators (not velocity/torque):
   More stable for RL; the PD controller handles low-level tracking,
   the policy only needs to output desired EE positions.

3. Simplified Franka model (not mujoco_menagerie):
   Self-contained — no external dependencies. 7 hinge joints with
   realistic Franka-like joint limits and alternating Z/Y axes.

4. Single block for v1.0.0:
   Multi-object spawning comes in v1.0.1. Base env has one red block
   to validate the full pipeline (env → obs → action → step → render).

5. Base env has no reward:
   Reward shaping is task-specific. Subclasses (PickPlaceEnv, etc.)
   override _compute_reward() and _check_terminated().
```

### Files added/modified
```
NEW:  pyproject.toml
NEW:  Dockerfile
NEW:  LICENSE
NEW:  .gitignore
NEW:  .github/workflows/ci.yml
NEW:  ROADMAP.md
NEW:  envs/__init__.py
NEW:  envs/tabletop.py
NEW:  envs/assets/tabletop_scene.xml
NEW:  planner/__init__.py
NEW:  policies/__init__.py
NEW:  training/__init__.py
NEW:  evaluation/__init__.py
NEW:  evaluation/results/.gitkeep
NEW:  tests/__init__.py
NEW:  tests/conftest.py
NEW:  tests/test_envs.py
NEW:  notebooks/.gitkeep
NEW:  videos/.gitkeep
MOD:  DEVELOPMENT_LOG.md
```

---

## v1.0.3 — Pick Primitive Training (2026-02-22)

### What was built
Training pipeline for the L1 pick-and-place task using SAC.

**Training script (`training/train_pick.py`):**
- Full training loop with SAC on PickPlaceEnv
- Periodic evaluation (deterministic policy, configurable episodes)
- Best-model checkpointing (saved when success rate improves)
- JSON results export with full eval history
- CLI interface with configurable steps, seed, eval interval

**Training results (seed=42, 130K steps):**
- Approach learning confirmed: reward −141 → −50 in 20K steps
- Early grasp success: 6.7% at 25K steps
- Entropy tuning converges: α 1.0 → 0.005
- Throughput: ~170 FPS on Apple Silicon (CPU-only)
- Full pick-place chain requires more steps or curriculum for sustained success

**Bug fix in PickPlaceEnv:**
- `reset()` was double-appending goal to obs (35D vs expected 32D)
- Fix: set placeholder goal before `super().reset()`, then rebuild obs

### Design decisions
```
1. 200-step episodes (not 500):
   Balance between giving agent enough time and not wasting steps
   on unrecoverable states. 200 at 20Hz = 10 seconds sim time.

2. Separate eval env (not reusing train env):
   Avoids state contamination. Eval uses deterministic policy
   and fixed seed offsets for reproducibility.

3. Best-model checkpointing:
   Only saves when success rate improves. Prevents disk fill
   from frequent saves during plateau phases.

4. Realistic reporting:
   6.7% success is honest. Full pipeline needs curriculum or
   500K+ steps for robust multi-phase manipulation.
```

### Files added/modified
```
NEW:  training/train_pick.py
NEW:  tests/test_training.py
NEW:  evaluation/results/pick_training_results.json
NEW:  evaluation/results/pick_training_summary.json
NEW:  checkpoints/pick/pick_best.pt (not committed — binary)
MOD:  envs/pick_place.py (reset bug fix)
MOD:  ROADMAP.md
MOD:  DEVELOPMENT_LOG.md
```

---

## v1.0.2 — SAC Implementation (2026-02-22)

### What was built
Clean from-scratch Soft Actor-Critic (Haarnoja et al. 2018) with all
components needed for RL training on the tabletop environments.

**Networks (`policies/networks.py`):**
- Actor: obs → 256 → 256 → (mean, log_std), squashed Gaussian via tanh
- Critic: (obs, action) → 256 → 256 → scalar Q-value
- TwinCritic: two independent Q-networks for clipped double-Q

**Replay buffer (`training/replay_buffer.py`):**
- Fixed-size circular buffer, numpy storage, torch batch sampling
- Capacity: 1M transitions (configurable), batch size 256

**SAC agent (`policies/sac.py`):**
- Automatic entropy tuning: target entropy = -dim(A)
- Polyak-averaged target networks (τ=0.005)
- Warmup: 5K random steps before learning begins
- Save/load checkpoint round-trip (all weights, optimizers, counters)
- Deterministic and stochastic action selection

**Training loop (`training/train.py`):**
- Generic env loop: collect, store, update, evaluate
- Periodic evaluation with deterministic policy
- Checkpoint saving at configurable intervals
- Episode reward tracking + periodic print logging

### Design decisions
```
1. Squashed Gaussian (not beta or truncated normal):
   Standard SAC formulation — tanh squashing with log-prob correction.
   Numerically stable with 1e-6 epsilon in log(1-tanh²+ε).

2. CPU-only torch for now:
   MuJoCo is CPU-bound anyway. GPU training scheduled for v1.0.3
   when we start 500K-step runs.

3. Lazy __init__ imports to break circular dependency:
   policies/__init__ → policies.sac → training.replay_buffer →
   training/__init__ → training.train → policies.sac.
   Fix: __getattr__ lazy imports in both __init__.py files.

4. SACConfig dataclass (not YAML/JSON):
   Simple, type-safe, no file I/O needed. Config is passed in code.
   Hyperparameter sweep will use grid over SACConfig fields.

5. Separate replay buffer (not integrated in agent):
   Clean separation; buffer can be reused for other algorithms.
   Agent owns a buffer instance but doesn't subclass it.
```

### Files added/modified
```
NEW:  policies/networks.py
NEW:  policies/sac.py
NEW:  training/replay_buffer.py
NEW:  training/train.py
NEW:  tests/test_sac.py
MOD:  policies/__init__.py
MOD:  training/__init__.py
MOD:  ROADMAP.md
MOD:  DEVELOPMENT_LOG.md
```

---## v1.0.1 — Environment Variants + Object Spawning (2026-02-22)

### What was built
Dynamic multi-object spawning system and three task-specific environments
with shaped rewards for hierarchical skill learning.

**Object spawner (`envs/object_spawner.py`):**
- 3 shapes (box, cylinder, sphere) × 6 colours (red, green, blue, yellow, orange, purple)
- Non-overlapping position sampling with 6 cm minimum spacing
- XML injection: reads base MJCF, removes hardcoded block, injects N free-joint bodies
- Keyframe qpos auto-generated to include all object initial states
- Deterministic with seeded numpy Generator

**Multi-object base env (`envs/multi_object_env.py`):**
- Extends Gymnasium Env (not TabletopEnv) — clean reimplementation with dynamic model loading
- Scene rebuilt on every reset (new random objects each episode)
- Variable obs dim: 22 + 7 × n_objects (arm state + EE pose + per-object pose)
- Same delta-EE Jacobian pseudoinverse control as v1.0.0
- Convenience: `object_pos(i)`, `object_color(i)`, `object_shape(i)`

**Pick-Place env L1 (`envs/pick_place.py`):**
- 1 object, randomized goal position on table
- 4-phase shaped reward: approach → grasp → lift → place
- Obs augmented with goal position (+3D → 32D total)
- Terminates on successful placement (obj within 4 cm of goal, after lift)

**Color-Pick env L2 (`envs/color_pick.py`):**
- N objects with unique colours, random target colour per episode
- Obs augmented with one-hot colour target (+6D → 49D for n=3)
- Correct-grasp bonus (+10), wrong-grasp penalty (−5), lift+success bonus (+65)
- Terminates when correct object is lifted

**Stack env L3 (`envs/stack.py`):**
- N objects, random stack order, random stack target position
- Obs augmented with stack pos, order, progress (+3+n+1 → 50D for n=3)
- Contiguous stack detection: checks XY alignment + Z height per level
- Rewards: per-object stack bonus (+20), full-stack bonus (+100)
- Terminates when all objects correctly stacked

### Design decisions
```
1. Dynamic XML injection (not multiple MJCF files):
   Single base scene definition; objects added at runtime via
   ElementTree manipulation. Avoids file explosion for N/shape/colour
   combinations. Model reloaded each reset — fast enough (<5ms).

2. MultiObjectEnv separate from TabletopEnv:
   Clean break — TabletopEnv stays as the minimal 1-block test env.
   MultiObjectEnv handles variable object count, dynamic model loading,
   and variable-size observations.

3. Shaped rewards with explicit phases:
   Clear reward signal per sub-task (approach, grasp, lift, place).
   Makes SAC training feasible in v1.0.3 without curriculum.

4. Stack detection is contiguous-from-bottom:
   Object at level k counts as stacked only if levels 0..k-1 are also
   correct. Prevents credit for floating objects.

5. ColorPick reset ordering fix:
   Parent reset() calls _get_obs() which needs _target_color. Set
   placeholder before super().reset(), then pick real target after.
```

### Files added/modified
```
NEW:  envs/object_spawner.py
NEW:  envs/multi_object_env.py
NEW:  envs/pick_place.py
NEW:  envs/color_pick.py
NEW:  envs/stack.py
NEW:  tests/test_object_spawner.py
NEW:  tests/test_task_envs.py
MOD:  envs/__init__.py
MOD:  ROADMAP.md
MOD:  DEVELOPMENT_LOG.md
```

---
