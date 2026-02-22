# RoboLLM — Development Log

> Real-time build diary. Updated as work happens — not retroactively.

---

## Status: v1.0.0 COMPLETE — Scaffold + MuJoCo Environment

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
