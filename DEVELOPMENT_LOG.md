# RoboLLM — Development Log

> Real-time build diary. Updated as work happens — not retroactively.

---

## Status: v1.0.8 COMPLETE — Benchmark Suite + Demo Videos

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

## v1.0.8 — Benchmark Suite + Demo Videos (2026-02-22)

### What was built
Full benchmark evaluation suite and video recording pipeline.

**Benchmark runner (`evaluation/benchmark.py`):**
- `evaluate_policy()`: generic evaluator for any (env, policy) pair
- `run_full_benchmark()`: 100 episodes × 8 env/policy configs = 800 total
- Wilson score 95% CI for binomial success rate
- Auto-export: JSON + CSV to `evaluation/results/`
- `check_thresholds()`: validates against PROJECT_SPEC targets
- CLI entry point with --episodes, --seed, --output args

**Video recorder (`evaluation/record_video.py`):**
- `record_episode()`: captures RGB frames from MuJoCo renderer
- `save_gif()`: Pillow-based animated GIF export (no ffmpeg needed)
- `save_frames_as_png()`: fallback individual frame export
- `record_demos()`: records 3 task types (PickPlace, MoveTo, Sort)

**Benchmark results (100 episodes per pair, seed=42):**

| Environment    | Policy   | Success Rate | Mean Return | Mean Length |
|---------------|----------|-------------|-------------|-------------|
| L1-PickPlace  | random   | 0.0% ± 1.8% | -142.4      | 200.0       |
| L1-PickPlace  | scripted | 0.0% ± 1.8% | -53.2       | 200.0       |
| MoveTo        | random   | 0.0% ± 1.8% | -715.8      | 200.0       |
| MoveTo        | scripted | 20.0% ± 7.8% | -260.3     | 187.6       |
| L2-ColorPick  | random   | 0.0% ± 1.8% | -142.6      | 200.0       |
| L3-Stack      | random   | 0.0% ± 1.8% | -138.0      | 200.0       |
| L4-Sort       | random   | 0.0% ± 1.8% | -706.5      | 200.0       |
| L5-Language   | random   | 4.0% ± 4.1% | -955.2      | 192.0       |

**Demo GIFs recorded:**
- `videos/demo_pick_place.gif` — L1 PickPlace, scripted, 201 frames
- `videos/demo_move_to.gif` — MoveTo, scripted, 125 frames (success)
- `videos/demo_sort_random.gif` — L4 Sort, random baseline, 201 frames

**Key observations:**
- Scripted MoveTo: 20% SR with P-controller. Simple approach tasks solvable.
- L1 scripted: 0% SR in 200 steps. The 8-phase sequence is tight. RL training
  (v1.0.3) showed 6.7% after 130K steps — needs more steps or curriculum.
- L5 random: 4% SR by chance (some "move left" conditions met randomly).
- Scripted controller gets 2.6× better returns than random on L1 despite 0% SR.

### Files added/modified
```
NEW:  evaluation/benchmark.py
NEW:  evaluation/record_video.py
NEW:  evaluation/results/benchmark_results.json
NEW:  evaluation/results/benchmark_results.csv
NEW:  videos/demo_pick_place.gif
NEW:  videos/demo_move_to.gif
NEW:  videos/demo_sort_random.gif
NEW:  tests/test_v108.py
MOD:  ROADMAP.md
MOD:  DEVELOPMENT_LOG.md
```

---

## v1.0.7 — Full Hierarchical Pipeline (2026-02-22)

### What was built
End-to-end hierarchical execution pipeline, two new task environments (L4, L5),
and full integration of planner → grounder → scripted policies → MuJoCo.

**HierarchicalExecutor (`evaluation/pipeline.py`):**
- Takes a TaskPlan + env, executes each sub-task sequentially
- Per-subtask: ground target → select policy → run episodes → record result
- Supports move_to, pick, and place primitives
- StepResult/ExecutionResult dataclasses with success, reward, step counts
- Handles grounding failures gracefully (no crash, error in result)
- `_build_scene_info()` extracts object specs from env for grounder

**SortEnv L4 (`envs/sort.py`):**
- N objects → N shuffled target zones on table
- 52D obs: arm (22) + objects (7×3) + zone positions (3×3)
- Per-object reward (+20 when in correct zone) + full-sort bonus (+100)
- Terminates when all objects in correct zones (5 cm threshold)
- Shuffled zone assignment ensures different ordering each episode

**ComplexLanguageEnv L5 (`envs/complex_language.py`):**
- 5 instruction templates requiring multi-step execution
- Templates: pick-and-place, move-left, stack, move-all-center, pick-both
- 59D obs: arm (22) + objects (7×3) + instruction encoding (16D hash)
- Condition checking: obj_near, obj_left, obj_on, all_center
- Instruction text in info dict for planner consumption
- Per-condition reward (+20 met, -5 unmet) + all-conditions bonus (+100)

**Test results: 268 tests pass (23 new)**
- SortEnv: 8/8 (obs shape, action space, zones, reward, episodes)
- ComplexLanguageEnv: 8/8 (obs, instructions, conditions, determinism)
- HierarchicalExecutor: 6/6 (move_to, pick+place, empty plan, grounding fail)
- EndToEnd: 1/1 (full planner → executor pipeline)

### Bug fixed
- `ObjectSpec.color` → `ObjectSpec.color_name`: The dataclass uses
  `color_name` (str) not `color`. Fixed in both `complex_language.py`
  and `evaluation/pipeline.py`. Root cause: field naming assumption
  without checking the actual dataclass definition.

### Design decisions
```
1. Sequential sub-task execution (not parallel):
   Robot has one arm — sub-tasks are inherently sequential.
   Failure in one doesn't abort the rest (continues for logging).

2. 16D instruction encoding (hash-based placeholder):
   Real instruction embeddings would come from sentence-transformers.
   Hash-based encoding preserves determinism without model dependency.

3. SortEnv shuffled zones (not fixed assignment):
   Prevents memorization. Each reset produces a different object→zone
   mapping, forcing generalisation.

4. 5 condition types (not arbitrary):
   Covers the key spatial relations: proximity, direction, stacking,
   and global arrangement. Extensible for v1.0.8.

5. Tests use actual spawned colors (not hardcoded):
   After the ObjectSpec bug, tests query env._obj_specs[0].color_name
   to build instructions that match the actual scene.
```

### Files added/modified
```
NEW:  evaluation/pipeline.py
NEW:  envs/sort.py
NEW:  envs/complex_language.py
NEW:  tests/test_v107.py
MOD:  envs/__init__.py (added SortEnv, ComplexLanguageEnv)
MOD:  ROADMAP.md
MOD:  DEVELOPMENT_LOG.md
```

---

## v1.0.6 — Object Grounding (2026-02-22)

### What was built
Object grounding system that maps text descriptions to sim object poses.

**SimGrounder (`planner/grounder.py`):**
- Score-based matching: color match (+1.0), shape match (+0.5), name match (+0.2)
- Color synonyms: 15+ aliases (crimson→red, azure→blue, jade→green, etc.)
- Shape synonyms: block/cube/brick→box, ball/orb/globe→sphere, tube/rod→cylinder
- Supports 3 scene info formats: objects list, object_names dict, obj_specs
- Returns GroundingResult with matched object, all candidates sorted by score

**VisualGrounder (optional):**
- DINOv2-small backbone (lazy-loaded, CPU or GPU)
- Image crop → 384D embedding → nearest-neighbor match
- Falls back to SimGrounder when no image provided

**Grounding accuracy: 20/20 queries on 3-object scene**
- Canonical names: "red block", "blue cylinder", "green sphere"
- Synonyms: "crimson cube", "azure tube", "emerald ball"
- Partial: "the block" (shape-only), "red" (color-only)
- Verbose: "red box on the table", "big blue cylinder"

### Files added/modified
```
NEW:  planner/grounder.py
NEW:  tests/test_v106.py
MOD:  planner/__init__.py
MOD:  ROADMAP.md
MOD:  DEVELOPMENT_LOG.md
```

---

## v1.0.5 — VLM Planner Integration (2026-02-22)

### What was built
Full VLM-based task decomposition pipeline with abstract interface,
mock backend for testing, and optional HuggingFace Transformers backend.

**VLM wrapper (`planner/vlm_wrapper.py`):**
- VLMBase: abstract base with generate() + decompose() + JSON parsing
- MockVLM: rule-based keyword matching for testing (no GPU required)
  - Detects pick/place/move/stack/sort via word sets
  - Handles 6 colors, implicit pick when place detected
  - Override decompose() to skip prompt wrapping (avoids keyword pollution)
- TransformersVLM: HuggingFace backend for PaliGemma/Phi-3-Vision
  - Lazy model loading, 4-bit quantization support
  - PIL/numpy image handling, prompt processing

**Prompt templates (`planner/prompt_templates.py`):**
- System prompt defining available primitives (move_to, pick, place)
- Object color and shape vocabulary
- Strict output format: JSON array with primitive/target/description
- Ordering rules: must approach before pick, pick before place
- 20+ evaluation scenarios with expected primitive sequences

**Task parser (`planner/task_parser.py`):**
- SubTask and TaskPlan dataclasses
- Primitive validation (move_to, pick, place only)
- Color normalisation with aliases (crimson→red, azure→blue, etc.)
- Ordering validation: warns on double-pick or place-without-pick
- Handles malformed input gracefully (non-dicts, missing fields)

**Planner (`planner/planner.py`):**
- High-level orchestrator: instruction → VLM → parse → TaskPlan
- Planning history tracking for logging/debugging
- Clean interface: `planner.plan("pick up the red block")`

**Decomposition accuracy on 20+ test scenarios:**
- All 15 scenarios with exact primitive expectations: 100% match
- All 20 scenarios produce valid, non-empty plans
- MockVLM correctly handles: simple pick, pick+place, move-only,
  stack (multi-object), sort, ambiguous instructions

### Design decisions
```
1. Abstract VLM interface (not hardcoded PaliGemma):
   VLMBase → MockVLM for testing, TransformersVLM for real models.
   Clean separation lets pipeline tests run without GPU.

2. MockVLM keyword matching (not LLM):
   Deterministic, fast, no dependencies. Perfect for pipeline
   integration tests. Accuracy measured on 20+ scenarios.

3. MockVLM overrides decompose() directly:
   The base class wraps instruction in a system prompt before
   generate(), which would pollute keyword matching. Override
   skips the prompt template for clean word detection.

4. Task parser as separate validation layer:
   VLM output → parse_subtasks() → TaskPlan. Handles malformed
   JSON, unknown primitives, ordering violations. Defense in depth.

5. Color aliases:
   VLMs may output "crimson" instead of "red". Normalisation layer
   maps 15+ aliases to the 6 canonical colors.
```

### Files added/modified
```
NEW:  planner/vlm_wrapper.py
NEW:  planner/prompt_templates.py
NEW:  planner/task_parser.py
NEW:  planner/planner.py
NEW:  tests/test_v105.py
MOD:  planner/__init__.py
MOD:  ROADMAP.md
MOD:  DEVELOPMENT_LOG.md
```

---

## v1.0.4 — Place + Move Primitives (2026-02-22)

### What was built
Two new primitive environments and scripted baseline controllers for comparison.

**MoveToEnv (`envs/move_to.py`):**
- Simplest primitive — move EE close to object, no grasping
- 29D observation (base multi-object obs, 1 object)
- Reward: -5 × dist(EE, obj) + 50 bonus when within 3 cm
- Terminates on success (3 cm threshold)

**PlaceEnv (`envs/place.py`):**
- Place held object at goal position on table
- Object initialised near EE with closed gripper (post-grasp simulation)
- 32D observation (29 base + 3 goal position)
- Reward: -10 × dist(obj, goal) - height penalty + 50 success bonus
- Terminates when object within 4 cm of goal

**Scripted baselines (`policies/scripted.py`):**
- ScriptedPickPlace: 8-phase state machine (approach → descend → grasp → lift → move → lower → release → done)
- ScriptedMoveTo: Simple P-controller pointing EE toward object
- Both use privileged info (exact object/goal positions from env info dict)

**Episode truncation (`envs/multi_object_env.py`):**
- Added `max_episode_steps=200` default to all MultiObjectEnv subclasses
- Step counter reset on `reset()`, truncation flag set in `step()`
- Critical for evaluation — random agents no longer run forever

**Comparison results (25 episodes per pair):**
| Environment | Policy   | Success% | Mean Return | Mean Length |
|-------------|----------|----------|-------------|-------------|
| Move To     | Random   |     0.0% |      -722.3 |       200.0 |
| Move To     | Scripted |    20.0% |      -281.0 |       186.0 |
| Pick Place  | Random   |     0.0% |      -142.7 |       200.0 |
| Pick Place  | Scripted |     0.0% |       -64.6 |       200.0 |
| Place       | Random   |     0.0% |    -10297.7 |       200.0 |
| Place       | Scripted |     0.0% |    -11733.5 |       200.0 |

Scripted MoveTo achieves 20% success (simple P-control can reach objects).
PickPlace scripted gets 2.2× better returns than random but 0% full success —
the multi-phase task is hard even for a privileged controller within 200 steps.

### Design decisions
```
1. 200-step truncation (not infinite):
   200 steps × 20Hz = 10s sim time. Without truncation, random eval
   runs forever. Truncation is in the base class so all envs get it.

2. Post-grasp init for PlaceEnv:
   Object spawns in gripper, avoiding the grasp problem entirely.
   Tests place skill in isolation — clean primitive decomposition.

3. Scripted baselines use privileged info:
   Fair comparison: scripted knows exact poses (from info dict),
   RL agent learns from obs only. This sets an upper bound.

4. 25 episodes (not 100):
   MuJoCo stepping at 200 steps × 25 substeps = 5000 physics steps/ep.
   25 episodes × 6 pairs ≈ 750K physics steps — runs in ~8s.
```

### Files added/modified
```
NEW:  envs/move_to.py
NEW:  envs/place.py
NEW:  policies/scripted.py
NEW:  evaluation/compare_primitives.py
NEW:  evaluation/results/primitive_comparison.json
NEW:  tests/test_v104.py
MOD:  envs/__init__.py (added MoveToEnv, PlaceEnv exports)
MOD:  envs/multi_object_env.py (added max_episode_steps truncation)
MOD:  policies/__init__.py (added ScriptedPickPlace, ScriptedMoveTo)
MOD:  ROADMAP.md
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
