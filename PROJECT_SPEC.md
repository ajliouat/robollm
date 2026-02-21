# RoboLLM — Technical Specification

## 1. Problem Statement

Current robotic systems either use hand-coded task planners (brittle, can't generalize to new instructions) or end-to-end learned policies (require enormous data, poor at multi-step tasks). The emerging approach is hierarchical: use a foundation model (LLM/VLM) for high-level planning and RL-trained policies for low-level control.

This project implements this hierarchical architecture at a scale reproducible on a single T4 GPU, using MuJoCo simulation.

## 2. System Components

### 2.1 MuJoCo Simulation Environment

**Robot:** 7-DoF arm (Franka Emika Panda model) with parallel-jaw gripper
**Workspace:** 60×60cm tabletop with randomized object placement
**Objects:** Colored blocks (4cm cubes), cylinders, spheres — spawned randomly per episode
**Physics:** Timestep 0.002s, control frequency 20Hz (100 sim steps per action)

**Observation space:**
- Joint positions (7)
- Joint velocities (7)  
- Gripper state (1)
- End-effector pose (7: xyz + quaternion)
- Object poses (N × 7)
- RGB image from overhead camera (128×128)

**Action space:** Delta end-effector position (3) + gripper open/close (1) = 4 continuous

### 2.2 VLM Planner

**Model:** PaliGemma-3B (4-bit GPTQ) or Phi-3-Vision (4-bit)

**Input:** Overhead RGB image + natural language instruction
**Output:** JSON list of sub-tasks

**Prompt template:**
```
You are a robot task planner. Given an image of a tabletop workspace and a human instruction, 
decompose the instruction into a sequence of primitive actions.

Available primitives:
- pick(object_description) — grasp the described object
- place(x, y) — place held object at relative position
- place_on(object_description) — place held object on top of described object
- move_to(x, y) — move arm above position without grasping

Instruction: "{instruction}"

Output JSON array of primitives:
```

**Grounding:** Each primitive is mapped to a target pose in the environment using object detection (DINOv2 features + nearest neighbor matching).

### 2.3 RL Policies

Train one policy per primitive action (pick, place, move):

**Algorithm:** SAC (Soft Actor-Critic)  
- Off-policy, sample-efficient
- Automatic entropy tuning
- Better than PPO for continuous control with shaped rewards

**Network architecture:**
- Actor: MLP [obs_dim → 256 → 256 → action_dim], tanh squashing
- Critic (×2): MLP [obs_dim + action_dim → 256 → 256 → 1]
- State representation: proprioceptive + object pose (no raw images for policy)

**Reward shaping (pick primitive):**
```python
reward = (
    -0.1 * distance_to_object          # Approach reward
    + 1.0 * grasp_success              # Grasp reward (binary)
    + 2.0 * lift_success               # Lift reward (object > 5cm above table)
    - 0.01 * action_penalty             # Smooth actions
)
```

**Training budget:**
- 500K environment steps per primitive (~4 hours on T4)
- Replay buffer: 1M transitions
- Batch size: 256, learning rate: 3e-4

### 2.4 Evaluation Protocol

**Per-task evaluation:**
- 100 episodes, randomized initial object placement
- Max 200 steps per episode
- Success: task-specific binary criterion (see README task table)
- Report: mean success rate ± 95% confidence interval

**Baselines:**
1. **Scripted planner:** Hard-coded sub-task sequence + PD controller (upper bound for planning, lower bound for generalization)
2. **Random policy:** Random actions (sanity check)
3. **End-to-end RL:** Single policy for entire task (no hierarchical decomposition)

## 3. Data & Artifacts

| Artifact | Format | Storage |
|----------|--------|---------|
| RL checkpoints | `.pt` files | `checkpoints/` in repo |
| Training curves | TensorBoard logs | `runs/` in repo |
| Demo videos | MP4/GIF | `videos/` in repo + README |
| VLM planner outputs | JSON logs | `evaluation/results/` |
| Benchmark tables | CSV | `evaluation/results/` |

## 4. Success Criteria

| Metric | Threshold |
|--------|-----------|
| L1 pick-place success rate | > 85% |
| L2 color-conditioned success | > 70% |
| L3 stacking success | > 50% |
| VLM planner decomposition accuracy | > 80% correct sub-task sequences |
| LLM-guided > scripted baseline on L5 | Yes |
| Demo videos showing successful episodes | At least 3 task types |
| CI tests pass | ✓ |

## 5. Timeline

| Week | Milestone |
|------|-----------|
| 1-2 | MuJoCo env setup (tabletop, robot, objects). Verify rendering on Mac. |
| 3-4 | Implement SAC. Train pick primitive on T4. Record first success videos. |
| 5 | Train place, move primitives. Scripted baseline evaluation. |
| 6-7 | VLM planner integration. Prompt engineering. Test decomposition accuracy. |
| 8 | Full hierarchical pipeline. Evaluate L1-L5 tasks. |
| 9-10 | Benchmark suite. Demo videos. README with real results. Blog post. |
