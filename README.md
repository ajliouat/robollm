# RoboLLM — Language-Grounded Robotic Manipulation

**LLM × Robotics × GPU Compute**

> A language-driven robotic manipulation system that uses a small VLM for task planning and RL-trained policies for motor control, evaluated in MuJoCo simulation.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---

## Overview

RoboLLM bridges the gap between language understanding and physical manipulation. A user gives a natural language instruction (e.g., "stack the red block on the blue one"), a vision-language model decomposes it into sub-tasks, and RL-trained motor policies execute each step in simulation.

This is in the lineage of SayCan (Ahn et al., 2022), Code as Policies (Liang et al., 2023), and RT-2 (Brohan et al., 2023) — but scoped to be reproducible on a single T4 GPU.

## Architecture

```
User instruction (text)
        │
        ▼
┌─────────────────┐     ┌──────────────┐
│  VLM Planner    │────▶│  Sub-task     │
│  (PaliGemma-3B  │     │  Sequence     │
│   4-bit quant)  │     │  [pick, move, │
│                 │◀────│   place, ...]  │
│  Scene image    │     └──────────────┘
└─────────────────┘            │
                               ▼
                    ┌──────────────────┐
                    │  RL Policy Head  │
                    │  (SAC/PPO)       │
                    │  per sub-task    │
                    └──────────────────┘
                               │
                               ▼
                    ┌──────────────────┐
                    │  MuJoCo          │
                    │  Tabletop Env    │
                    │  (7-DoF arm)     │
                    └──────────────────┘
```

## Tasks

| Task Level | Description | Objects | Success Metric |
|------------|-------------|---------|----------------|
| **L1 — Basic** | Pick and place single object | 1 | Object at target ± 2cm |
| **L2 — Color** | "Pick the red block" (distinguish by color) | 3 | Correct object at target |
| **L3 — Stack** | Stack blocks in specified order | 2-3 | Stable stack, correct order |
| **L4 — Sort** | Sort objects by color into bins | 4-6 | All objects in correct bins |
| **L5 — Language** | Complex multi-step instruction | 3+ | All sub-tasks completed |

## Project Structure

```
robollm/
├── README.md
├── PROJECT_SPEC.md
├── DEVELOPMENT_LOG.md
├── ROADMAP.md
├── LICENSE
├── pyproject.toml
├── Dockerfile
├── envs/                        # MuJoCo environments
│   ├── tabletop.py              # Base tabletop environment (29D obs)
│   ├── multi_object_env.py      # Multi-object base (dynamic spawning)
│   ├── object_spawner.py        # MJCF injection (3 shapes × 6 colors)
│   ├── pick_place.py            # L1: single pick-place (32D obs)
│   ├── color_pick.py            # L2: color-conditioned pick (49D obs)
│   ├── stack.py                 # L3: stacking (50D obs)
│   ├── sort.py                  # L4: multi-object sorting (52D obs)
│   ├── complex_language.py      # L5: complex instructions (59D obs)
│   ├── move_to.py               # Approach primitive (29D obs)
│   ├── place.py                 # Place primitive (32D obs)
│   └── assets/
│       └── tabletop_scene.xml   # MuJoCo scene definition
├── planner/                     # VLM-based task decomposition
│   ├── vlm_wrapper.py           # VLMBase / MockVLM / TransformersVLM
│   ├── prompt_templates.py      # System prompts + eval scenarios
│   ├── task_parser.py           # SubTask/TaskPlan + validation
│   ├── planner.py               # High-level orchestrator
│   └── grounder.py              # SimGrounder / VisualGrounder
├── policies/                    # RL + scripted policies
│   ├── sac.py                   # SAC agent (auto entropy, Polyak)
│   ├── networks.py              # Actor-Critic MLPs
│   └── scripted.py              # ScriptedPickPlace, ScriptedMoveTo
├── training/
│   ├── train.py                 # Generic training loop
│   ├── train_pick.py            # Pick primitive training script
│   └── replay_buffer.py         # 1M circular replay buffer
├── evaluation/
│   ├── benchmark.py             # Full benchmark suite (100 ep/task)
│   ├── pipeline.py              # HierarchicalExecutor (end-to-end)
│   ├── compare_primitives.py    # Scripted vs random comparison
│   ├── record_video.py          # GIF/video recorder
│   └── results/                 # JSON/CSV benchmark outputs
├── tests/                       # 288 tests across 8 test files
├── videos/                      # Demo GIFs
└── .github/workflows/ci.yml
```

## Hardware

- **Development:** Apple Silicon Mac (MuJoCo runs natively on macOS)
- **RL Training:** AWS g4dn.xlarge (T4 16GB) — SAC/PPO with batch sizes ~256
- **VLM Inference:** T4 with 4-bit quantization (PaliGemma-3B fits in ~4GB VRAM)

## Models

| Component | Model | Size | Quantization |
|-----------|-------|------|--------------|
| VLM Planner | PaliGemma-3B or Phi-3-Vision | 3B params | GPTQ 4-bit |
| RL Policy | Custom MLP (Actor-Critic) | ~200K params | fp32 |
| Scene Encoder | DINOv2-small | 22M params | fp16 |

## Benchmarks

_100 episodes per task/policy pair, seed=42, 200-step max episodes._

| Task | Random Baseline | Scripted Baseline | Notes |
|------|----------------|-------------------|-------|
| L1 — Pick & Place | 0.0% ± 1.8% | 0.0% ± 1.8% | Scripted gets 2.6× better returns |
| L2 — Color Pick | 0.0% ± 1.8% | — | Multi-object color selection |
| L3 — Stack | 0.0% ± 1.8% | — | Requires sequential precision |
| L4 — Sort | 0.0% ± 1.8% | — | N objects → N zones |
| L5 — Language | 4.0% ± 4.1% | — | Random meets some conditions by chance |
| Move To | 0.0% ± 1.8% | 20.0% ± 7.8% | P-controller reaches targets |

_Success rate = % of episodes where task completed within 200 steps. 95% Wilson CI._

### Primitive Comparison

| Environment | Policy | Mean Return | Mean Length |
|-------------|--------|-------------|-------------|
| L1 PickPlace | Random | -142.4 | 200.0 |
| L1 PickPlace | Scripted | -53.2 | 200.0 |
| MoveTo | Random | -715.8 | 200.0 |
| MoveTo | Scripted | -260.3 | 187.6 |

### VLM Planner Accuracy

| Metric | Result |
|--------|--------|
| Decomposition accuracy (MockVLM, 20+ scenarios) | 100% |
| Object grounding accuracy (SimGrounder, 20+ queries) | 100% |
| Color synonym coverage | 15+ aliases → 6 canonical |
| Shape synonym coverage | 9 aliases → 3 canonical |

### Demo Videos

| Task | Policy | Frames | Success |
|------|--------|--------|---------|
| L1 PickPlace | Scripted | 201 | ✗ |
| MoveTo | Scripted | 125 | ✓ |
| L4 Sort | Random | 201 | ✗ |

## References

- [SayCan: Do As I Can, Not As I Say (Ahn et al., 2022)](https://say-can.github.io/)
- [RT-2: Vision-Language-Action Models (Brohan et al., 2023)](https://robotics-transformer2.github.io/)
- [Code as Policies (Liang et al., 2023)](https://code-as-policies.github.io/)
- [MuJoCo: Multi-Joint dynamics with Contact](https://mujoco.org/)
- [PaliGemma (Google)](https://ai.google.dev/gemma/docs/paligemma)

## License

Apache 2.0
