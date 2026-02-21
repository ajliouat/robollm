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
├── LICENSE
├── pyproject.toml
├── Dockerfile
├── envs/                        # MuJoCo environments
│   ├── tabletop.py              # Base tabletop environment
│   ├── pick_place.py            # L1: single pick-place
│   ├── color_pick.py            # L2: color-conditioned pick
│   ├── stack.py                 # L3: stacking
│   ├── sort.py                  # L4: sorting
│   └── assets/                  # MuJoCo XML models
│       ├── robot_arm.xml
│       ├── table.xml
│       └── objects.xml
├── planner/                     # VLM-based task decomposition
│   ├── vlm_planner.py           # PaliGemma / Phi-3-Vision wrapper
│   ├── task_parser.py           # Parse VLM output → sub-task sequence
│   └── prompts/                 # Prompt templates
│       └── decompose.txt
├── policies/                    # RL policy training
│   ├── sac.py                   # SAC implementation
│   ├── ppo.py                   # PPO implementation
│   ├── networks.py              # Actor-Critic architectures
│   └── replay_buffer.py
├── training/
│   ├── train_policy.py          # Main RL training script
│   ├── train_config.yaml        # Hyperparameters
│   └── curriculum.py            # Task complexity curriculum
├── evaluation/
│   ├── evaluate.py              # Benchmark task success rates
│   ├── record_video.py          # Save demo videos/GIFs
│   └── results/
│       └── .gitkeep
├── tests/
│   ├── test_envs.py
│   ├── test_planner.py
│   └── test_policies.py
├── notebooks/
│   └── demo.ipynb               # Interactive demo notebook
├── videos/                      # Demo GIFs for README
│   └── .gitkeep
└── .github/
    └── workflows/
        └── ci.yml
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

_To be populated with real results. Format:_

| Task | Scripted Baseline | LLM-Guided (ours) | Improvement |
|------|------------------|--------------------|-------------|
| L1 — Pick & Place | —% | —% | — |
| L2 — Color Pick | —% | —% | — |
| L3 — Stack | —% | —% | — |
| L4 — Sort | —% | —% | — |
| L5 — Language | N/A | —% | — |

_Success rate = % of episodes where task completed within 200 steps. 100 eval episodes per task._

## References

- [SayCan: Do As I Can, Not As I Say (Ahn et al., 2022)](https://say-can.github.io/)
- [RT-2: Vision-Language-Action Models (Brohan et al., 2023)](https://robotics-transformer2.github.io/)
- [Code as Policies (Liang et al., 2023)](https://code-as-policies.github.io/)
- [MuJoCo: Multi-Joint dynamics with Contact](https://mujoco.org/)
- [PaliGemma (Google)](https://ai.google.dev/gemma/docs/paligemma)

## License

Apache 2.0
