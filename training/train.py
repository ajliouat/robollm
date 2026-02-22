"""
Training loop for SAC.

Runs the agent in an environment, stores transitions, and updates.
Logs to TensorBoard and prints periodic summaries.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from policies.sac import SACAgent, SACConfig


@dataclass
class TrainConfig:
    """Training loop configuration."""

    total_steps: int = 500_000
    eval_interval: int = 10_000
    eval_episodes: int = 10
    log_interval: int = 1_000
    save_interval: int = 50_000
    max_episode_steps: int = 200
    checkpoint_dir: str = "checkpoints"


def train(
    env_fn,
    *,
    sac_config: SACConfig | None = None,
    train_config: TrainConfig | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Run SAC training loop.

    Parameters
    ----------
    env_fn : callable
        Factory that returns a Gymnasium environment.
    sac_config : SACConfig
        SAC hyperparameters.
    train_config : TrainConfig
        Training loop settings.
    seed : int
        Random seed.

    Returns
    -------
    dict with training history.
    """
    sac_cfg = sac_config or SACConfig()
    train_cfg = train_config or TrainConfig()

    env = env_fn()
    eval_env = env_fn()

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = SACAgent(obs_dim, act_dim, sac_cfg)

    ckpt_dir = Path(train_cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ────────────────────────────────────────────────────
    history: dict[str, list] = {
        "episode_rewards": [],
        "episode_lengths": [],
        "eval_rewards": [],
        "eval_steps": [],
        "metrics": [],
    }

    obs, _ = env.reset(seed=seed)
    episode_reward = 0.0
    episode_len = 0
    episode_count = 0
    t_start = time.time()

    for step in range(1, train_cfg.total_steps + 1):
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_len += 1

        # Truncation: don't bootstrap on time limit
        real_done = terminated and not truncated
        agent.store_transition(obs, action, reward, next_obs, real_done)

        obs = next_obs
        episode_reward += reward

        # Update
        if step > sac_cfg.warmup_steps:
            metrics = agent.update()
        else:
            metrics = {}

        # Episode done
        if done or episode_len >= train_cfg.max_episode_steps:
            history["episode_rewards"].append(episode_reward)
            history["episode_lengths"].append(episode_len)
            episode_count += 1
            obs, _ = env.reset()
            episode_reward = 0.0
            episode_len = 0

        # Logging
        if step % train_cfg.log_interval == 0 and metrics:
            elapsed = time.time() - t_start
            fps = step / max(elapsed, 1e-6)
            recent = history["episode_rewards"][-10:]
            avg_r = np.mean(recent) if recent else 0.0
            print(
                f"Step {step:>8d} | Ep {episode_count:>4d} | "
                f"R(10) {avg_r:>8.2f} | α {metrics.get('alpha', 0):.3f} | "
                f"Q1 {metrics.get('q1_mean', 0):.2f} | "
                f"FPS {fps:.0f}"
            )

        # Evaluation
        if step % train_cfg.eval_interval == 0:
            eval_r = _evaluate(eval_env, agent, train_cfg.eval_episodes)
            history["eval_rewards"].append(eval_r)
            history["eval_steps"].append(step)
            print(f"  ── Eval @ {step}: mean reward = {eval_r:.2f}")

        # Checkpoint
        if step % train_cfg.save_interval == 0:
            agent.save(ckpt_dir / f"sac_step_{step}.pt")

    # Final save
    agent.save(ckpt_dir / "sac_final.pt")
    env.close()
    eval_env.close()

    return history


def _evaluate(env, agent: SACAgent, n_episodes: int) -> float:
    """Run n deterministic episodes, return mean reward."""
    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=10000 + ep)
        total = 0.0
        for _ in range(200):
            action = agent.select_action(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(action)
            total += r
            if terminated or truncated:
                break
        rewards.append(total)
    return float(np.mean(rewards))
