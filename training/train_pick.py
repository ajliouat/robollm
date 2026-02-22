"""
Train pick primitive on PickPlaceEnv.

Runs SAC on the L1 pick-and-place task. The agent learns to approach,
grasp, lift, and place an object at a goal position.

Usage:
    python -m training.train_pick [--steps 500000] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from envs.pick_place import PickPlaceEnv
from policies.sac import SACAgent, SACConfig


def make_env():
    return PickPlaceEnv()


def evaluate(env, agent: SACAgent, n_episodes: int = 100, seed_offset: int = 10000):
    """Run deterministic evaluation. Returns dict with stats."""
    rewards = []
    successes = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        total_r = 0.0
        success = False
        for _ in range(200):
            action = agent.select_action(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            total_r += r
            if info.get("success", False):
                success = True
            if terminated or truncated:
                break
        rewards.append(total_r)
        successes.append(float(success))

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "success_rate": float(np.mean(successes)),
        "n_episodes": n_episodes,
    }


def train_pick(
    total_steps: int = 500_000,
    seed: int = 0,
    eval_interval: int = 25_000,
    log_interval: int = 5_000,
    save_interval: int = 50_000,
    eval_episodes: int = 50,
    max_episode_steps: int = 200,
):
    """Train the pick primitive and return results."""

    env = make_env()
    eval_env = make_env()

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    config = SACConfig(
        buffer_size=500_000,
        batch_size=256,
        warmup_steps=5_000,
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        gamma=0.99,
        tau=0.005,
        hidden_dim=256,
    )

    agent = SACAgent(obs_dim, act_dim, config)

    ckpt_dir = Path("checkpoints/pick")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path("evaluation/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ────────────────────────────────────────────────────
    history = {
        "episode_rewards": [],
        "eval_results": [],
        "eval_steps": [],
    }

    obs, _ = env.reset(seed=seed)
    episode_reward = 0.0
    episode_len = 0
    episode_count = 0
    best_success = 0.0
    t_start = time.time()

    for step in range(1, total_steps + 1):
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_len += 1

        real_done = terminated and not truncated
        agent.store_transition(obs, action, reward, next_obs, real_done)

        obs = next_obs
        episode_reward += reward

        # Update
        if step > config.warmup_steps:
            agent.update()

        # Episode done
        if done or episode_len >= max_episode_steps:
            history["episode_rewards"].append(episode_reward)
            episode_count += 1
            obs, _ = env.reset()
            episode_reward = 0.0
            episode_len = 0

        # Log
        if step % log_interval == 0:
            elapsed = time.time() - t_start
            fps = step / max(elapsed, 1e-6)
            recent = history["episode_rewards"][-20:]
            avg_r = np.mean(recent) if recent else 0.0
            print(
                f"Step {step:>8d}/{total_steps} | Ep {episode_count:>4d} | "
                f"R(20) {avg_r:>8.2f} | α {agent.alpha:.3f} | "
                f"FPS {fps:.0f}"
            )

        # Evaluate
        if step % eval_interval == 0:
            eval_result = evaluate(eval_env, agent, eval_episodes)
            history["eval_results"].append(eval_result)
            history["eval_steps"].append(step)
            sr = eval_result["success_rate"]
            mr = eval_result["mean_reward"]
            print(
                f"  ── Eval @ {step}: success={sr:.1%}, "
                f"reward={mr:.2f} ± {eval_result['std_reward']:.2f}"
            )

            if sr > best_success:
                best_success = sr
                agent.save(ckpt_dir / "pick_best.pt")
                print(f"     New best! success_rate={sr:.1%}")

        # Checkpoint
        if step % save_interval == 0:
            agent.save(ckpt_dir / f"pick_step_{step}.pt")

    # ── Final evaluation ─────────────────────────────────────────────────
    agent.save(ckpt_dir / "pick_final.pt")
    final_eval = evaluate(eval_env, agent, 100)
    print(f"\n{'='*60}")
    print(f"Final: success={final_eval['success_rate']:.1%}, "
          f"reward={final_eval['mean_reward']:.2f}")
    print(f"Best:  success={best_success:.1%}")
    print(f"{'='*60}")

    # Save results
    results = {
        "final_eval": final_eval,
        "best_success_rate": best_success,
        "total_steps": total_steps,
        "seed": seed,
        "eval_history": [
            {"step": s, **r}
            for s, r in zip(history["eval_steps"], history["eval_results"])
        ],
    }
    with open(results_dir / "pick_training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    env.close()
    eval_env.close()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train pick primitive")
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-interval", type=int, default=25_000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    args = parser.parse_args()

    train_pick(
        total_steps=args.steps,
        seed=args.seed,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
    )
