"""
Train all RoboLLM primitives via SAC and run benchmark.

Usage (local, CPU):
    python -m training.train_all --primitives pick --steps 10000

Usage (AWS T4, full):
    python -m training.train_all --primitives all --steps 500000 --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from policies.sac import SACAgent, SACConfig
from training.train import TrainConfig, train


# ── Primitive registry ─────────────────────────────────────────────

PRIMITIVES: dict[str, dict] = {}


def register(name: str, env_fn, obs_dim: int, act_dim: int, **kwargs):
    PRIMITIVES[name] = {
        "env_fn": env_fn,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        **kwargs,
    }


register(
    "pick",
    lambda: __import__("envs.pick_place", fromlist=["PickPlaceEnv"]).PickPlaceEnv(),
    obs_dim=32, act_dim=4,
)
register(
    "move_to",
    lambda: __import__("envs.move_to", fromlist=["MoveToEnv"]).MoveToEnv(),
    obs_dim=29, act_dim=4,
)
register(
    "place",
    lambda: __import__("envs.place", fromlist=["PlaceEnv"]).PlaceEnv(),
    obs_dim=32, act_dim=4,
)
register(
    "color_pick",
    lambda: __import__("envs.color_pick", fromlist=["ColorPickEnv"]).ColorPickEnv(),
    obs_dim=35, act_dim=4,
)


def train_primitive(
    name: str,
    total_steps: int = 500_000,
    seed: int = 0,
    device: str = "cpu",
    log_dir: str = "checkpoints",
) -> dict:
    info = PRIMITIVES[name]
    env = info["env_fn"]()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    sac_cfg = SACConfig(
        buffer_size=500_000,
        batch_size=256,
        warmup_steps=5_000,
        gamma=0.99,
        tau=0.005,
        hidden_dim=256,
        device=device,
    )

    agent = SACAgent(obs_dim, act_dim, sac_cfg)

    ckpt_dir = Path(log_dir) / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path("evaluation/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=str(ckpt_dir / "tensorboard"))
    except ImportError:
        tb_writer = None

    # ── Training loop ────────────────────────────────────────────
    history = {"episode_rewards": [], "eval_results": [], "eval_steps": []}

    obs, _ = env.reset(seed=seed)
    episode_reward = 0.0
    episode_len = 0
    episode_count = 0
    best_success = 0.0
    t_start = time.time()

    eval_interval = 25_000 if total_steps >= 100_000 else max(total_steps // 10, 100)
    log_interval = max(total_steps // 100, 100)

    for step in range(1, total_steps + 1):
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_len += 1

        real_done = terminated and not truncated
        agent.store_transition(obs, action, reward, next_obs, real_done)

        obs = next_obs
        episode_reward += reward

        if step > sac_cfg.warmup_steps:
            metrics = agent.update()
        else:
            metrics = {}

        if done or episode_len >= 200:
            history["episode_rewards"].append(episode_reward)
            episode_count += 1
            obs, _ = env.reset()
            episode_reward = 0.0
            episode_len = 0

        if step % log_interval == 0 and step > 0:
            elapsed = time.time() - t_start
            fps = step / max(elapsed, 1e-6)
            recent = history["episode_rewards"][-20:]
            avg_r = np.mean(recent) if recent else 0.0
            alpha = metrics.get("alpha", 0.0) if metrics else agent.alpha
            print(
                f"[{name}] Step {step:>8d}/{total_steps} | "
                f"Ep {episode_count:>4d} | R(20) {avg_r:>8.2f} | "
                f"\u03b1 {alpha:.3f} | FPS {fps:.0f}"
            )
            if tb_writer:
                tb_writer.add_scalar("train/avg_reward_20", avg_r, step)
                tb_writer.add_scalar("train/alpha", float(alpha), step)
                tb_writer.add_scalar("train/fps", fps, step)

        if step % eval_interval == 0 and step > 0:
            eval_result = _evaluate_primitive(env, agent, n_episodes=50)
            history["eval_results"].append(eval_result)
            history["eval_steps"].append(step)
            sr = eval_result["success_rate"]
            mr = eval_result["mean_reward"]
            print(
                f"  \u2514\u2500 Eval @ {step}: success={sr:.1%}, "
                f"reward={mr:.2f}"
            )
            if tb_writer:
                tb_writer.add_scalar("eval/success_rate", sr, step)
                tb_writer.add_scalar("eval/mean_reward", mr, step)

            if sr > best_success:
                best_success = sr
                agent.save(ckpt_dir / "best.pt")
                print(f"     New best! success_rate={sr:.1%}")

    # ── Final ────────────────────────────────────────────────────
    agent.save(ckpt_dir / "final.pt")
    final_eval = _evaluate_primitive(env, agent, n_episodes=100)

    print(f"\n[{name}] Final: success={final_eval['success_rate']:.1%}, "
          f"reward={final_eval['mean_reward']:.2f}")
    print(f"[{name}] Best:  success={best_success:.1%}")

    results = {
        "primitive": name,
        "final_eval": final_eval,
        "best_success_rate": best_success,
        "total_steps": total_steps,
        "seed": seed,
        "device": device,
        "eval_history": [
            {"step": s, **r}
            for s, r in zip(history["eval_steps"], history["eval_results"])
        ],
    }
    with open(results_dir / f"{name}_training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    if tb_writer:
        tb_writer.close()
    env.close()
    return results


def _evaluate_primitive(env, agent: SACAgent, n_episodes: int = 100,
                        seed_offset: int = 10000) -> dict:
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


# ── CLI ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train RoboLLM primitives with SAC"
    )
    parser.add_argument("--primitives", type=str, default="pick",
                        help="Comma-separated list or 'all'")
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    if args.primitives == "all":
        primitives = list(PRIMITIVES.keys())
    else:
        primitives = [p.strip() for p in args.primitives.split(",")]

    print(f"Training primitives: {primitives}")
    print(f"Total steps each: {args.steps}")
    print(f"Device: {args.device}\n")

    all_results = {}
    for name in primitives:
        if name not in PRIMITIVES:
            print(f"Unknown primitive '{name}', skipping.")
            continue
        print(f"{'='*60}")
        print(f"Training: {name}")
        print(f"{'='*60}")
        result = train_primitive(
            name=name,
            total_steps=args.steps,
            seed=args.seed,
            device=args.device,
            log_dir=args.log_dir,
        )
        all_results[name] = result["best_success_rate"]
        print()

    print(f"{'='*60}")
    print("Training complete!")
    for name, sr in all_results.items():
        print(f"  {name}: best success rate = {sr:.1%}")


if __name__ == "__main__":
    main()
