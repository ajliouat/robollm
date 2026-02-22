"""
Evaluate all primitives: RL vs scripted vs random baselines.

Runs 100 episodes per (env, policy) pair and reports success rate,
mean return, and mean episode length.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from envs.move_to import MoveToEnv
from envs.pick_place import PickPlaceEnv
from envs.place import PlaceEnv
from policies.scripted import ScriptedPickPlace, ScriptedMoveTo


def evaluate_random(env, n_episodes: int = 100, seed: int = 0):
    """Evaluate a random policy."""
    returns, successes, lengths = [], [], []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        total_reward = 0.0
        done = False
        steps = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            steps += 1
            done = term or trunc
        returns.append(total_reward)
        successes.append(info.get("success", False))
        lengths.append(steps)
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "success_rate": float(np.mean(successes)),
        "mean_length": float(np.mean(lengths)),
    }


def evaluate_scripted(env, policy, n_episodes: int = 100, seed: int = 0):
    """Evaluate a scripted policy using env info dict."""
    returns, successes, lengths = [], [], []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        policy.reset()
        total_reward = 0.0
        done = False
        steps = 0
        while not done:
            action = policy.act(info)
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            steps += 1
            done = term or trunc
        returns.append(total_reward)
        successes.append(info.get("success", False))
        lengths.append(steps)
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "success_rate": float(np.mean(successes)),
        "mean_length": float(np.mean(lengths)),
    }


def run_comparison(n_episodes: int = 100, seed: int = 0):
    """Run full comparison across all primitives."""
    results = {}

    # ── MoveToEnv ──
    print("=== MoveToEnv ===")
    env = MoveToEnv()

    print("  Random...")
    results["move_to_random"] = evaluate_random(env, n_episodes, seed)
    print(f"    SR={results['move_to_random']['success_rate']:.2%}  "
          f"R={results['move_to_random']['mean_return']:.1f}")

    print("  Scripted...")
    results["move_to_scripted"] = evaluate_scripted(
        env, ScriptedMoveTo(), n_episodes, seed)
    print(f"    SR={results['move_to_scripted']['success_rate']:.2%}  "
          f"R={results['move_to_scripted']['mean_return']:.1f}")
    env.close()

    # ── PickPlaceEnv ──
    print("=== PickPlaceEnv ===")
    env = PickPlaceEnv()

    print("  Random...")
    results["pick_place_random"] = evaluate_random(env, n_episodes, seed)
    print(f"    SR={results['pick_place_random']['success_rate']:.2%}  "
          f"R={results['pick_place_random']['mean_return']:.1f}")

    print("  Scripted...")
    results["pick_place_scripted"] = evaluate_scripted(
        env, ScriptedPickPlace(), n_episodes, seed)
    print(f"    SR={results['pick_place_scripted']['success_rate']:.2%}  "
          f"R={results['pick_place_scripted']['mean_return']:.1f}")
    env.close()

    # ── PlaceEnv ──
    print("=== PlaceEnv ===")
    env = PlaceEnv()

    print("  Random...")
    results["place_random"] = evaluate_random(env, n_episodes, seed)
    print(f"    SR={results['place_random']['success_rate']:.2%}  "
          f"R={results['place_random']['mean_return']:.1f}")

    print("  Scripted (MoveTo as approx)...")
    results["place_scripted"] = evaluate_scripted(
        env, ScriptedMoveTo(), n_episodes, seed)
    print(f"    SR={results['place_scripted']['success_rate']:.2%}  "
          f"R={results['place_scripted']['mean_return']:.1f}")
    env.close()

    return results


def main():
    t0 = time.time()
    results = run_comparison(n_episodes=25, seed=0)
    elapsed = time.time() - t0

    results["_meta"] = {
        "n_episodes": 25,
        "seed": 0,
        "elapsed_sec": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    out_dir = Path("evaluation/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "primitive_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}  ({elapsed:.1f}s)")

    # Print comparison table
    print("\n" + "=" * 70)
    print(f"{'Environment':<18} {'Policy':<12} {'Success%':>10} "
          f"{'Mean Return':>12} {'Mean Len':>10}")
    print("-" * 70)
    for key, res in results.items():
        if key.startswith("_"):
            continue
        parts = key.rsplit("_", 1)
        env_name = parts[0].replace("_", " ").title()
        policy_name = parts[1].title()
        print(f"{env_name:<18} {policy_name:<12} "
              f"{res['success_rate']:>9.1%} "
              f"{res['mean_return']:>12.1f} "
              f"{res['mean_length']:>10.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
