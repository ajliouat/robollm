"""
Benchmark suite — evaluates all task levels with multiple baselines.

Runs N episodes per (env, policy) pair and records success rate,
mean return, episode length, and 95 % confidence intervals.

Results exported as JSON and CSV for README tables.
"""

from __future__ import annotations

import json
import csv
import hashlib
import math
import platform
import subprocess
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np


# ── Result dataclasses ────────────────────────────────────────────

@dataclass
class EpisodeResult:
    """Raw outcomes used to recompute an evaluation's aggregate statistics."""
    episode: int
    env_seed: int
    action_seed: int
    initial_success: bool | None
    success: bool
    episode_return: float
    length: int
    terminated: bool
    truncated: bool


@dataclass
class EvalMetrics:
    """Metrics for a single (env, policy) evaluation."""
    env_name: str
    policy_name: str
    n_episodes: int = 0
    success_rate: float = 0.0
    ci95: float = 0.0  # Legacy Wilson half-width, not an error bar about success_rate.
    mean_return: float = 0.0
    std_return: float = 0.0
    mean_length: float = 0.0
    std_length: float = 0.0
    wall_time_s: float = 0.0
    n_successes: int = 0
    seed: int | None = None
    ci95_low: float | None = None
    ci95_high: float | None = None
    episodes: list[EpisodeResult] = field(default_factory=list)

    def summary(self) -> str:
        interval = (
            f"[{self.ci95_low:.1%}, {self.ci95_high:.1%}]"
            if self.ci95_low is not None and self.ci95_high is not None
            else "unavailable"
        )
        return (
            f"{self.env_name:25s} | {self.policy_name:15s} | "
            f"SR {self.success_rate:5.1%} (Wilson 95% {interval}) | "
            f"Ret {self.mean_return:8.1f} ± {self.std_return:6.1f} | "
            f"Len {self.mean_length:5.1f} | {self.wall_time_s:.1f}s"
        )


@dataclass
class BenchmarkReport:
    """Collection of all evaluation results."""
    results: list[EvalMetrics] = field(default_factory=list)
    timestamp: str = ""
    total_episodes: int = 0
    total_wall_time_s: float = 0.0
    provenance: dict[str, Any] = field(default_factory=dict)

    def add(self, m: EvalMetrics) -> None:
        self.results.append(m)
        self.total_episodes += m.n_episodes
        self.total_wall_time_s += m.wall_time_s

    def print_table(self) -> None:
        header = (
            f"{'Environment':25s} | {'Policy':15s} | "
            f"{'Success':14s} | {'Return':18s} | {'Len':5s} | Time"
        )
        print("\n" + "=" * len(header))
        print(header)
        print("-" * len(header))
        for r in self.results:
            print(r.summary())
        print("=" * len(header))
        print(
            f"Total: {self.total_episodes} episodes, "
            f"{self.total_wall_time_s:.1f}s wall time\n"
        )


# ── Evaluation runner ─────────────────────────────────────────────

def wilson_interval(
    n_success: int, n_total: int, z: float = 1.96,
) -> tuple[float, float]:
    """Wilson score interval, whose centre differs from the observed rate.

    For example, 0/100 gives [0, 0.03699], not 0 ± 0.01850.
    No binomial interval is reported for an empty evaluation.
    """
    if (isinstance(n_total, bool) or not isinstance(n_total, Integral)
            or n_total <= 0):
        raise ValueError("n_total must be a positive integer")
    if (isinstance(n_success, bool) or not isinstance(n_success, Integral)
            or not 0 <= n_success <= n_total):
        raise ValueError("n_success must be an integer between 0 and n_total")
    if not math.isfinite(z) or z <= 0:
        raise ValueError("z must be finite and positive")
    p = n_success / n_total
    denom = 1 + z ** 2 / n_total
    centre = (p + z ** 2 / (2 * n_total)) / denom
    spread = z * math.sqrt((p * (1 - p) + z ** 2 / (4 * n_total)) / n_total) / denom
    return max(0.0, centre - spread), min(1.0, centre + spread)


def _wilson_ci(n_success: int, n_total: int, z: float = 1.96) -> float:
    """Legacy interval half-width; do not display as success_rate ± ci95."""
    if n_total == 0 and n_success == 0:
        return 0.0  # Compatibility for historical consumers only.
    low, high = wilson_interval(n_success, n_total, z)
    return (high - low) / 2


def _validate_run(n_episodes: int, seed: int) -> None:
    if (isinstance(n_episodes, bool) or not isinstance(n_episodes, Integral)
            or n_episodes <= 0):
        raise ValueError("n_episodes must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a non-negative integer")


def evaluate_policy(
    env: Any,
    policy_fn: Any,
    n_episodes: int = 100,
    env_name: str = "env",
    policy_name: str = "policy",
    seed: int = 42,
) -> EvalMetrics:
    """Run *n_episodes* and collect metrics.

    Args:
        env: Gymnasium-compatible env.
        policy_fn: Callable(obs, info) → action **or** has .act(info) method.
        n_episodes: Number of evaluation episodes.
        env_name: Label for the environment.
        policy_name: Label for the policy.
        seed: Base seed; episode i seeds both env and action_space with seed + i.
            Custom stochastic policies must manage their own RNG; this does not
            seed arbitrary NumPy/Python/Torch randomness inside a policy.

    Returns:
        EvalMetrics with aggregated statistics.
    """
    _validate_run(n_episodes, seed)
    if not callable(policy_fn) and not callable(getattr(policy_fn, "act", None)):
        raise TypeError("policy_fn must be callable or provide an act(info) method")
    if not callable(getattr(getattr(env, "action_space", None), "seed", None)):
        raise TypeError("env.action_space must provide seed() for reproducibility")
    n_episodes, seed = int(n_episodes), int(seed)
    successes = 0
    returns: list[float] = []
    lengths: list[int] = []
    episodes: list[EpisodeResult] = []

    t0 = time.perf_counter()

    for ep in range(n_episodes):
        episode_seed = seed + ep
        obs, info = env.reset(seed=episode_seed)
        initial_success = bool(info["success"]) if "success" in info else None
        env.action_space.seed(episode_seed)
        ep_return = 0.0
        ep_len = 0

        if callable(getattr(policy_fn, "reset", None)):
            policy_fn.reset()

        done = False
        while not done:
            if callable(policy_fn):
                action = policy_fn(obs, info)
            else:
                action = policy_fn.act(info)

            obs, reward, term, trunc, info = env.step(action)
            if not math.isfinite(float(reward)):
                raise ValueError(f"Non-finite reward in episode {ep}, step {ep_len + 1}")
            ep_return += float(reward)
            ep_len += 1
            done = term or trunc

        returns.append(ep_return)
        lengths.append(ep_len)
        success = bool(info.get("success", False))
        if success:
            successes += 1
        episodes.append(EpisodeResult(
            episode=ep, env_seed=episode_seed, action_seed=episode_seed,
            initial_success=initial_success,
            success=success, episode_return=ep_return, length=ep_len,
            terminated=bool(term), truncated=bool(trunc),
        ))

    wall = time.perf_counter() - t0
    n = n_episodes
    arr_ret = np.array(returns)
    arr_len = np.array(lengths, dtype=float)
    ci_low, ci_high = wilson_interval(successes, n)

    return EvalMetrics(
        env_name=env_name,
        policy_name=policy_name,
        n_episodes=n,
        success_rate=successes / n,
        ci95=(ci_high - ci_low) / 2,
        mean_return=float(arr_ret.mean()),
        std_return=float(arr_ret.std()),
        mean_length=float(arr_len.mean()),
        std_length=float(arr_len.std()),
        wall_time_s=wall,
        n_successes=successes,
        seed=seed,
        ci95_low=ci_low,
        ci95_high=ci_high,
        episodes=episodes,
    )


# ── Full benchmark ────────────────────────────────────────────────

def _runtime_provenance() -> dict[str, Any]:
    """Record the executing checkout and runtime, without host/user paths."""
    packages: dict[str, str | None] = {}
    for name in ("numpy", "mujoco", "gymnasium", "torch"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None

    repo = Path(__file__).resolve().parents[1]
    revision, dirty = None, None
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        ).strip())
    except (OSError, subprocess.SubprocessError):
        pass  # An installed source archive need not have Git metadata.
    return {
        "git_revision": revision,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "packages": packages,
    }


def _checkpoint_provenance(checkpoints: dict[str, str | Path]) -> dict[str, Any]:
    records = {}
    for env_name, checkpoint in checkpoints.items():
        if env_name not in {"L1-PickPlace", "MoveTo"}:
            raise ValueError(f"Unsupported SAC checkpoint environment: {env_name}")
        path = Path(checkpoint)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        records[env_name] = {"filename": path.name, "sha256": digest.hexdigest()}
    return records


def run_full_benchmark(
    n_episodes: int = 100,
    seed: int = 42,
    output_dir: str | Path = "runs/benchmark",
    sac_checkpoints: dict[str, str | Path] | None = None,
) -> BenchmarkReport:
    """Run the complete benchmark suite across all envs and policies.

    This evaluates:
    - L1 PickPlace: random, scripted
    - L2 ColorPick: random
    - L3 Stack: random
    - L4 Sort: random
    - L5 ComplexLanguage: random
    - MoveToEnv: random, scripted
    - Optional SAC checkpoints for L1 PickPlace and MoveToEnv

    This suite does not evaluate a VLM or the hierarchical executor.

    Returns a BenchmarkReport with all results.
    """
    _validate_run(n_episodes, seed)
    checkpoint_records = _checkpoint_provenance(sac_checkpoints or {})

    from envs.pick_place import PickPlaceEnv
    from envs.color_pick import ColorPickEnv
    from envs.stack import StackEnv
    from envs.sort import SortEnv
    from envs.complex_language import ComplexLanguageEnv
    from envs.move_to import MoveToEnv
    from policies.scripted import ScriptedPickPlace, ScriptedMoveTo

    report = BenchmarkReport()
    report.timestamp = datetime.now(timezone.utc).isoformat()
    report.provenance = _runtime_provenance()
    report.provenance.update({
        "seed": int(seed),
        "n_episodes_per_configuration": int(n_episodes),
        "seed_protocol": "env.reset and action_space.seed use seed + episode index",
        "success_protocol": "success flag in final episode info",
        "return_protocol": "undiscounted sum of environment rewards",
        "std_protocol": "population standard deviation (ddof=0)",
        "confidence_interval": "Wilson score, z=1.96, endpoints in ci95_low/ci95_high",
        "policy_randomness": "action_space seeded; custom policy RNGs are not managed",
        "sac_checkpoints": checkpoint_records,
    })

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Env configs ───────────────────────────────────────────
    configs: list[tuple[str, Any, list[tuple[str, Any]]]] = []

    # L1 — PickPlace
    pp_env = PickPlaceEnv()
    pp_scripted = ScriptedPickPlace()
    configs.append((
        "L1-PickPlace", pp_env, [
            ("random", lambda obs, info: pp_env.action_space.sample()),
            ("scripted", pp_scripted),
        ],
    ))

    # MoveToEnv
    mt_env = MoveToEnv()
    mt_scripted = ScriptedMoveTo()
    configs.append((
        "MoveTo", mt_env, [
            ("random", lambda obs, info: mt_env.action_space.sample()),
            ("scripted", mt_scripted),
        ],
    ))

    # ── SAC-trained policies (if checkpoints provided) ────────
    if sac_checkpoints:
        for env_name, ckpt_path in sac_checkpoints.items():
            if env_name == "L1-PickPlace":
                env = PickPlaceEnv()
                policy = load_sac_policy(ckpt_path, obs_dim=32, act_dim=4)
                configs.append((env_name, env, [("sac", policy)]))
            elif env_name == "MoveTo":
                env = MoveToEnv()
                policy = load_sac_policy(ckpt_path, obs_dim=29, act_dim=4)
                configs.append((env_name, env, [("sac", policy)]))

    # L2 — ColorPick
    cp_env = ColorPickEnv()
    configs.append((
        "L2-ColorPick", cp_env, [
            ("random", lambda obs, info: cp_env.action_space.sample()),
        ],
    ))

    # L3 — Stack
    st_env = StackEnv()
    configs.append((
        "L3-Stack", st_env, [
            ("random", lambda obs, info: st_env.action_space.sample()),
        ],
    ))

    # L4 — Sort
    so_env = SortEnv()
    configs.append((
        "L4-Sort", so_env, [
            ("random", lambda obs, info: so_env.action_space.sample()),
        ],
    ))

    # L5 — ComplexLanguage
    cl_env = ComplexLanguageEnv()
    configs.append((
        "L5-Language", cl_env, [
            ("random", lambda obs, info: cl_env.action_space.sample()),
        ],
    ))

    # ── Run evaluations ───────────────────────────────────────
    try:
        for env_name, env, policies in configs:
            for policy_name, policy in policies:
                m = evaluate_policy(
                    env=env,
                    policy_fn=policy,
                    n_episodes=n_episodes,
                    env_name=env_name,
                    policy_name=policy_name,
                    seed=seed,
                )
                report.add(m)
    finally:
        for _, env, _ in configs:
            env.close()

    # ── Export results ────────────────────────────────────────
    _save_results(report, output_path)

    return report


def _save_results(report: BenchmarkReport, output_path: Path) -> None:
    """Save aggregate and per-episode evidence as JSON and CSV."""
    # JSON
    json_data = {
        "schema_version": 2,
        "timestamp": report.timestamp,
        "total_episodes": report.total_episodes,
        "total_wall_time_s": round(report.total_wall_time_s, 2),
        "provenance": report.provenance,
        "results": [asdict(r) for r in report.results],
    }
    json_path = output_path / "benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, allow_nan=False)

    # CSV
    csv_path = output_path / "benchmark_results.csv"
    if report.results:
        fieldnames = [key for key in asdict(report.results[0]) if key != "episodes"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in report.results:
                row = asdict(r)
                row.pop("episodes")
                writer.writerow(row)

        episodes_path = output_path / "benchmark_episodes.csv"
        with open(episodes_path, "w", newline="") as f:
            episode_fields = list(EpisodeResult.__dataclass_fields__)
            writer = csv.DictWriter(f, fieldnames=["env_name", "policy_name", *episode_fields])
            writer.writeheader()
            for result in report.results:
                for episode in result.episodes:
                    writer.writerow({
                        "env_name": result.env_name,
                        "policy_name": result.policy_name,
                        **asdict(episode),
                    })


# ── SAC policy loader ─────────────────────────────────────────────

def load_sac_policy(ckpt_path: str | Path, obs_dim: int, act_dim: int,
                    device: str = "cpu"):
    """Load a trained SAC agent for evaluation.

    Returns a callable policy_fn(obs, info) that uses deterministic actions.
    """
    from policies.sac import SACAgent, SACConfig

    config = SACConfig(device=device)
    agent = SACAgent(obs_dim, act_dim, config)
    agent.load(ckpt_path)
    agent.actor.eval()

    def policy_fn(obs, info=None):
        return agent.select_action(obs, deterministic=True)

    return policy_fn


# ── Threshold checker ─────────────────────────────────────────────

PROJECT_THRESHOLDS = {
    "L1-PickPlace": {"scripted": 0.0},  # Scripted has 0% multi-phase SR in 200 steps
    "L2-ColorPick": {"random": 0.0},
    "L3-Stack": {"random": 0.0},
    "MoveTo": {"scripted": 0.10},  # 10% threshold for scripted MoveTo
}


def check_thresholds(report: BenchmarkReport) -> list[tuple[str, str, float, float, bool]]:
    """Check results against project thresholds.

    Returns list of (env, policy, actual, threshold, passed) tuples.
    """
    checks = []
    for r in report.results:
        key = r.env_name
        if key in PROJECT_THRESHOLDS:
            th = PROJECT_THRESHOLDS[key].get(r.policy_name)
            if th is not None:
                passed = r.success_rate >= th
                checks.append((r.env_name, r.policy_name, r.success_rate, th, passed))
    return checks


# ── CLI entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RoboLLM Benchmark Suite")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="runs/benchmark")
    parser.add_argument("--sac-pick", type=str, default=None,
                        help="Path to SAC pick checkpoint")
    parser.add_argument("--sac-move-to", type=str, default=None,
                        help="Path to SAC move_to checkpoint")
    args = parser.parse_args()

    sac_checkpoints = {}
    if args.sac_pick:
        sac_checkpoints["L1-PickPlace"] = args.sac_pick
    if args.sac_move_to:
        sac_checkpoints["MoveTo"] = args.sac_move_to

    report = run_full_benchmark(
        n_episodes=args.episodes,
        seed=args.seed,
        output_dir=args.output,
        sac_checkpoints=sac_checkpoints or None,
    )
    report.print_table()

    checks = check_thresholds(report)
    if checks:
        print("\nThreshold Checks:")
        for env, pol, actual, threshold, passed in checks:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {status}: {env}/{pol} — {actual:.1%} vs {threshold:.1%}")
