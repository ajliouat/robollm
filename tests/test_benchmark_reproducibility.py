"""Regression tests for truthful intervals and replayable CPU evaluations."""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict
from statistics import mean, pstdev

import numpy as np
import pytest

from evaluation.benchmark import (
    BenchmarkReport, _save_results, evaluate_policy, wilson_interval,
)


class SeededActionSpace:
    def __init__(self):
        self.rng = np.random.default_rng()

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def sample(self):
        return float(self.rng.uniform(-1, 1))


class TinyEnv:
    """Three-step environment with separately seeded dynamics and actions."""
    def __init__(self):
        self.action_space = SeededActionSpace()
        self.actions = []
        self.reset_seeds = []

    def reset(self, *, seed):
        self.reset_seeds.append(seed)
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.success = seed % 2 == 0
        return np.zeros(1), {}

    def step(self, action):
        self.actions.append(action)
        self.steps += 1
        done = self.steps == 3
        return (
            np.zeros(1), float(action + self.rng.normal()),
            done and self.success, done and not self.success,
            {"success": done and self.success},
        )


@pytest.mark.parametrize("successes,total,expected", [
    (0, 100, (0.0, 0.03699480747600191)),
    (100, 100, (0.9630051925239981, 1.0)),
    (20, 100, (0.1333659225590988, 0.28883096192650335)),
    (4, 100, (0.0156630505518485, 0.0983721723260736)),
    (50, 100, (0.40382982859014716, 0.5961701714098528)),
])
def test_wilson_known_intervals(successes, total, expected):
    assert wilson_interval(successes, total) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("successes,total", [
    (0, 0), (-1, 5), (6, 5), (1.5, 5), (1, 2.5), (True, 5), (1, True),
])
def test_wilson_rejects_invalid_counts(successes, total):
    with pytest.raises(ValueError):
        wilson_interval(successes, total)


def test_action_randomness_replays_even_after_rng_has_advanced():
    env = TinyEnv()
    policy = lambda obs, info: env.action_space.sample()
    first = evaluate_policy(env, policy, n_episodes=4, seed=42)
    first_actions = env.actions.copy()
    env.actions.clear()
    # Perturb only action RNG, which env.reset(seed=...) does not affect.
    for _ in range(17):
        env.action_space.sample()
    second = evaluate_policy(env, policy, n_episodes=4, seed=42)
    assert env.actions == first_actions
    assert first.episodes == second.episodes
    assert first.mean_return == second.mean_return
    assert env.reset_seeds == [42, 43, 44, 45] * 2

    third = evaluate_policy(env, policy, n_episodes=4, seed=43)
    assert first.episodes != third.episodes


def test_episode_evidence_reconstructs_aggregates_and_end_conditions():
    env = TinyEnv()
    metrics = evaluate_policy(env, lambda obs, info: env.action_space.sample(),
                              n_episodes=5, seed=42)
    episodes = metrics.episodes
    returns = [episode.episode_return for episode in episodes]
    assert metrics.n_successes == sum(episode.success for episode in episodes) == 3
    assert metrics.success_rate == 3 / 5
    assert metrics.mean_return == pytest.approx(mean(returns))
    assert metrics.std_return == pytest.approx(pstdev(returns))
    assert metrics.mean_length == 3
    assert metrics.std_length == 0
    assert [episode.env_seed for episode in episodes] == list(range(42, 47))
    assert all(episode.action_seed == episode.env_seed for episode in episodes)
    assert all(episode.terminated == episode.success for episode in episodes)
    assert all(episode.truncated != episode.terminated for episode in episodes)


def test_summary_uses_endpoints_instead_of_halfwidth_about_observed_rate():
    env = TinyEnv()
    result = evaluate_policy(env, lambda obs, info: 0, n_episodes=1, seed=1)
    assert "SR  0.0% (Wilson 95% [0.0%, 79.3%])" in result.summary()


def test_preexisting_success_is_recorded_separately_from_final_success():
    env = TinyEnv()
    reset = env.reset

    def already_satisfied(*, seed):
        obs, info = reset(seed=seed)
        return obs, {**info, "success": True}

    env.reset = already_satisfied
    metrics = evaluate_policy(env, lambda obs, info: 0, n_episodes=2, seed=42)
    assert all(episode.initial_success for episode in metrics.episodes)
    assert metrics.episodes[0].success is True
    assert metrics.episodes[1].success is False


@pytest.mark.parametrize("episodes", [0, -1, 2.5, True])
def test_empty_or_invalid_evaluation_fails_before_touching_env(episodes):
    env = TinyEnv()
    with pytest.raises(ValueError, match="n_episodes"):
        evaluate_policy(env, lambda obs, info: 0, n_episodes=episodes)
    assert not env.reset_seeds


@pytest.mark.parametrize("seed", [-1, 1.5, True])
def test_invalid_seed_fails_before_touching_env(seed):
    env = TinyEnv()
    with pytest.raises(ValueError, match="seed"):
        evaluate_policy(env, lambda obs, info: 0, seed=seed)
    assert not env.reset_seeds


def test_invalid_policy_does_not_silently_evaluate_random_policy():
    env = TinyEnv()
    with pytest.raises(TypeError, match="policy_fn"):
        evaluate_policy(env, object(), n_episodes=1)
    assert not env.reset_seeds


@pytest.mark.parametrize("bad_reward", [math.nan, math.inf, -math.inf])
def test_nonfinite_reward_cannot_be_exported_as_measured_result(bad_reward):
    env = TinyEnv()
    env.step = lambda action: (np.zeros(1), bad_reward, True, False, {})
    with pytest.raises(ValueError, match="Non-finite reward"):
        evaluate_policy(env, lambda obs, info: 0, n_episodes=1)


def test_exports_keep_aggregate_and_episode_evidence_consistent(tmp_path):
    env = TinyEnv()
    metrics = evaluate_policy(env, lambda obs, info: env.action_space.sample(),
                              n_episodes=4, seed=42)
    report = BenchmarkReport(provenance={"test_fixture": True})
    report.add(metrics)
    _save_results(report, tmp_path)

    saved = json.loads((tmp_path / "benchmark_results.json").read_text())
    assert saved["schema_version"] == 2
    assert saved["provenance"] == {"test_fixture": True}
    assert saved["results"] == [asdict(metrics)]
    with (tmp_path / "benchmark_results.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert "episodes" not in rows[0]
    assert int(rows[0]["n_successes"]) == 2
    assert float(rows[0]["ci95_low"]) == metrics.ci95_low
    with (tmp_path / "benchmark_episodes.csv").open() as f:
        raw_rows = list(csv.DictReader(f))
    assert len(raw_rows) == 4
    assert [int(row["env_seed"]) for row in raw_rows] == [42, 43, 44, 45]
    assert sum(row["success"] == "True" for row in raw_rows) == 2


def test_random_mujoco_rollouts_reproduce_without_rendering():
    from envs.move_to import MoveToEnv

    env = MoveToEnv()
    try:
        policy = lambda obs, info: env.action_space.sample()
        first = evaluate_policy(env, policy, n_episodes=2, seed=42)
        second = evaluate_policy(env, policy, n_episodes=2, seed=42)
        assert first.episodes == second.episodes
        assert all(episode.length == 200 for episode in first.episodes)
    finally:
        env.close()
