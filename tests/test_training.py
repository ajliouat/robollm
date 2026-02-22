"""Tests for training scripts — quick sanity checks."""

import numpy as np
import pytest

from envs.pick_place import PickPlaceEnv
from policies.sac import SACAgent, SACConfig
from training.train_pick import evaluate, make_env


class TestTrainPick:
    """Training script components work correctly."""

    def test_make_env(self):
        env = make_env()
        obs, _ = env.reset(seed=0)
        assert obs.shape == (32,)
        env.close()

    def test_evaluate_returns_dict(self):
        env = PickPlaceEnv()
        config = SACConfig(warmup_steps=0)
        agent = SACAgent(
            obs_dim=env.observation_space.shape[0],
            act_dim=env.action_space.shape[0],
            config=config,
        )
        agent.total_steps = 99999  # skip warmup
        result = evaluate(env, agent, n_episodes=3)
        assert "mean_reward" in result
        assert "success_rate" in result
        assert result["n_episodes"] == 3
        env.close()

    def test_short_training_loop(self):
        """Run 100 steps to verify training loop doesn't crash."""
        from training.train_pick import train_pick
        results = train_pick(
            total_steps=200,
            seed=0,
            eval_interval=100,
            eval_episodes=2,
            log_interval=100,
            save_interval=10000,
            max_episode_steps=50,
        )
        assert "final_eval" in results
        assert "best_success_rate" in results
        assert isinstance(results["final_eval"]["success_rate"], float)
