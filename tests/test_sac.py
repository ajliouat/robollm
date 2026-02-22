"""Tests for SAC networks, replay buffer, and agent."""

import numpy as np
import pytest
import torch

from policies.networks import Actor, Critic, TwinCritic
from policies.sac import SACAgent, SACConfig
from training.replay_buffer import ReplayBuffer


# ═══════════════════════════════════════════════════════════════════════════
#  Networks
# ═══════════════════════════════════════════════════════════════════════════


class TestActor:
    """Actor network produces correct outputs."""

    def test_forward_shapes(self):
        actor = Actor(obs_dim=29, act_dim=4)
        obs = torch.randn(8, 29)
        mean, log_std = actor(obs)
        assert mean.shape == (8, 4)
        assert log_std.shape == (8, 4)

    def test_sample_shapes(self):
        actor = Actor(obs_dim=29, act_dim=4)
        obs = torch.randn(8, 29)
        action, log_prob, mean = actor.sample(obs)
        assert action.shape == (8, 4)
        assert log_prob.shape == (8, 1)
        assert mean.shape == (8, 4)

    def test_actions_bounded(self):
        actor = Actor(obs_dim=29, act_dim=4)
        obs = torch.randn(32, 29)
        action, _, _ = actor.sample(obs)
        assert (action >= -1.0).all()
        assert (action <= 1.0).all()

    def test_deterministic_is_bounded(self):
        actor = Actor(obs_dim=29, act_dim=4)
        obs = torch.randn(8, 29)
        action = actor.deterministic(obs)
        assert (action >= -1.0).all()
        assert (action <= 1.0).all()

    def test_log_std_clamped(self):
        actor = Actor(obs_dim=10, act_dim=2)
        obs = torch.randn(4, 10) * 100  # large input
        _, log_std = actor(obs)
        assert (log_std >= -20.0).all()
        assert (log_std <= 2.0).all()


class TestCritic:
    """Single critic network."""

    def test_forward_shape(self):
        critic = Critic(obs_dim=29, act_dim=4)
        obs = torch.randn(8, 29)
        action = torch.randn(8, 4)
        q = critic(obs, action)
        assert q.shape == (8, 1)


class TestTwinCritic:
    """Twin Q-networks."""

    def test_forward_shapes(self):
        twin = TwinCritic(obs_dim=29, act_dim=4)
        obs = torch.randn(8, 29)
        action = torch.randn(8, 4)
        q1, q2 = twin(obs, action)
        assert q1.shape == (8, 1)
        assert q2.shape == (8, 1)

    def test_q1_forward(self):
        twin = TwinCritic(obs_dim=29, act_dim=4)
        obs = torch.randn(8, 29)
        action = torch.randn(8, 4)
        q1_only = twin.q1_forward(obs, action)
        assert q1_only.shape == (8, 1)

    def test_twin_q_different(self):
        """Q1 and Q2 should generally differ (different random init)."""
        twin = TwinCritic(obs_dim=29, act_dim=4)
        obs = torch.randn(8, 29)
        action = torch.randn(8, 4)
        q1, q2 = twin(obs, action)
        assert not torch.allclose(q1, q2, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════════
#  Replay Buffer
# ═══════════════════════════════════════════════════════════════════════════


class TestReplayBuffer:
    """Replay buffer stores and samples correctly."""

    def test_empty_buffer(self):
        buf = ReplayBuffer(100, obs_dim=4, act_dim=2)
        assert len(buf) == 0

    def test_add_single(self):
        buf = ReplayBuffer(100, obs_dim=4, act_dim=2)
        buf.add(np.zeros(4), np.zeros(2), 1.0, np.zeros(4), False)
        assert len(buf) == 1

    def test_capacity_respected(self):
        buf = ReplayBuffer(10, obs_dim=4, act_dim=2)
        for i in range(20):
            buf.add(np.ones(4) * i, np.ones(2), float(i), np.ones(4), False)
        assert len(buf) == 10

    def test_sample_shapes(self):
        buf = ReplayBuffer(100, obs_dim=4, act_dim=2)
        for _ in range(20):
            buf.add(
                np.random.randn(4), np.random.randn(2),
                1.0, np.random.randn(4), False,
            )
        batch = buf.sample(8)
        assert batch["obs"].shape == (8, 4)
        assert batch["actions"].shape == (8, 2)
        assert batch["rewards"].shape == (8, 1)
        assert batch["next_obs"].shape == (8, 4)
        assert batch["dones"].shape == (8, 1)

    def test_sample_returns_tensors(self):
        buf = ReplayBuffer(100, obs_dim=4, act_dim=2)
        for _ in range(10):
            buf.add(np.random.randn(4), np.random.randn(2), 1.0, np.random.randn(4), False)
        batch = buf.sample(4)
        for v in batch.values():
            assert isinstance(v, torch.Tensor)

    def test_circular_overwrites(self):
        buf = ReplayBuffer(5, obs_dim=2, act_dim=1)
        for i in range(7):
            buf.add(np.array([i, i]), np.array([i]), float(i), np.array([i, i]), False)
        # Buffer should have entries 2,3,4,5,6 (oldest overwritten)
        assert len(buf) == 5
        # Latest entry should be index (7-1)%5 = 1
        assert buf.rewards[(7 - 1) % 5, 0] == 6.0


# ═══════════════════════════════════════════════════════════════════════════
#  SAC Agent
# ═══════════════════════════════════════════════════════════════════════════


class TestSACAgent:
    """SAC agent initialization and basic methods."""

    def test_creates(self):
        agent = SACAgent(obs_dim=29, act_dim=4)
        assert agent is not None

    def test_select_action_shape(self):
        agent = SACAgent(obs_dim=10, act_dim=3)
        obs = np.random.randn(10).astype(np.float32)
        action = agent.select_action(obs)
        assert action.shape == (3,)

    def test_action_bounded(self):
        agent = SACAgent(obs_dim=10, act_dim=3)
        obs = np.random.randn(10).astype(np.float32)
        action = agent.select_action(obs)
        assert np.all(action >= -1.0)
        assert np.all(action <= 1.0)

    def test_warmup_random(self):
        """During warmup, actions should be random (not from policy)."""
        cfg = SACConfig(warmup_steps=100)
        agent = SACAgent(obs_dim=5, act_dim=2, config=cfg)
        agent.total_steps = 0  # still in warmup
        actions = [agent.select_action(np.zeros(5)) for _ in range(20)]
        # Not all identical (random)
        assert not all(np.allclose(a, actions[0]) for a in actions)

    def test_store_transition(self):
        agent = SACAgent(obs_dim=5, act_dim=2)
        agent.store_transition(
            np.zeros(5), np.zeros(2), 1.0, np.zeros(5), False,
        )
        assert len(agent.buffer) == 1
        assert agent.total_steps == 1

    def test_update_returns_empty_when_buffer_small(self):
        agent = SACAgent(obs_dim=5, act_dim=2)
        for _ in range(10):
            agent.store_transition(
                np.random.randn(5), np.random.randn(2),
                1.0, np.random.randn(5), False,
            )
        result = agent.update()
        assert result == {}  # batch_size=256 > 10

    def test_single_gradient_step(self):
        """Fill buffer and run one update — should return metrics."""
        cfg = SACConfig(batch_size=16, warmup_steps=0)
        agent = SACAgent(obs_dim=5, act_dim=2, config=cfg)
        for _ in range(20):
            agent.store_transition(
                np.random.randn(5), np.random.randn(2),
                np.random.randn(), np.random.randn(5), False,
            )
        metrics = agent.update()
        assert "critic_loss" in metrics
        assert "actor_loss" in metrics
        assert "alpha_loss" in metrics
        assert "alpha" in metrics

    def test_target_entropy_default(self):
        agent = SACAgent(obs_dim=10, act_dim=4)
        assert agent.target_entropy == -4.0

    def test_deterministic_action(self):
        agent = SACAgent(obs_dim=5, act_dim=2)
        agent.total_steps = 99999  # past warmup
        obs = np.random.randn(5).astype(np.float32)
        a1 = agent.select_action(obs, deterministic=True)
        a2 = agent.select_action(obs, deterministic=True)
        np.testing.assert_array_equal(a1, a2)


class TestSACAgentSaveLoad:
    """Save and load round-trip."""

    def test_save_load(self, tmp_path):
        cfg = SACConfig(batch_size=8, warmup_steps=0)
        agent = SACAgent(obs_dim=5, act_dim=2, config=cfg)
        for _ in range(10):
            agent.store_transition(
                np.random.randn(5), np.random.randn(2),
                1.0, np.random.randn(5), False,
            )
        agent.update()

        path = tmp_path / "test_ckpt.pt"
        agent.save(path)
        assert path.exists()

        # Load into fresh agent
        agent2 = SACAgent(obs_dim=5, act_dim=2, config=cfg)
        agent2.load(path)
        assert agent2.total_steps == agent.total_steps

        # Same deterministic action
        obs = np.random.randn(5).astype(np.float32)
        agent.total_steps = 99999
        agent2.total_steps = 99999
        a1 = agent.select_action(obs, deterministic=True)
        a2 = agent2.select_action(obs, deterministic=True)
        np.testing.assert_array_almost_equal(a1, a2)
