"""Tests for v1.0.4 — MoveToEnv, PlaceEnv, and scripted baselines."""

from __future__ import annotations

import numpy as np
import pytest

from envs.move_to import MoveToEnv
from envs.place import PlaceEnv
from policies.scripted import ScriptedPickPlace, ScriptedMoveTo, Phase


# ── MoveToEnv ──────────────────────────────────────────────────────

class TestMoveToEnv:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.env = MoveToEnv()
        yield
        self.env.close()

    def test_obs_shape(self):
        obs, _ = self.env.reset(seed=0)
        assert obs.shape == (self.env.observation_space.shape[0],)

    def test_action_space(self):
        assert self.env.action_space.shape == (4,)

    def test_step_returns_five(self):
        self.env.reset(seed=0)
        obs, rew, term, trunc, info = self.env.step(self.env.action_space.sample())
        assert isinstance(rew, float)
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)

    def test_reset_deterministic(self):
        o1, _ = self.env.reset(seed=42)
        o2, _ = self.env.reset(seed=42)
        np.testing.assert_array_equal(o1, o2)

    def test_reward_negative_or_bonus(self):
        """Reward should be finite for random actions."""
        self.env.reset(seed=1)
        for _ in range(5):
            _, r, _, _, _ = self.env.step(self.env.action_space.sample())
            assert np.isfinite(r)

    def test_multiple_episodes(self):
        for seed in range(3):
            obs, _ = self.env.reset(seed=seed)
            for _ in range(10):
                obs, _, term, trunc, _ = self.env.step(
                    self.env.action_space.sample())
                if term or trunc:
                    break


# ── PlaceEnv ───────────────────────────────────────────────────────

class TestPlaceEnv:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.env = PlaceEnv()
        yield
        self.env.close()

    def test_obs_shape(self):
        obs, _ = self.env.reset(seed=0)
        assert obs.shape == (self.env.observation_space.shape[0],)

    def test_obs_includes_goal(self):
        """PlaceEnv obs = 29 base + 3 goal = 32."""
        obs, _ = self.env.reset(seed=0)
        assert obs.shape[0] == 32

    def test_action_space(self):
        assert self.env.action_space.shape == (4,)

    def test_step_returns_five(self):
        self.env.reset(seed=0)
        obs, rew, term, trunc, info = self.env.step(self.env.action_space.sample())
        assert isinstance(rew, float)
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)

    def test_reset_deterministic(self):
        o1, _ = self.env.reset(seed=42)
        o2, _ = self.env.reset(seed=42)
        np.testing.assert_array_equal(o1, o2)

    def test_goal_in_info(self):
        _, info = self.env.reset(seed=0)
        assert "goal_pos" in info
        assert info["goal_pos"].shape == (3,)

    def test_reward_finite(self):
        self.env.reset(seed=1)
        for _ in range(5):
            _, r, _, _, _ = self.env.step(self.env.action_space.sample())
            assert np.isfinite(r)

    def test_multiple_episodes(self):
        for seed in range(3):
            obs, _ = self.env.reset(seed=seed)
            for _ in range(10):
                obs, _, term, trunc, _ = self.env.step(
                    self.env.action_space.sample())
                if term or trunc:
                    break


# ── ScriptedPickPlace ──────────────────────────────────────────────

class TestScriptedPickPlace:
    def test_init_phase(self):
        sp = ScriptedPickPlace()
        assert sp.phase == Phase.APPROACH

    def test_reset_resets_phase(self):
        sp = ScriptedPickPlace()
        sp.phase = Phase.LIFT
        sp.reset()
        assert sp.phase == Phase.APPROACH

    def test_act_returns_4d(self):
        sp = ScriptedPickPlace()
        info = {
            "ee_pos": np.array([0.5, 0.0, 0.6]),
            "obj_pos": np.array([0.5, 0.0, 0.435]),
            "goal_pos": np.array([0.6, 0.0, 0.435]),
        }
        action = sp.act(info)
        assert action.shape == (4,)
        assert np.all(np.abs(action) <= 1.0)

    def test_phase_transitions(self):
        """Run scripted policy; it should progress through phases."""
        sp = ScriptedPickPlace(position_gain=50.0, threshold=0.05)
        # Simulate approach -> descend transition
        info = {
            "ee_pos": np.array([0.5, 0.0, 0.565]),  # above object
            "obj_pos": np.array([0.5, 0.0, 0.435]),
            "goal_pos": np.array([0.6, 0.0, 0.435]),
        }
        sp.act(info)
        assert sp.phase in (Phase.APPROACH, Phase.DESCEND)

    def test_gripper_open_during_approach(self):
        sp = ScriptedPickPlace()
        info = {
            "ee_pos": np.array([0.3, 0.1, 0.6]),
            "obj_pos": np.array([0.5, 0.0, 0.435]),
        }
        action = sp.act(info)
        assert action[3] == 1.0  # open

    def test_gripper_closed_after_grasp(self):
        sp = ScriptedPickPlace()
        sp.phase = Phase.LIFT
        info = {
            "ee_pos": np.array([0.5, 0.0, 0.45]),
            "obj_pos": np.array([0.5, 0.0, 0.435]),
            "goal_pos": np.array([0.6, 0.0, 0.435]),
        }
        action = sp.act(info)
        assert action[3] == -1.0  # closed

    def test_done_phase(self):
        sp = ScriptedPickPlace()
        sp.phase = Phase.DONE
        info = {
            "ee_pos": np.array([0.6, 0.0, 0.45]),
            "obj_pos": np.array([0.5, 0.0, 0.435]),
            "goal_pos": np.array([0.6, 0.0, 0.435]),
        }
        action = sp.act(info)
        assert action[3] == 1.0  # open

    def test_obj_positions_dict_format(self):
        """Support info with obj_positions dict (from MultiObjectEnv)."""
        sp = ScriptedPickPlace()
        info = {
            "ee_pos": np.array([0.5, 0.0, 0.6]),
            "obj_positions": {"red_box_0": np.array([0.5, 0.0, 0.435])},
            "goal_pos": np.array([0.6, 0.0, 0.435]),
        }
        action = sp.act(info)
        assert action.shape == (4,)


# ── ScriptedMoveTo ─────────────────────────────────────────────────

class TestScriptedMoveTo:
    def test_act_returns_4d(self):
        sm = ScriptedMoveTo()
        info = {
            "ee_pos": np.array([0.5, 0.0, 0.6]),
            "obj_pos": np.array([0.5, 0.0, 0.435]),
        }
        action = sm.act(info)
        assert action.shape == (4,)
        assert np.all(np.abs(action) <= 1.0)

    def test_gripper_always_open(self):
        sm = ScriptedMoveTo()
        info = {
            "ee_pos": np.array([0.3, 0.1, 0.6]),
            "obj_pos": np.array([0.5, 0.0, 0.435]),
        }
        action = sm.act(info)
        assert action[3] == 1.0

    def test_proportional_direction(self):
        sm = ScriptedMoveTo(gain=1.0)
        info = {
            "ee_pos": np.array([0.0, 0.0, 0.5]),
            "obj_pos": np.array([0.5, 0.0, 0.5]),
        }
        action = sm.act(info)
        assert action[0] > 0  # should move +x toward object

    def test_obj_positions_dict(self):
        sm = ScriptedMoveTo()
        info = {
            "ee_pos": np.array([0.5, 0.0, 0.6]),
            "obj_positions": {"blue_cyl_0": np.array([0.5, 0.0, 0.435])},
        }
        action = sm.act(info)
        assert action.shape == (4,)

    def test_reset_noop(self):
        sm = ScriptedMoveTo()
        sm.reset()  # should not raise
