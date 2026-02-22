"""Tests for multi-object and task-specific environments."""

import numpy as np
import pytest

from envs.multi_object_env import MultiObjectEnv
from envs.pick_place import PickPlaceEnv
from envs.color_pick import ColorPickEnv
from envs.stack import StackEnv


# ═══════════════════════════════════════════════════════════════════════════
#  MultiObjectEnv
# ═══════════════════════════════════════════════════════════════════════════


class TestMultiObjectEnvCreation:
    """MultiObjectEnv creates and exposes correct spaces."""

    def test_creates_without_error(self):
        env = MultiObjectEnv(n_objects=2)
        assert env is not None
        env.close()

    def test_obs_space_shape_2_objects(self):
        env = MultiObjectEnv(n_objects=2)
        # 22 base + 7*2 objects = 36
        assert env.observation_space.shape == (36,)
        env.close()

    def test_obs_space_shape_4_objects(self):
        env = MultiObjectEnv(n_objects=4)
        # 22 base + 7*4 = 50
        assert env.observation_space.shape == (50,)
        env.close()

    def test_action_space(self):
        env = MultiObjectEnv(n_objects=3)
        assert env.action_space.shape == (4,)
        env.close()


class TestMultiObjectEnvReset:
    """Reset generates new scenes with correct object count."""

    def test_reset_returns_obs_info(self):
        env = MultiObjectEnv(n_objects=3)
        obs, info = env.reset(seed=42)
        assert obs.shape == env.observation_space.shape
        assert isinstance(info, dict)
        env.close()

    def test_obs_is_finite(self):
        env = MultiObjectEnv(n_objects=3)
        obs, _ = env.reset(seed=42)
        assert np.all(np.isfinite(obs))
        env.close()

    def test_n_objects_in_info(self):
        env = MultiObjectEnv(n_objects=4)
        _, info = env.reset(seed=42)
        assert info["n_objects"] == 4
        assert len(info["obj_positions"]) == 4
        env.close()

    def test_different_seeds_different_objects(self):
        env = MultiObjectEnv(n_objects=3)
        obs1, info1 = env.reset(seed=1)
        obs2, info2 = env.reset(seed=2)
        # Object positions should differ
        assert not np.allclose(obs1, obs2)
        env.close()

    def test_deterministic_same_seed(self):
        env = MultiObjectEnv(n_objects=3)
        obs1, _ = env.reset(seed=77)
        obs2, _ = env.reset(seed=77)
        np.testing.assert_array_almost_equal(obs1, obs2, decimal=6)
        env.close()


class TestMultiObjectEnvStep:
    """Step works with multi-object scene."""

    def test_step_returns_correct_tuple(self):
        env = MultiObjectEnv(n_objects=3)
        env.reset(seed=42)
        obs, rew, term, trunc, info = env.step(np.zeros(4))
        assert obs.shape == env.observation_space.shape
        assert isinstance(rew, float)
        assert not term  # base env never terminates
        env.close()

    def test_multiple_steps(self):
        env = MultiObjectEnv(n_objects=2)
        env.reset(seed=42)
        for _ in range(20):
            obs, _, _, _, _ = env.step(env.action_space.sample())
            assert np.all(np.isfinite(obs))
        env.close()


class TestMultiObjectEnvRender:
    """Rendering works with dynamic scenes."""

    def test_render_rgb(self):
        env = MultiObjectEnv(n_objects=3, render_mode="rgb_array")
        env.reset(seed=42)
        img = env.render()
        assert img is not None
        assert img.shape == (128, 128, 3)
        env.close()


class TestMultiObjectEnvConvenience:
    """Convenience methods work."""

    def test_object_pos(self):
        env = MultiObjectEnv(n_objects=3)
        env.reset(seed=42)
        for i in range(3):
            pos = env.object_pos(i)
            assert pos.shape == (3,)
            assert np.all(np.isfinite(pos))
        env.close()

    def test_object_color(self):
        env = MultiObjectEnv(n_objects=3)
        env.reset(seed=42)
        colors = [env.object_color(i) for i in range(3)]
        # unique_colors=True by default → all different
        assert len(set(colors)) == 3
        env.close()


# ═══════════════════════════════════════════════════════════════════════════
#  PickPlaceEnv (L1)
# ═══════════════════════════════════════════════════════════════════════════


class TestPickPlaceEnvCreation:
    def test_creates(self):
        env = PickPlaceEnv()
        assert env is not None
        env.close()

    def test_obs_space_includes_goal(self):
        env = PickPlaceEnv()
        # 22 + 7*1 + 3 = 32
        assert env.observation_space.shape == (32,)
        env.close()


class TestPickPlaceEnvReset:
    def test_reset_has_goal(self):
        env = PickPlaceEnv()
        obs, info = env.reset(seed=42)
        assert "goal_pos" in info
        assert info["goal_pos"].shape == (3,)
        env.close()

    def test_goal_on_table(self):
        env = PickPlaceEnv()
        _, info = env.reset(seed=42)
        gp = info["goal_pos"]
        assert 0.25 < gp[0] < 0.75
        assert -0.15 < gp[1] < 0.30
        env.close()


class TestPickPlaceEnvStep:
    def test_step_reward_nonzero(self):
        """Approaching object should give shaped reward."""
        env = PickPlaceEnv()
        env.reset(seed=42)
        _, rew, _, _, _ = env.step(np.zeros(4))
        # Approach penalty: should be negative (distance-based)
        assert rew < 0
        env.close()

    def test_info_has_success(self):
        env = PickPlaceEnv()
        env.reset(seed=42)
        _, _, _, _, info = env.step(np.zeros(4))
        assert "success" in info
        assert isinstance(info["success"], bool)
        env.close()


# ═══════════════════════════════════════════════════════════════════════════
#  ColorPickEnv (L2)
# ═══════════════════════════════════════════════════════════════════════════


class TestColorPickEnvCreation:
    def test_creates(self):
        env = ColorPickEnv(n_objects=3)
        assert env is not None
        env.close()

    def test_obs_space_includes_color_onehot(self):
        env = ColorPickEnv(n_objects=3)
        # 22 + 7*3 + 6 = 49
        assert env.observation_space.shape == (49,)
        env.close()


class TestColorPickEnvReset:
    def test_has_target_color(self):
        env = ColorPickEnv(n_objects=3)
        _, info = env.reset(seed=42)
        assert "target_color" in info
        assert info["target_color"] in [
            "red", "green", "blue", "yellow", "orange", "purple",
        ]
        env.close()

    def test_target_idx_valid(self):
        env = ColorPickEnv(n_objects=3)
        _, info = env.reset(seed=42)
        assert 0 <= info["target_idx"] < 3
        env.close()


class TestColorPickEnvStep:
    def test_step_shaped_reward(self):
        env = ColorPickEnv(n_objects=3)
        env.reset(seed=42)
        _, rew, _, _, _ = env.step(np.zeros(4))
        # Should have approach penalty
        assert rew < 0
        env.close()

    def test_info_has_success(self):
        env = ColorPickEnv(n_objects=3)
        env.reset(seed=42)
        _, _, _, _, info = env.step(np.zeros(4))
        assert "success" in info
        env.close()


# ═══════════════════════════════════════════════════════════════════════════
#  StackEnv (L3)
# ═══════════════════════════════════════════════════════════════════════════


class TestStackEnvCreation:
    def test_creates(self):
        env = StackEnv(n_objects=3)
        assert env is not None
        env.close()

    def test_obs_space_includes_stack_info(self):
        env = StackEnv(n_objects=3)
        # 22 + 7*3 + 3 + 3 + 1 = 50
        assert env.observation_space.shape == (50,)
        env.close()


class TestStackEnvReset:
    def test_has_stack_order(self):
        env = StackEnv(n_objects=3)
        _, info = env.reset(seed=42)
        assert "stack_order" in info
        assert len(info["stack_order"]) == 3
        # Must be a permutation of [0,1,2]
        assert sorted(info["stack_order"]) == [0, 1, 2]
        env.close()

    def test_has_stack_pos(self):
        env = StackEnv(n_objects=3)
        _, info = env.reset(seed=42)
        assert info["stack_pos"].shape == (3,)
        env.close()


class TestStackEnvStep:
    def test_step_shaped_reward(self):
        env = StackEnv(n_objects=3)
        env.reset(seed=42)
        _, rew, _, _, _ = env.step(np.zeros(4))
        # Approach penalty toward next object
        assert isinstance(rew, float)
        env.close()

    def test_info_has_success(self):
        env = StackEnv(n_objects=3)
        env.reset(seed=42)
        _, _, _, _, info = env.step(np.zeros(4))
        assert "success" in info
        assert info["n_stacked"] == 0  # nothing stacked yet
        env.close()
