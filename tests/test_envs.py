"""Tests for the MuJoCo tabletop environment."""

import numpy as np
import pytest

from envs.tabletop import TabletopEnv


class TestTabletopEnvCreation:
    """Environment can be created and introspected."""

    def test_creates_without_error(self):
        env = TabletopEnv()
        assert env is not None
        env.close()

    def test_observation_space_shape(self):
        env = TabletopEnv()
        assert env.observation_space.shape == (29,)
        env.close()

    def test_action_space_shape(self):
        env = TabletopEnv()
        assert env.action_space.shape == (4,)
        env.close()

    def test_action_space_bounds(self):
        env = TabletopEnv()
        np.testing.assert_array_equal(env.action_space.low, -1.0)
        np.testing.assert_array_equal(env.action_space.high, 1.0)
        env.close()


class TestTabletopEnvReset:
    """Environment resets correctly."""

    def test_reset_returns_obs_and_info(self):
        env = TabletopEnv()
        obs, info = env.reset(seed=42)
        assert obs.shape == (29,)
        assert isinstance(info, dict)
        env.close()

    def test_obs_is_finite(self):
        env = TabletopEnv()
        obs, _ = env.reset(seed=42)
        assert np.all(np.isfinite(obs)), f"Non-finite values in obs: {obs}"
        env.close()

    def test_deterministic_with_same_seed(self):
        env1 = TabletopEnv()
        obs1, _ = env1.reset(seed=123)
        env1.close()

        env2 = TabletopEnv()
        obs2, _ = env2.reset(seed=123)
        env2.close()

        np.testing.assert_array_equal(obs1, obs2)

    def test_different_seeds_give_different_obs(self):
        env = TabletopEnv()
        obs1, _ = env.reset(seed=1)
        obs2, _ = env.reset(seed=2)
        # Object positions should differ (last 7 dims include obj pose)
        assert not np.allclose(obs1, obs2)
        env.close()


class TestTabletopEnvStep:
    """Environment steps correctly."""

    def test_step_returns_correct_tuple(self):
        env = TabletopEnv()
        env.reset(seed=42)
        action = np.zeros(4)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (29,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        env.close()

    def test_step_with_random_action(self):
        env = TabletopEnv()
        env.reset(seed=42)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.all(np.isfinite(obs))
        env.close()

    def test_multiple_steps(self):
        env = TabletopEnv()
        env.reset(seed=42)
        for _ in range(20):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert np.all(np.isfinite(obs))
            if terminated:
                break
        env.close()

    def test_zero_action_is_stable(self):
        """With zero action, arm should stay reasonably near home position."""
        env = TabletopEnv()
        env.reset(seed=42)
        obs0, _ = env.reset(seed=42)
        for _ in range(10):
            obs, _, _, _, _ = env.step(np.zeros(4))
        # Joint positions (first 7) — some drift expected from gravity
        # on PD-controlled arm, but should not be extreme
        drift = np.abs(obs[:7] - obs0[:7])
        assert np.all(drift < 1.5), f"Joint drift too large: {drift}"
        env.close()


class TestTabletopEnvInfo:
    """Info dict contains useful data."""

    def test_info_keys(self):
        env = TabletopEnv()
        env.reset(seed=42)
        _, _, _, _, info = env.step(np.zeros(4))
        assert "ee_pos" in info
        assert "obj_pos" in info
        assert "distance" in info
        env.close()

    def test_info_shapes(self):
        env = TabletopEnv()
        env.reset(seed=42)
        _, _, _, _, info = env.step(np.zeros(4))
        assert info["ee_pos"].shape == (3,)
        assert info["obj_pos"].shape == (3,)
        assert isinstance(info["distance"], float)
        env.close()

    def test_distance_is_positive(self):
        env = TabletopEnv()
        env.reset(seed=42)
        _, _, _, _, info = env.step(np.zeros(4))
        assert info["distance"] >= 0
        env.close()


class TestTabletopEnvRender:
    """Rendering produces valid images."""

    def test_render_rgb_array(self):
        env = TabletopEnv(render_mode="rgb_array")
        env.reset(seed=42)
        img = env.render()
        assert img is not None
        assert img.shape == (128, 128, 3)
        assert img.dtype == np.uint8
        env.close()

    def test_render_none_mode(self):
        env = TabletopEnv(render_mode=None)
        env.reset(seed=42)
        result = env.render()
        assert result is None
        env.close()

    def test_custom_image_size(self):
        env = TabletopEnv(render_mode="rgb_array", image_size=64)
        env.reset(seed=42)
        img = env.render()
        assert img.shape == (64, 64, 3)
        env.close()


class TestTabletopEnvProperties:
    """Convenience properties work."""

    def test_ee_pos(self):
        env = TabletopEnv()
        env.reset(seed=42)
        ee = env.ee_pos
        assert ee.shape == (3,)
        assert np.all(np.isfinite(ee))
        env.close()

    def test_object_pos(self):
        env = TabletopEnv()
        env.reset(seed=42)
        obj = env.object_pos
        assert obj.shape == (3,)
        # Block should be on or near table surface (z ≈ 0.435)
        assert 0.3 < obj[2] < 0.6, f"Block z={obj[2]} seems wrong"
        env.close()
