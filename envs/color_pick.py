"""
Color-Pick environment (Task Level L2).

Multiple objects with distinct colours on the table. A target colour
is specified — the agent must pick the correct object.

Shaped reward:
    1. Approach — negative distance to the *correct* object
    2. Grasp   — bonus for grasping the correct one, penalty for wrong one
    3. Lift    — bonus for lifting the correct object
"""

from __future__ import annotations

import numpy as np

from envs.multi_object_env import MultiObjectEnv
from envs.object_spawner import COLOR_NAMES

# ── Reward constants ─────────────────────────────────────────────────────────

_APPROACH_SCALE = 1.0
_GRASP_CORRECT_BONUS = 10.0
_GRASP_WRONG_PENALTY = -5.0
_LIFT_BONUS = 15.0
_SUCCESS_BONUS = 50.0

_GRASP_THRESHOLD = 0.04
_LIFT_HEIGHT = 0.08
_TABLE_Z = 0.415


class ColorPickEnv(MultiObjectEnv):
    """
    L2 colour-selection task.

    N objects with unique colours. A target colour index is included in the
    observation as a one-hot vector (6D for 6 possible colours).

    Obs dim = 22 + 7*n_objects + 6  (6 = one-hot colour vector)
    """

    def __init__(
        self,
        n_objects: int = 3,
        render_mode: str | None = None,
        image_size: int = 128,
    ):
        super().__init__(
            n_objects=n_objects,
            render_mode=render_mode,
            image_size=image_size,
            unique_colors=True,
        )
        # Augment obs space with one-hot colour target (6D)
        base_dim = self.observation_space.shape[0]
        from gymnasium import spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(base_dim + len(COLOR_NAMES),),
            dtype=np.float64,
        )
        self._target_idx: int = 0
        self._target_color: str = ""
        self._grasped_correct = False
        self._lifted = False

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # We must initialise _target_color before super().reset() because
        # the parent calls _get_obs() which invokes _target_one_hot().
        # Use a temporary default so the first _get_obs() doesn't crash.
        self._target_color = COLOR_NAMES[0]  # placeholder
        self._target_idx = 0
        self._grasped_correct = False
        self._lifted = False

        # Parent builds the scene and calls _get_obs() / _get_info()
        obs, info = super().reset(seed=seed, options=options)

        # Now pick the real random target from spawned objects
        self._target_idx = int(self.np_random.integers(0, self.n_objects))
        self._target_color = self._obj_specs[self._target_idx].color_name

        # Rebuild obs and info with correct target
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def _target_one_hot(self) -> np.ndarray:
        vec = np.zeros(len(COLOR_NAMES), dtype=np.float64)
        idx = COLOR_NAMES.index(self._target_color)
        vec[idx] = 1.0
        return vec

    def _get_obs(self) -> np.ndarray:
        base_obs = super()._get_obs()
        return np.concatenate([base_obs, self._target_one_hot()])

    def _get_info(self) -> dict:
        info = super()._get_info()
        info["target_color"] = self._target_color
        info["target_idx"] = self._target_idx
        info["grasped_correct"] = self._grasped_correct
        info["lifted"] = self._lifted
        target_pos = self.object_pos(self._target_idx)
        info["target_pos"] = target_pos.copy()
        ee_target_dist = float(np.linalg.norm(self.ee_pos - target_pos))
        info["ee_target_distance"] = ee_target_dist
        info["success"] = self._lifted
        return info

    def _compute_reward(self) -> float:
        ee_pos = self.ee_pos
        target_pos = self.object_pos(self._target_idx)
        ee_target_dist = float(np.linalg.norm(ee_pos - target_pos))
        target_height = target_pos[2] - _TABLE_Z
        gripper_open = 0.5 * (self.data.qpos[7] + self.data.qpos[8])

        reward = 0.0

        # Phase 1: Approach correct object
        reward -= _APPROACH_SCALE * ee_target_dist

        # Phase 2: Grasp check
        if gripper_open < 0.015:
            # Check which object (if any) is closest to EE
            closest_idx = -1
            closest_dist = float("inf")
            for i in range(self.n_objects):
                d = float(np.linalg.norm(ee_pos - self.object_pos(i)))
                if d < closest_dist:
                    closest_dist = d
                    closest_idx = i

            if closest_dist < _GRASP_THRESHOLD:
                if closest_idx == self._target_idx:
                    self._grasped_correct = True
                    reward += _GRASP_CORRECT_BONUS
                else:
                    reward += _GRASP_WRONG_PENALTY

        # Phase 3: Lift correct object
        if self._grasped_correct and target_height > _LIFT_HEIGHT:
            self._lifted = True
            reward += _LIFT_BONUS + _SUCCESS_BONUS

        return reward

    def _check_terminated(self) -> bool:
        """Terminate when correct object is lifted."""
        return self._lifted
