"""
Pick-and-Place environment (Task Level L1).

The agent must pick up a single target object and place it at a
randomized goal position on the table. Shaped reward with four phases:

    1. Approach — negative distance from EE to object
    2. Grasp    — bonus when gripper is closed around object
    3. Lift     — reward for lifting object off table
    4. Place    — negative distance from object to goal position
"""

from __future__ import annotations

import mujoco
import numpy as np

from envs.multi_object_env import MultiObjectEnv

# ── Reward constants ─────────────────────────────────────────────────────────

_APPROACH_SCALE = 1.0
_GRASP_BONUS = 5.0
_LIFT_BONUS = 10.0
_PLACE_SCALE = 20.0
_SUCCESS_BONUS = 50.0

_GRASP_THRESHOLD = 0.04   # EE-to-object distance to count as "grasped"
_LIFT_HEIGHT = 0.08        # height above table surface to count as "lifted"
_PLACE_THRESHOLD = 0.04    # object-goal distance to count as "placed"
_TABLE_Z = 0.415


class PickPlaceEnv(MultiObjectEnv):
    """
    L1 pick-and-place task.

    One target object spawned on the table. The goal is to pick it up
    and move it to a randomized target position. Observation is augmented
    with the goal position (3D) → total obs dim = 22 + 7*n + 3.
    """

    def __init__(
        self,
        render_mode: str | None = None,
        image_size: int = 128,
    ):
        super().__init__(
            n_objects=1,
            render_mode=render_mode,
            image_size=image_size,
        )
        # Augment obs space with goal pos (3D)
        base_dim = self.observation_space.shape[0]
        from gymnasium import spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(base_dim + 3,),
            dtype=np.float64,
        )
        self._goal_pos: np.ndarray = np.zeros(3)
        self._grasped = False
        self._lifted = False

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)

        # Randomize goal position on table (different from object)
        obj_pos = self.object_pos(0)
        for _ in range(50):
            gx = self.np_random.uniform(0.30, 0.70)
            gy = self.np_random.uniform(-0.10, 0.25)
            gz = _TABLE_Z + 0.02  # just above table
            goal = np.array([gx, gy, gz])
            if np.linalg.norm(goal[:2] - obj_pos[:2]) > 0.08:
                break
        self._goal_pos = goal
        self._grasped = False
        self._lifted = False

        # Augment obs with goal
        obs = np.concatenate([obs, self._goal_pos])
        info["goal_pos"] = self._goal_pos.copy()
        return obs, info

    def _get_obs(self) -> np.ndarray:
        base_obs = super()._get_obs()
        return np.concatenate([base_obs, self._goal_pos])

    def _get_info(self) -> dict:
        info = super()._get_info()
        info["goal_pos"] = self._goal_pos.copy()
        obj_pos = self.object_pos(0)
        info["obj_goal_distance"] = float(
            np.linalg.norm(obj_pos - self._goal_pos)
        )
        info["grasped"] = self._grasped
        info["lifted"] = self._lifted
        info["success"] = info["obj_goal_distance"] < _PLACE_THRESHOLD
        return info

    def _compute_reward(self) -> float:
        ee_pos = self.ee_pos
        obj_pos = self.object_pos(0)
        ee_obj_dist = float(np.linalg.norm(ee_pos - obj_pos))
        obj_goal_dist = float(np.linalg.norm(obj_pos - self._goal_pos))
        obj_height = obj_pos[2] - _TABLE_Z

        reward = 0.0

        # Phase 1: Approach
        reward -= _APPROACH_SCALE * ee_obj_dist

        # Phase 2: Grasp (EE close + gripper closing)
        gripper_open = 0.5 * (self.data.qpos[7] + self.data.qpos[8])
        if ee_obj_dist < _GRASP_THRESHOLD and gripper_open < 0.015:
            self._grasped = True
            reward += _GRASP_BONUS

        # Phase 3: Lift
        if self._grasped and obj_height > _LIFT_HEIGHT:
            self._lifted = True
            reward += _LIFT_BONUS

        # Phase 4: Place (only after lifting)
        if self._lifted:
            reward -= _PLACE_SCALE * obj_goal_dist
            if obj_goal_dist < _PLACE_THRESHOLD:
                reward += _SUCCESS_BONUS

        return reward

    def _check_terminated(self) -> bool:
        """Terminate on successful placement."""
        obj_pos = self.object_pos(0)
        obj_goal_dist = float(np.linalg.norm(obj_pos - self._goal_pos))
        return obj_goal_dist < _PLACE_THRESHOLD and self._lifted
