"""
Place environment — place a held object at a target position.

Assumes the object starts near the gripper (simulating post-grasp).
The agent must move the object to a goal position and release.

Reward: negative distance object → goal. Bonus on placement.
"""

from __future__ import annotations

import numpy as np

from envs.multi_object_env import MultiObjectEnv

_TABLE_Z = 0.415
_PLACE_SCALE = 10.0
_HEIGHT_PENALTY = 2.0
_SUCCESS_BONUS = 50.0
_PLACE_THRESHOLD = 0.04  # 4 cm


class PlaceEnv(MultiObjectEnv):
    """
    Place primitive environment.

    One object is initialised near the EE (above table). Agent must
    lower it to a randomised goal position on the table.

    Obs augmented with goal position (3D) → 22 + 7*1 + 3 = 32D.
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
        base_dim = self.observation_space.shape[0]
        from gymnasium import spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(base_dim + 3,),
            dtype=np.float64,
        )
        self._goal_pos: np.ndarray = np.zeros(3)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._goal_pos = np.zeros(3)
        obs, info = super().reset(seed=seed, options=options)

        # Override object position: place near EE (above table)
        ee_pos = self.ee_pos
        self.data.qpos[9:12] = [ee_pos[0], ee_pos[1], ee_pos[2] - 0.02]
        self.data.qpos[12:16] = [1, 0, 0, 0]
        # Close gripper around object
        self.data.qpos[7] = 0.005
        self.data.qpos[8] = 0.005
        import mujoco
        mujoco.mj_forward(self.model, self.data)

        # Random goal on table
        self._goal_pos = np.array([
            self.np_random.uniform(0.30, 0.70),
            self.np_random.uniform(-0.10, 0.25),
            _TABLE_Z + 0.02,
        ])

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def _get_obs(self) -> np.ndarray:
        base_obs = super()._get_obs()
        return np.concatenate([base_obs, self._goal_pos])

    def _get_info(self) -> dict:
        info = super()._get_info()
        obj_pos = self.object_pos(0)
        obj_goal_dist = float(np.linalg.norm(obj_pos - self._goal_pos))
        info["goal_pos"] = self._goal_pos.copy()
        info["obj_goal_distance"] = obj_goal_dist
        info["success"] = obj_goal_dist < _PLACE_THRESHOLD
        return info

    def _compute_reward(self) -> float:
        obj_pos = self.object_pos(0)
        obj_goal_dist = float(np.linalg.norm(obj_pos - self._goal_pos))
        reward = -_PLACE_SCALE * obj_goal_dist

        # Penalise dropping object too high above goal
        height_above_goal = obj_pos[2] - self._goal_pos[2]
        if height_above_goal > 0.05:
            reward -= _HEIGHT_PENALTY * height_above_goal

        if obj_goal_dist < _PLACE_THRESHOLD:
            reward += _SUCCESS_BONUS

        return reward

    def _check_terminated(self) -> bool:
        obj_pos = self.object_pos(0)
        return float(np.linalg.norm(obj_pos - self._goal_pos)) < _PLACE_THRESHOLD
