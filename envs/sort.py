"""
Sort environment (L4) — multi-object bin assignment.

Agent must pick up objects and place them in the correct zone
based on their color. 3 objects → 3 separate target positions.
"""

from __future__ import annotations

import numpy as np

from envs.multi_object_env import MultiObjectEnv

_TABLE_Z = 0.415
_SORT_SCALE = 5.0
_PER_OBJECT_BONUS = 20.0
_ALL_SORTED_BONUS = 100.0
_SORT_THRESHOLD = 0.05  # 5 cm


# Fixed zone positions on the table (left, center, right)
_ZONE_POSITIONS = [
    np.array([0.35, -0.10, _TABLE_Z + 0.02]),
    np.array([0.50, -0.10, _TABLE_Z + 0.02]),
    np.array([0.65, -0.10, _TABLE_Z + 0.02]),
]


class SortEnv(MultiObjectEnv):
    """
    Sort environment: place N objects into N designated zones.

    Each object is assigned a target zone. Agent must pick and place
    each object into its zone. Obs augmented with zone positions.

    Obs = 22 + 7*N + 3*N (base + per-object zone target)
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
        )
        base_dim = self.observation_space.shape[0]
        from gymnasium import spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(base_dim + 3 * n_objects,),
            dtype=np.float64,
        )
        self._zone_positions: list[np.ndarray] = []
        self._sorted_flags: list[bool] = []

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._zone_positions = [np.zeros(3)] * self.n_objects
        self._sorted_flags = [False] * self.n_objects
        obs, info = super().reset(seed=seed, options=options)

        # Assign zones (shuffle order per episode)
        zone_order = list(range(self.n_objects))
        self.np_random.shuffle(zone_order)
        self._zone_positions = [
            _ZONE_POSITIONS[zone_order[i] % len(_ZONE_POSITIONS)]
            for i in range(self.n_objects)
        ]

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def _get_obs(self) -> np.ndarray:
        base_obs = super()._get_obs()
        zones = np.concatenate(self._zone_positions)
        return np.concatenate([base_obs, zones])

    def _compute_reward(self) -> float:
        reward = 0.0
        n_sorted = 0

        for i in range(self.n_objects):
            obj_pos = self.object_pos(i)
            zone_pos = self._zone_positions[i]
            dist = float(np.linalg.norm(obj_pos[:2] - zone_pos[:2]))

            # Shaped distance reward
            reward -= _SORT_SCALE * dist

            # Per-object bonus
            if dist < _SORT_THRESHOLD:
                if not self._sorted_flags[i]:
                    reward += _PER_OBJECT_BONUS
                    self._sorted_flags[i] = True
                n_sorted += 1

        # All sorted bonus
        if n_sorted == self.n_objects:
            reward += _ALL_SORTED_BONUS

        return reward

    def _check_terminated(self) -> bool:
        for i in range(self.n_objects):
            obj_pos = self.object_pos(i)
            zone_pos = self._zone_positions[i]
            if float(np.linalg.norm(obj_pos[:2] - zone_pos[:2])) >= _SORT_THRESHOLD:
                return False
        return True

    def _get_info(self) -> dict:
        info = super()._get_info()
        n_sorted = 0
        for i in range(self.n_objects):
            obj_pos = self.object_pos(i)
            zone_pos = self._zone_positions[i]
            if float(np.linalg.norm(obj_pos[:2] - zone_pos[:2])) < _SORT_THRESHOLD:
                n_sorted += 1
        info["n_sorted"] = n_sorted
        info["zone_positions"] = [z.copy() for z in self._zone_positions]
        info["success"] = n_sorted == self.n_objects
        return info
