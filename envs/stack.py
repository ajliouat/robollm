"""
Stack environment (Task Level L3).

Multiple objects on the table. The agent must stack them at a target
position in a specified order (bottom to top). Reward shaping:

    1. Approach — distance to the *next* object to pick
    2. Grasp   — bonus for grasping correct next object
    3. Lift    — bonus for lifting
    4. Place   — bonus for placing on stack at correct height
    5. Stability — bonus if stack is stable (objects stayed stacked)
"""

from __future__ import annotations

import numpy as np

from envs.multi_object_env import MultiObjectEnv

# ── Reward constants ─────────────────────────────────────────────────────────

_APPROACH_SCALE = 1.0
_GRASP_BONUS = 5.0
_LIFT_BONUS = 5.0
_STACK_BONUS = 20.0
_FULL_STACK_BONUS = 100.0

_GRASP_THRESHOLD = 0.04
_LIFT_HEIGHT = 0.06
_STACK_XY_THRESHOLD = 0.035  # object must be within this XY of stack centre
_STACK_Z_TOLERANCE = 0.015   # Z must be within this of expected height
_TABLE_Z = 0.415
_OBJ_HEIGHT = 0.04  # full height of 4cm objects (2cm half-extent)


class StackEnv(MultiObjectEnv):
    """
    L3 stacking task.

    N objects spawned on the table. A random stack order is chosen.
    Observation is augmented with:
        - target stack position (3D)
        - stack order as indices (n_objects D, each 0..n-1)
        - current stack progress (1D: fraction of objects stacked)

    Obs dim = 22 + 7*n_objects + 3 + n_objects + 1
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
        # Augment obs: stack_pos(3) + order(n_objects) + progress(1)
        base_dim = self.observation_space.shape[0]
        extra = 3 + n_objects + 1
        from gymnasium import spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(base_dim + extra,),
            dtype=np.float64,
        )
        self._stack_pos: np.ndarray = np.zeros(3)
        self._stack_order: list[int] = []
        self._n_stacked: int = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)

        # Random stack target position on table
        self._stack_pos = np.array([
            self.np_random.uniform(0.35, 0.65),
            self.np_random.uniform(-0.05, 0.20),
            _TABLE_Z + 0.02,  # base of stack
        ])

        # Random stack order
        self._stack_order = list(
            self.np_random.permutation(self.n_objects)
        )
        self._n_stacked = 0

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def _get_obs(self) -> np.ndarray:
        base_obs = super()._get_obs()
        order = np.array(self._stack_order, dtype=np.float64)
        progress = np.array([self._n_stacked / self.n_objects])
        return np.concatenate([base_obs, self._stack_pos, order, progress])

    def _get_info(self) -> dict:
        info = super()._get_info()
        info["stack_pos"] = self._stack_pos.copy()
        info["stack_order"] = list(self._stack_order)
        info["n_stacked"] = self._n_stacked
        info["success"] = self._n_stacked == self.n_objects
        return info

    def _expected_stack_z(self, level: int) -> float:
        """Expected Z for an object at stack level *level* (0-based)."""
        return _TABLE_Z + _OBJ_HEIGHT * (level + 0.5)

    def _check_stacked(self) -> int:
        """Count how many objects are correctly stacked (from bottom up)."""
        count = 0
        for level, obj_idx in enumerate(self._stack_order):
            pos = self.object_pos(obj_idx)
            expected_z = self._expected_stack_z(level)
            xy_dist = float(np.linalg.norm(pos[:2] - self._stack_pos[:2]))
            z_err = abs(pos[2] - expected_z)
            if xy_dist < _STACK_XY_THRESHOLD and z_err < _STACK_Z_TOLERANCE:
                count += 1
            else:
                break  # stack must be contiguous from bottom
        return count

    def _compute_reward(self) -> float:
        ee_pos = self.ee_pos
        reward = 0.0

        # Check current stack status
        new_stacked = self._check_stacked()

        # Bonus for newly stacked objects
        if new_stacked > self._n_stacked:
            reward += _STACK_BONUS * (new_stacked - self._n_stacked)
            self._n_stacked = new_stacked

        # Full stack bonus
        if self._n_stacked == self.n_objects:
            reward += _FULL_STACK_BONUS
            return reward

        # Guide towards next object to pick
        next_idx = self._stack_order[self._n_stacked]
        next_pos = self.object_pos(next_idx)
        ee_next_dist = float(np.linalg.norm(ee_pos - next_pos))

        # Phase 1: Approach next object
        reward -= _APPROACH_SCALE * ee_next_dist

        # Phase 2: Grasp
        gripper_open = 0.5 * (self.data.qpos[7] + self.data.qpos[8])
        if ee_next_dist < _GRASP_THRESHOLD and gripper_open < 0.015:
            reward += _GRASP_BONUS

        # Phase 3: Lift
        next_height = next_pos[2] - _TABLE_Z
        if next_height > _LIFT_HEIGHT:
            reward += _LIFT_BONUS

            # Guide towards stack position
            stack_target = np.array([
                self._stack_pos[0],
                self._stack_pos[1],
                self._expected_stack_z(self._n_stacked),
            ])
            place_dist = float(np.linalg.norm(next_pos - stack_target))
            reward -= place_dist * 5.0

        return reward

    def _check_terminated(self) -> bool:
        """Terminate when all objects are stacked."""
        return self._n_stacked == self.n_objects
