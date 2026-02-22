"""
Move-To environment — approach a target object without grasping.

The agent must move the end-effector close to a specified object.
This is the simplest primitive — no grasp or lift required.

Reward: negative distance EE → target object.
Terminates when EE is within threshold of the target.
"""

from __future__ import annotations

import numpy as np

from envs.multi_object_env import MultiObjectEnv

_APPROACH_SCALE = 5.0
_SUCCESS_BONUS = 50.0
_SUCCESS_THRESHOLD = 0.03  # 3 cm


class MoveToEnv(MultiObjectEnv):
    """
    Move-to primitive environment.

    One object on the table. Agent must move EE close to it.
    Obs = base (22 + 7*1 = 29D). No augmentation needed.
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

    def _compute_reward(self) -> float:
        ee_pos = self.ee_pos
        obj_pos = self.object_pos(0)
        dist = float(np.linalg.norm(ee_pos - obj_pos))
        reward = -_APPROACH_SCALE * dist
        if dist < _SUCCESS_THRESHOLD:
            reward += _SUCCESS_BONUS
        return reward

    def _check_terminated(self) -> bool:
        dist = float(np.linalg.norm(self.ee_pos - self.object_pos(0)))
        return dist < _SUCCESS_THRESHOLD

    def _get_info(self) -> dict:
        info = super()._get_info()
        dist = float(np.linalg.norm(self.ee_pos - self.object_pos(0)))
        info["success"] = dist < _SUCCESS_THRESHOLD
        info["ee_obj_distance"] = dist
        return info
