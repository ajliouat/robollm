"""
Scripted baseline policies for tabletop manipulation.

Hand-coded controllers that use privileged info (exact object/goal positions)
to execute pick-and-place. Used as a comparison baseline for RL agents.
"""

from __future__ import annotations

from enum import Enum, auto

import numpy as np


class Phase(Enum):
    APPROACH = auto()
    DESCEND = auto()
    GRASP = auto()
    LIFT = auto()
    MOVE = auto()
    LOWER = auto()
    RELEASE = auto()
    DONE = auto()


class ScriptedPickPlace:
    """Scripted pick-and-place controller using privileged info.

    Sequences: approach → descend → grasp → lift → move → lower → release.

    Returns actions in the same format as the RL agent: (delta_ee, gripper).
    """

    def __init__(
        self,
        approach_height: float = 0.15,
        grasp_height_offset: float = 0.02,
        lift_height: float = 0.20,
        place_height_offset: float = 0.03,
        position_gain: float = 8.0,
        threshold: float = 0.015,
    ):
        self.approach_height = approach_height
        self.grasp_height_offset = grasp_height_offset
        self.lift_height = lift_height
        self.place_height_offset = place_height_offset
        self.gain = position_gain
        self.threshold = threshold
        self.phase = Phase.APPROACH
        self._grasp_counter = 0

    def reset(self):
        self.phase = Phase.APPROACH
        self._grasp_counter = 0

    def act(self, info: dict) -> np.ndarray:
        """Compute action from environment info dict.

        Args:
            info: dict with 'ee_pos', 'obj_positions' (or 'obj_pos'),
                  and optionally 'goal_pos'.
        Returns:
            action: (4,) array — [delta_ee_x, delta_ee_y, delta_ee_z, gripper]
        """
        ee_pos = info["ee_pos"]

        # Get object position
        if "obj_positions" in info:
            obj_positions = info["obj_positions"]
            obj_pos = next(iter(obj_positions.values()))
        else:
            obj_pos = info.get("obj_pos", np.array([0.5, 0.0, 0.435]))

        goal_pos = info.get("goal_pos", obj_pos + np.array([0.1, 0.0, 0.0]))

        table_z = 0.415
        action = np.zeros(4)

        if self.phase == Phase.APPROACH:
            # Move above object
            target = np.array([
                obj_pos[0], obj_pos[1], table_z + self.approach_height,
            ])
            delta = self._proportional(ee_pos, target)
            action[:3] = delta
            action[3] = 1.0  # open gripper
            if np.linalg.norm(ee_pos[:2] - target[:2]) < self.threshold:
                self.phase = Phase.DESCEND

        elif self.phase == Phase.DESCEND:
            # Lower to grasp height
            target = np.array([
                obj_pos[0], obj_pos[1],
                obj_pos[2] + self.grasp_height_offset,
            ])
            delta = self._proportional(ee_pos, target)
            action[:3] = delta
            action[3] = 1.0  # keep open
            if np.linalg.norm(ee_pos - target) < self.threshold:
                self.phase = Phase.GRASP
                self._grasp_counter = 0

        elif self.phase == Phase.GRASP:
            # Close gripper (hold position)
            action[:3] = 0.0
            action[3] = -1.0  # close
            self._grasp_counter += 1
            if self._grasp_counter >= 5:
                self.phase = Phase.LIFT

        elif self.phase == Phase.LIFT:
            # Lift object
            target = np.array([
                ee_pos[0], ee_pos[1], table_z + self.lift_height,
            ])
            delta = self._proportional(ee_pos, target)
            action[:3] = delta
            action[3] = -1.0  # keep closed
            if ee_pos[2] > table_z + self.lift_height - 0.02:
                self.phase = Phase.MOVE

        elif self.phase == Phase.MOVE:
            # Move above goal
            target = np.array([
                goal_pos[0], goal_pos[1], table_z + self.lift_height,
            ])
            delta = self._proportional(ee_pos, target)
            action[:3] = delta
            action[3] = -1.0  # keep closed
            if np.linalg.norm(ee_pos[:2] - target[:2]) < self.threshold:
                self.phase = Phase.LOWER

        elif self.phase == Phase.LOWER:
            # Lower to place height
            target = np.array([
                goal_pos[0], goal_pos[1],
                goal_pos[2] + self.place_height_offset,
            ])
            delta = self._proportional(ee_pos, target)
            action[:3] = delta
            action[3] = -1.0  # keep closed
            if np.linalg.norm(ee_pos - target) < self.threshold:
                self.phase = Phase.RELEASE

        elif self.phase == Phase.RELEASE:
            action[:3] = 0.0
            action[3] = 1.0  # open
            self.phase = Phase.DONE

        elif self.phase == Phase.DONE:
            action[:3] = 0.0
            action[3] = 1.0

        return np.clip(action, -1.0, 1.0)

    def _proportional(
        self, current: np.ndarray, target: np.ndarray,
    ) -> np.ndarray:
        """P-controller: scale error by gain, clip to [-1, 1]."""
        error = target - current
        return np.clip(error * self.gain, -1.0, 1.0)


class ScriptedMoveTo:
    """Scripted move-to controller — just approach the object."""

    def __init__(self, gain: float = 8.0, threshold: float = 0.03):
        self.gain = gain
        self.threshold = threshold

    def reset(self):
        pass

    def act(self, info: dict) -> np.ndarray:
        ee_pos = info["ee_pos"]
        if "obj_positions" in info:
            obj_pos = next(iter(info["obj_positions"].values()))
        else:
            obj_pos = info.get("obj_pos", np.array([0.5, 0.0, 0.435]))

        error = obj_pos - ee_pos
        action = np.zeros(4)
        action[:3] = np.clip(error * self.gain, -1.0, 1.0)
        action[3] = 1.0  # keep gripper open
        return action
