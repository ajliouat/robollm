"""
Complex language environment (L5) — multi-step instruction execution.

Presents a scene with multiple objects and a complex natural language
instruction that requires multiple sub-tasks. The environment scores
execution of the full instruction by checking final state conditions.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from envs.multi_object_env import MultiObjectEnv

_TABLE_Z = 0.415
_CONDITION_THRESHOLD = 0.05  # 5 cm


# ── Instruction templates ─────────────────────────────────────────

INSTRUCTION_TEMPLATES = [
    {
        "template": "pick up the {color0} {shape0} and place it next to the {color1} {shape1}",
        "n_objects": 2,
        "conditions": ["obj0_near_obj1"],
    },
    {
        "template": "move the {color0} {shape0} to the left side of the table",
        "n_objects": 2,
        "conditions": ["obj0_left"],
    },
    {
        "template": "stack the {color0} {shape0} on top of the {color1} {shape1}",
        "n_objects": 2,
        "conditions": ["obj0_on_obj1"],
    },
    {
        "template": "move all objects to the center of the table",
        "n_objects": 3,
        "conditions": ["all_center"],
    },
    {
        "template": "pick up the {color0} {shape0} and the {color1} {shape1}, then place them together",
        "n_objects": 3,
        "conditions": ["obj0_near_obj1"],
    },
]


class ComplexLanguageEnv(MultiObjectEnv):
    """
    L5: Complex natural language instruction environment.

    Each episode samples a random instruction template, fills it with
    actual object colors/shapes, and defines success conditions.

    Obs augmented with instruction embedding placeholder (+16D).
    The actual instruction text is in the info dict.
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
        # Augment obs with instruction encoding (16D placeholder)
        base_dim = self.observation_space.shape[0]
        from gymnasium import spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(base_dim + 16,),
            dtype=np.float64,
        )
        self._instruction: str = ""
        self._conditions: list[str] = []
        self._instruction_encoding: np.ndarray = np.zeros(16)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._instruction = ""
        self._conditions = []
        self._instruction_encoding = np.zeros(16)
        obs, info = super().reset(seed=seed, options=options)

        # Pick a random instruction template
        template_idx = int(self.np_random.integers(len(INSTRUCTION_TEMPLATES)))
        template = INSTRUCTION_TEMPLATES[template_idx]

        # Fill template with actual object colors/shapes
        fmt_kwargs = {}
        for i in range(min(template["n_objects"], self.n_objects)):
            spec = self._obj_specs[i]
            fmt_kwargs[f"color{i}"] = spec.color_name
            fmt_kwargs[f"shape{i}"] = spec.shape

        self._instruction = template["template"].format(**fmt_kwargs)
        self._conditions = list(template["conditions"])

        # Simple instruction encoding (hash-based placeholder)
        hash_val = hash(self._instruction) % (2**31)
        rng = np.random.RandomState(hash_val)
        self._instruction_encoding = rng.randn(16).astype(np.float64)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def _get_obs(self) -> np.ndarray:
        base_obs = super()._get_obs()
        return np.concatenate([base_obs, self._instruction_encoding])

    def _compute_reward(self) -> float:
        reward = 0.0
        n_met = 0

        for cond in self._conditions:
            met = self._check_condition(cond)
            if met:
                n_met += 1
                reward += 20.0

        # Penalty proportional to unmet conditions
        reward -= 5.0 * (len(self._conditions) - n_met)

        # All conditions met bonus
        if n_met == len(self._conditions):
            reward += 100.0

        return reward

    def _check_terminated(self) -> bool:
        return all(self._check_condition(c) for c in self._conditions)

    def _check_condition(self, condition: str) -> bool:
        """Check if a specific condition is met."""
        if condition == "obj0_near_obj1" and self.n_objects >= 2:
            d = float(np.linalg.norm(
                self.object_pos(0)[:2] - self.object_pos(1)[:2]))
            return d < _CONDITION_THRESHOLD

        elif condition == "obj0_left":
            return float(self.object_pos(0)[1]) < -0.05

        elif condition == "obj0_on_obj1" and self.n_objects >= 2:
            pos0 = self.object_pos(0)
            pos1 = self.object_pos(1)
            xy_close = float(np.linalg.norm(pos0[:2] - pos1[:2])) < 0.03
            z_above = pos0[2] > pos1[2] + 0.02
            return xy_close and z_above

        elif condition == "all_center":
            center = np.array([0.5, 0.05])
            for i in range(self.n_objects):
                d = float(np.linalg.norm(
                    self.object_pos(i)[:2] - center))
                if d >= _CONDITION_THRESHOLD:
                    return False
            return True

        return False

    def _get_info(self) -> dict:
        info = super()._get_info()
        conditions_met = sum(
            1 for c in self._conditions if self._check_condition(c))
        info["instruction"] = self._instruction
        info["conditions"] = self._conditions
        info["conditions_met"] = conditions_met
        info["conditions_total"] = len(self._conditions)
        info["success"] = conditions_met == len(self._conditions)
        return info
