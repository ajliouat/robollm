"""
Multi-object tabletop environment.

Extends TabletopEnv with dynamic object spawning: loads the base MJCF,
injects N randomized objects, and provides observation / info helpers
that expose all object states.

This is the base class for PickPlaceEnv, ColorPickEnv, and StackEnv.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from envs.object_spawner import (
    COLOR_NAMES,
    SHAPES,
    ObjectSpec,
    generate_object_specs,
    get_object_qpos_slice,
    inject_objects_into_xml,
)

# ── Paths ────────────────────────────────────────────────────────────────────

_ASSET_DIR = Path(__file__).parent / "assets"
_SCENE_XML = _ASSET_DIR / "tabletop_scene.xml"

# ── Control parameters (same as TabletopEnv) ─────────────────────────────────

_CTRL_DT = 0.05   # 20 Hz
_SIM_DT = 0.002   # 500 Hz
_N_SUBSTEPS = int(_CTRL_DT / _SIM_DT)
_DELTA_SCALE = 0.05  # m per action unit

_N_ARM_JOINTS = 7
_HOME_QPOS = np.array(
    [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.02, 0.02],
    dtype=np.float64,
)


class MultiObjectEnv(gym.Env):
    """
    Tabletop environment with N dynamically-spawned objects.

    Observation layout (variable-length, depends on n_objects):
        joint_pos (7) + joint_vel (7) + gripper (1)
        + ee_pos (3) + ee_quat (4)
        + per-object: obj_pos (3) + obj_quat (4) = 7 each

    Total obs dim = 22 + 7 * n_objects

    Action (4D): delta_ee (3) + gripper (1)  — same as TabletopEnv.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": int(1 / _CTRL_DT),
    }

    def __init__(
        self,
        n_objects: int = 3,
        render_mode: str | None = None,
        image_size: int = 128,
        allowed_shapes: Sequence[str] = SHAPES,
        allowed_colors: Sequence[str] | None = None,
        unique_colors: bool = True,
    ):
        super().__init__()

        if not _SCENE_XML.exists():
            raise FileNotFoundError(f"MuJoCo scene not found: {_SCENE_XML}")

        self.n_objects = n_objects
        self.render_mode = render_mode
        self._image_size = image_size
        self._allowed_shapes = list(allowed_shapes)
        self._allowed_colors = (
            list(allowed_colors) if allowed_colors else None
        )
        self._unique_colors = unique_colors

        # Will be populated on first reset
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self._renderer: mujoco.Renderer | None = None
        self._obj_specs: list[ObjectSpec] = []
        self._obj_body_ids: list[int] = []

        # Cache (set after first model load)
        self._arm_jnt_ids: np.ndarray | None = None
        self._arm_act_ids: np.ndarray | None = None
        self._finger_l_act: int = -1
        self._finger_r_act: int = -1
        self._ee_site_id: int = -1
        self._jnt_lo: np.ndarray | None = None
        self._jnt_hi: np.ndarray | None = None

        # ── Spaces ───────────────────────────────────────────────────────
        obs_dim = 22 + 7 * n_objects
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float64,
        )

    # ─── Model loading ──────────────────────────────────────────────────────

    def _load_scene(self, obj_specs: list[ObjectSpec]) -> None:
        """Build MJCF with injected objects and load into MuJoCo."""
        xml_str = inject_objects_into_xml(
            _SCENE_XML, obj_specs, remove_existing_block=True,
        )
        self.model = mujoco.MjModel.from_xml_string(xml_str)
        self.data = mujoco.MjData(self.model)
        self._obj_specs = obj_specs

        # Close old renderer if model changed
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

        # Cache IDs
        self._arm_jnt_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"j{i}")
            for i in range(1, 8)
        ])
        self._arm_act_ids = np.array([
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_j{i}",
            )
            for i in range(1, 8)
        ])
        self._finger_l_act = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_finger_l",
        )
        self._finger_r_act = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_finger_r",
        )
        self._ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site",
        )
        self._jnt_lo = np.array([
            self.model.jnt_range[jid, 0] for jid in self._arm_jnt_ids
        ])
        self._jnt_hi = np.array([
            self.model.jnt_range[jid, 1] for jid in self._arm_jnt_ids
        ])

        # Object body IDs
        self._obj_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, spec.name)
            for spec in obj_specs
        ]

    # ─── Gymnasium interface ────────────────────────────────────────────────

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        # Generate new randomized objects
        obj_specs = generate_object_specs(
            self.n_objects,
            self.np_random,
            allowed_shapes=self._allowed_shapes,
            allowed_colors=self._allowed_colors,
            unique_colors=self._unique_colors,
        )
        self._load_scene(obj_specs)

        mujoco.mj_resetData(self.model, self.data)

        # Set arm to home
        self.data.qpos[:9] = _HOME_QPOS

        # Set object positions (randomized by spawner)
        for i, spec in enumerate(self._obj_specs):
            sl = get_object_qpos_slice(i)
            self.data.qpos[sl] = np.array([
                *spec.init_pos, 1.0, 0.0, 0.0, 0.0,
            ])

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, -1.0, 1.0)

        # Delta EE → joint targets
        delta_ee = action[:3] * _DELTA_SCALE
        joint_targets = self._delta_ee_to_joints(delta_ee)

        # Gripper: [-1, 1] → [0, 0.04]
        gripper_pos = 0.02 * (1.0 + action[3])

        for i, act_id in enumerate(self._arm_act_ids):
            self.data.ctrl[act_id] = joint_targets[i]
        self.data.ctrl[self._finger_l_act] = gripper_pos
        self.data.ctrl[self._finger_r_act] = gripper_pos

        for _ in range(_N_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._check_terminated()
        truncated = False
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None or self.model is None:
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model, self._image_size, self._image_size,
            )
        self._renderer.update_scene(self.data, camera="overhead")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ─── Observation / info ─────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """Build observation: arm state + EE pose + all object poses."""
        joint_pos = self.data.qpos[:_N_ARM_JOINTS].copy()
        joint_vel = self.data.qvel[:_N_ARM_JOINTS].copy()
        gripper = np.array([
            0.5 * (self.data.qpos[7] + self.data.qpos[8]),
        ])

        ee_pos = self.data.site_xpos[self._ee_site_id].copy()
        ee_mat = self.data.site_xmat[self._ee_site_id]
        ee_quat = np.zeros(4)
        mujoco.mju_mat2Quat(ee_quat, ee_mat)

        parts = [joint_pos, joint_vel, gripper, ee_pos, ee_quat]

        # All objects
        for bid in self._obj_body_ids:
            parts.append(self.data.xpos[bid].copy())
            parts.append(self.data.xquat[bid].copy())

        return np.concatenate(parts)

    def _get_info(self) -> dict:
        """Return EE/object positions, distances, and object metadata."""
        ee_pos = self.data.site_xpos[self._ee_site_id].copy()

        obj_positions = {}
        obj_distances = {}
        for i, (bid, spec) in enumerate(
            zip(self._obj_body_ids, self._obj_specs)
        ):
            pos = self.data.xpos[bid].copy()
            obj_positions[spec.name] = pos
            obj_distances[spec.name] = float(np.linalg.norm(ee_pos - pos))

        return {
            "ee_pos": ee_pos,
            "obj_positions": obj_positions,
            "obj_distances": obj_distances,
            "obj_specs": self._obj_specs,
            "n_objects": self.n_objects,
        }

    # ─── Control ────────────────────────────────────────────────────────────

    def _delta_ee_to_joints(self, delta_ee: np.ndarray) -> np.ndarray:
        """Damped Jacobian pseudoinverse IK (same as TabletopEnv)."""
        q_current = self.data.qpos[:_N_ARM_JOINTS].copy()

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(
            self.model, self.data, jacp, jacr, self._ee_site_id,
        )
        J = jacp[:, :_N_ARM_JOINTS]

        lam = 0.01
        JJT = J @ J.T + lam * np.eye(3)
        J_pinv = J.T @ np.linalg.inv(JJT)

        dq = J_pinv @ delta_ee
        q_target = q_current + dq
        q_target = np.clip(q_target, self._jnt_lo, self._jnt_hi)
        return q_target

    # ─── Reward / termination (overridden by subclasses) ────────────────────

    def _compute_reward(self) -> float:
        return 0.0

    def _check_terminated(self) -> bool:
        return False

    # ─── Convenience ────────────────────────────────────────────────────────

    @property
    def ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._ee_site_id].copy()

    def object_pos(self, index: int = 0) -> np.ndarray:
        """World position of object at *index*."""
        return self.data.xpos[self._obj_body_ids[index]].copy()

    def object_name(self, index: int = 0) -> str:
        return self._obj_specs[index].name

    def object_color(self, index: int = 0) -> str:
        return self._obj_specs[index].color_name

    def object_shape(self, index: int = 0) -> str:
        return self._obj_specs[index].shape
