"""
MuJoCo tabletop environment with a 7-DOF Franka-inspired arm.

Gymnasium-compatible environment for robotic manipulation tasks.
The arm is controlled via delta end-effector commands and a binary
gripper — matching the action space specified in PROJECT_SPEC.md.

Observation (29D):
    joint_pos (7) + joint_vel (7) + gripper (1)
    + ee_pos (3) + ee_quat (4)
    + obj_pos (3) + obj_quat (4)

Action (4D):
    delta_ee_x, delta_ee_y, delta_ee_z  (scaled by DELTA_SCALE)
    gripper: -1 = close, +1 = open
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

# ── Paths ────────────────────────────────────────────────────────────────────

_ASSET_DIR = Path(__file__).parent / "assets"
_SCENE_XML = _ASSET_DIR / "tabletop_scene.xml"

# ── Control parameters ───────────────────────────────────────────────────────

_CTRL_DT = 0.05  # 20 Hz control frequency
_SIM_DT = 0.002  # 500 Hz physics
_N_SUBSTEPS = int(_CTRL_DT / _SIM_DT)  # 25 sim steps per control step
_DELTA_SCALE = 0.05  # max EE displacement per step (m)

# ── Constants ────────────────────────────────────────────────────────────────

_N_ARM_JOINTS = 7
_HOME_QPOS = np.array(
    [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785,  # arm
     0.02, 0.02],                                      # fingers (half open)
    dtype=np.float64,
)
_OBS_DIM = 29  # 7+7+1+3+4+3+4


class TabletopEnv(gym.Env):
    """
    Base MuJoCo tabletop environment.

    A 7-DOF arm on a 60×60 cm table with one block. The agent controls
    the end-effector via delta position commands (3D) and a gripper
    open/close command (1D).

    Subclasses add task-specific rewards and termination conditions.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": int(1 / _CTRL_DT),
    }

    def __init__(
        self,
        render_mode: str | None = None,
        image_size: int = 128,
    ):
        super().__init__()

        if not _SCENE_XML.exists():
            raise FileNotFoundError(f"MuJoCo scene not found: {_SCENE_XML}")

        # ── Load model ───────────────────────────────────────────────────
        self.model = mujoco.MjModel.from_xml_path(str(_SCENE_XML))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self._image_size = image_size
        self._renderer: mujoco.Renderer | None = None

        # ── Cache IDs ────────────────────────────────────────────────────
        self._arm_jnt_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"j{i}")
            for i in range(1, 8)
        ])
        self._finger_l_jnt = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_l",
        )
        self._finger_r_jnt = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_r",
        )
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
        self._block_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "block_red",
        )

        # ── Joint limit cache ────────────────────────────────────────────
        self._jnt_lo = np.array([
            self.model.jnt_range[jid, 0] for jid in self._arm_jnt_ids
        ])
        self._jnt_hi = np.array([
            self.model.jnt_range[jid, 1] for jid in self._arm_jnt_ids
        ])

        # ── Spaces ───────────────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(_OBS_DIM,), dtype=np.float64,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float64,
        )

    # ─── Gymnasium interface ────────────────────────────────────────────────

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Arm → home configuration
        self.data.qpos[:9] = _HOME_QPOS

        # Randomise block position on table surface
        bx = self.np_random.uniform(0.35, 0.65)
        by = self.np_random.uniform(-0.05, 0.20)
        bz = 0.435  # table surface (0.415) + half block height (0.02)
        self.data.qpos[9:12] = [bx, by, bz]
        self.data.qpos[12:16] = [1.0, 0.0, 0.0, 0.0]  # identity quat

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, -1.0, 1.0)

        # Delta EE → joint targets via Jacobian pseudoinverse
        delta_ee = action[:3] * _DELTA_SCALE
        joint_targets = self._delta_ee_to_joints(delta_ee)

        # Gripper: map [-1, 1] → [0, 0.04]
        gripper_pos = 0.02 * (1.0 + action[3])

        # Set actuator controls
        for i, act_id in enumerate(self._arm_act_ids):
            self.data.ctrl[act_id] = joint_targets[i]
        self.data.ctrl[self._finger_l_act] = gripper_pos
        self.data.ctrl[self._finger_r_act] = gripper_pos

        # Substep simulation
        for _ in range(_N_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._check_terminated()
        truncated = False
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
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
        """Build 29-D observation vector."""
        joint_pos = self.data.qpos[:_N_ARM_JOINTS].copy()
        joint_vel = self.data.qvel[:_N_ARM_JOINTS].copy()
        gripper = np.array([
            0.5 * (self.data.qpos[7] + self.data.qpos[8]),
        ])

        ee_pos = self.data.site_xpos[self._ee_site_id].copy()
        ee_mat = self.data.site_xmat[self._ee_site_id]  # (9,)
        ee_quat = np.zeros(4)
        mujoco.mju_mat2Quat(ee_quat, ee_mat)

        obj_pos = self.data.xpos[self._block_body_id].copy()
        obj_quat = self.data.xquat[self._block_body_id].copy()

        return np.concatenate([
            joint_pos, joint_vel, gripper,
            ee_pos, ee_quat,
            obj_pos, obj_quat,
        ])

    def _get_info(self) -> dict:
        ee_pos = self.data.site_xpos[self._ee_site_id].copy()
        obj_pos = self.data.xpos[self._block_body_id].copy()
        return {
            "ee_pos": ee_pos,
            "obj_pos": obj_pos,
            "distance": float(np.linalg.norm(ee_pos - obj_pos)),
        }

    # ─── Control ────────────────────────────────────────────────────────────

    def _delta_ee_to_joints(self, delta_ee: np.ndarray) -> np.ndarray:
        """Convert delta EE position command to joint position targets.

        Uses the Jacobian pseudoinverse (damped least squares) so we get
        smooth, singularity-resistant IK at every control step.
        """
        q_current = self.data.qpos[:_N_ARM_JOINTS].copy()

        # Compute positional Jacobian at EE site
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(
            self.model, self.data, jacp, jacr, self._ee_site_id,
        )
        J = jacp[:, :_N_ARM_JOINTS]  # only arm DOFs

        # Damped pseudoinverse  (J^T (J J^T + λI)^{-1})
        lam = 0.01
        JJT = J @ J.T + lam * np.eye(3)
        J_pinv = J.T @ np.linalg.inv(JJT)

        dq = J_pinv @ delta_ee
        q_target = q_current + dq

        # Clip to joint limits
        q_target = np.clip(q_target, self._jnt_lo, self._jnt_hi)
        return q_target

    # ─── Reward / termination (overridden by subclasses) ────────────────────

    def _compute_reward(self) -> float:
        """Base env: zero reward. Override in task-specific subclasses."""
        return 0.0

    def _check_terminated(self) -> bool:
        """Base env: never terminates. Override in task-specific subclasses."""
        return False

    # ─── Convenience properties ─────────────────────────────────────────────

    @property
    def ee_pos(self) -> np.ndarray:
        """Current end-effector world position (3,)."""
        return self.data.site_xpos[self._ee_site_id].copy()

    @property
    def object_pos(self) -> np.ndarray:
        """Current block world position (3,)."""
        return self.data.xpos[self._block_body_id].copy()
