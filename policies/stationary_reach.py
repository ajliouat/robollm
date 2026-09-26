"""Bounded, privileged-state joint planning for a stationary tabletop hover.

This controller uses the existing model/servos; it never changes live physics.
A finite multi-start IK search screens endpoints and straight joint paths with
3mm collision margins on a private model. MuJoCo's collision masks and adjacent
body exclusions still apply. Discrete planning checks are not a continuous-path
or hardware guarantee: execution needs the independent 500Hz safety monitor.
"""
from __future__ import annotations

import copy
from math import ceil, sqrt
from numbers import Integral

import mujoco
import numpy as np


class StationaryReachController:
    """Plan once from the current reset; return seven position-servo targets.

    ``metadata`` is JSON-safe and separates no-plan/tracking failure from a
    feasible plan. ``command`` only reads the live environment. Failed plans
    return a hold command so callers can record a failure without resampling.
    The caller owns stepping, open-finger commands and the task/safety oracle.
    """

    N_STARTS = 64
    IK_ITERATIONS = 250
    IK_TOLERANCE_M = 0.001
    IK_DAMPING = 1e-4
    IK_MAX_JOINT_STEP = 0.12
    JOINT_LIMIT_MARGIN = 0.002
    COLLISION_MARGIN_M = 0.003
    PATH_RESOLUTION_RAD = 0.02
    CONTROL_DT = 0.05
    OPENING_HOLD_SECONDS = 0.25
    MAX_JOINT_SPEED = 0.8
    MAX_JOINT_ACCELERATION = 1.0
    MIN_MOTION_SECONDS = 2.0
    MAX_MOTION_SECONDS = 7.5

    def __init__(self, env, goal, seed: int = 1729):
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("Planner seed must be a non-negative integer")
        self.env = env
        self.goal = np.asarray(goal, dtype=float).copy()
        if self.goal.shape != (3,) or not np.isfinite(self.goal).all():
            raise ValueError("Goal must be a finite 3-vector")
        self._live_model = env.model
        self._model = copy.copy(env.model)
        self._data = mujoco.MjData(self._model)
        self._initial_qpos = env.data.qpos.copy()
        self._data.qpos[:] = self._initial_qpos
        joint_ids = np.asarray(env._arm_jnt_ids, dtype=int)
        self._qpos_ids = self._model.jnt_qposadr[joint_ids]
        self._dof_ids = self._model.jnt_dofadr[joint_ids]
        self._home = self._initial_qpos[self._qpos_ids].copy()
        self._lo = self._model.jnt_range[joint_ids, 0].copy()
        self._hi = self._model.jnt_range[joint_ids, 1].copy()
        self._site = int(env._ee_site_id)
        self._start_time = float(env.data.time)
        if len(joint_ids) != 7 or not np.isfinite(self._home).all():
            raise ValueError("A finite seven-joint arm state is required")
        self._robot_bodies = {int(self._model.jnt_bodyid[joint_ids[0]])}
        for body in range(self._model.nbody):
            if int(self._model.body_parentid[body]) in self._robot_bodies:
                self._robot_bodies.add(body)
        robot_geoms = np.isin(self._model.geom_bodyid, list(self._robot_bodies))
        self._model.geom_margin[robot_geoms] = np.maximum(
            self._model.geom_margin[robot_geoms], self.COLLISION_MARGIN_M,
        )
        self._finger_qpos_ids = [
            int(self._model.jnt_qposadr[self._model.actuator_trnid[act, 0]])
            for act in (env._finger_l_act, env._finger_r_act)
        ]
        self._data.qpos[self._finger_qpos_ids] = 0.04
        self._jacp = np.zeros((3, self._model.nv))
        self._jacr = np.zeros_like(self._jacp)
        self._target = self._home.copy()
        self._duration = 0.0
        self.metadata = {
            "planner": "multistart_ik_straight_joint_path",
            "seed": int(seed), "planning_status": "planning", "failure": None,
            "budget": {"starts": self.N_STARTS, "iterations_per_start": self.IK_ITERATIONS,
                       "ik_tolerance_m": self.IK_TOLERANCE_M,
                       "ik_damping": self.IK_DAMPING,
                       "ik_max_joint_step_rad": self.IK_MAX_JOINT_STEP,
                       "joint_limit_margin_rad": self.JOINT_LIMIT_MARGIN,
                       "collision_margin_m": self.COLLISION_MARGIN_M,
                       "path_resolution_rad": self.PATH_RESOLUTION_RAD,
                       "control_dt_seconds": self.CONTROL_DT,
                       "opening_hold_seconds": self.OPENING_HOLD_SECONDS,
                       "max_joint_speed_rad_s": self.MAX_JOINT_SPEED,
                       "max_joint_acceleration_rad_s2": self.MAX_JOINT_ACCELERATION,
                       "min_motion_seconds": self.MIN_MOTION_SECONDS,
                       "max_motion_seconds": self.MAX_MOTION_SECONDS},
            "starts_attempted": 0, "ik_iterations": 0, "ik_solutions": 0,
            "collision_free_endpoints": 0, "collision_free_paths": 0,
            "collision_checks": 0, "commands": 0,
            "goal": self.goal.tolist(), "initial_joints": self._home.tolist(),
            "selected_joints": None, "motion_seconds": None,
        }
        self._plan(np.random.default_rng(int(seed)))

    def _collision_free(self, q) -> bool:
        self.metadata["collision_checks"] += 1
        self._data.qpos[self._qpos_ids] = q
        mujoco.mj_forward(self._model, self._data)
        return not any(
            int(self._model.geom_bodyid[c.geom1]) in self._robot_bodies
            or int(self._model.geom_bodyid[c.geom2]) in self._robot_bodies
            for c in self._data.contact
        )

    def _path_free(self, start, end) -> bool:
        n = max(1, ceil(float(np.max(np.abs(end - start))) / self.PATH_RESOLUTION_RAD))
        return all(self._collision_free(start + (end - start) * i / n)
                   for i in range(n + 1))

    def _ik(self, start):
        q = start.copy()
        for _ in range(self.IK_ITERATIONS):
            self.metadata["ik_iterations"] += 1
            self._data.qpos[self._qpos_ids] = q
            mujoco.mj_forward(self._model, self._data)
            error = self.goal - self._data.site_xpos[self._site]
            if np.linalg.norm(error) < self.IK_TOLERANCE_M:
                return q
            mujoco.mj_jacSite(self._model, self._data, self._jacp, self._jacr, self._site)
            jacobian = self._jacp[:, self._dof_ids]
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + self.IK_DAMPING * np.eye(3), error,
            )
            scale = min(1.0, self.IK_MAX_JOINT_STEP / max(float(np.max(np.abs(delta))), 1e-12))
            q = np.clip(q + scale * delta,
                        self._lo + self.JOINT_LIMIT_MARGIN, self._hi - self.JOINT_LIMIT_MARGIN)
        return None

    def _motion_duration(self, q) -> float:
        distance = float(np.max(np.abs(q - self._home)))
        # Exact maxima of the quintic 10t^3-15t^4+6t^5 interpolation.
        seconds = max(self.MIN_MOTION_SECONDS, 1.875 * distance / self.MAX_JOINT_SPEED,
                      sqrt((10 / sqrt(3)) * distance / self.MAX_JOINT_ACCELERATION))
        return ceil(seconds / self.CONTROL_DT) * self.CONTROL_DT

    def _plan(self, rng):
        if not self._collision_free(self._home):
            self.metadata.update(planning_status="failed", failure="initial_clearance")
            return
        best = None
        for attempt in range(self.N_STARTS):
            self.metadata["starts_attempted"] += 1
            start = self._home.copy() if attempt == 0 else rng.uniform(self._lo, self._hi)
            q = self._ik(start)
            if q is None:
                continue
            self.metadata["ik_solutions"] += 1
            if not self._collision_free(q):
                continue
            self.metadata["collision_free_endpoints"] += 1
            duration = self._motion_duration(q)
            if duration > self.MAX_MOTION_SECONDS or not self._path_free(self._home, q):
                continue
            self.metadata["collision_free_paths"] += 1
            score = float(np.linalg.norm(q - self._home))
            if best is None or score < best[0]:
                best = (score, q.copy(), duration, attempt)
        if best is None:
            self.metadata.update(planning_status="failed", failure="no_collision_free_path")
            return
        _, self._target, self._duration, attempt = best
        self.metadata.update(planning_status="ready", selected_joints=self._target.tolist(),
                             motion_seconds=self._duration, selected_restart=attempt)

    def command(self) -> np.ndarray:
        self.metadata["commands"] += 1
        if self.env.model is not self._live_model:
            raise ValueError("Environment was reset after planning")
        actual = self.env.data.qpos[self._qpos_ids].copy()
        if not np.isfinite(actual).all():
            raise ValueError("Nonfinite observed arm state")
        now = float(self.env.data.time)
        if not np.isfinite(now) or now < self._start_time:
            raise ValueError("Invalid simulator time after planning")
        if self.metadata["planning_status"] != "ready":
            return actual
        elapsed = max(0., now - self._start_time - self.OPENING_HOLD_SECONDS)
        phase = min(1., elapsed / self._duration)
        blend = phase ** 3 * (10. + phase * (-15. + 6. * phase))
        target = self._home + blend * (self._target - self._home)
        # Refresh the planning clone from observed state before checking the next
        # short connector. This guards changed object poses and tracking drift;
        # actual physics substeps remain the independent monitor's responsibility.
        self._data.qpos[:] = self.env.data.qpos
        self._data.qpos[self._finger_qpos_ids] = 0.04
        if not self._path_free(actual, target):
            self.metadata.update(planning_status="failed", failure="tracking_clearance")
            return actual
        return target.copy()
