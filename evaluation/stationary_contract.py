"""Fixed-goal reaching/holding contract sampled at 500 Hz.

The object-pose bound is conservative, including for symmetric objects. This
does not certify collision freedom or continuity between simulator samples.
The observer refreshes a private MjData, leaving the live dynamics untouched.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np


GOAL_CLEARANCE = 0.10
GOAL_RADIUS = 0.02
PHYSICS_DT = 0.002
HOLD_INTERVALS = 500
HORIZON_INTERVALS = 5000
MAX_OBJECT_POSE_DISPLACEMENT = 0.005
MOVING_ROBOT_BODIES = frozenset(
    [f"link{i}" for i in range(1, 8)] + ["gripper_base", "finger_l", "finger_r"]
)


def _vector(value, length: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ValueError(f"Expected a finite vector of length {length}")
    return result


def _quaternion(value) -> np.ndarray:
    result = _vector(value, 4)
    norm = float(np.linalg.norm(result))
    if not math.isclose(norm, 1., rel_tol=0., abs_tol=1e-6):
        raise ValueError("Expected a unit quaternion")
    return result / norm


def pose_displacement_bound(initial: dict, current: dict) -> float:
    """Upper bound on a material point's displacement about the body origin."""
    radius = float(initial["radius_bound_m"])
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("Object radius bound must be finite and positive")
    if current["name"] != initial["name"] or current["radius_bound_m"] != radius:
        raise ValueError("Object identity or geometry changed during the episode")
    translation = np.linalg.norm(_vector(current["position"], 3) - _vector(initial["position"], 3))
    dot = min(1., abs(float(_quaternion(initial["quaternion"]) @ _quaternion(current["quaternion"]))))
    # abs(dot) makes q and -q equivalent; this is 2*r*sin(angle/2).
    rotation = 2. * radius * math.sqrt(max(0., 1. - dot * dot))
    return float(translation + rotation)


def _shape_extent(kind: int, size: np.ndarray, rotation: np.ndarray) -> tuple[float, float]:
    """Return exact world-Z half extent and conservative body-centred radius."""
    import mujoco

    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        return float(np.abs(rotation[2]) @ size), float(np.linalg.norm(size))
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        return float(size[0]), float(size[0])
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        vertical = size[0] * np.linalg.norm(rotation[2, :2]) + size[1] * abs(rotation[2, 2])
        return float(vertical), float(np.hypot(size[0], size[1]))
    raise ValueError("Stationary contract supports box, cylinder and sphere objects only")


class MujocoObserver:
    """Observe a reset MultiObjectEnv without modifying its integration state.

    Contacts involving moving robot bodies are classified as potentially
    forbidden. Static mounting contacts, object/table and object/object pairs
    remain allowed. The monitor applies the signed-distance criterion.
    """

    def __init__(self, env):
        import mujoco

        if env.model is None or env.data is None:
            raise ValueError("Reset the environment before creating its observer")
        self.env, self.model = env, env.model
        self.data = mujoco.MjData(self.model)
        self.ee_site_id = int(env._ee_site_id)
        self.object_body_ids = tuple(int(i) for i in env._obj_body_ids)
        if not self.object_body_ids or len(set(self.object_body_ids)) != len(self.object_body_ids):
            raise ValueError("Expected distinct object bodies")
        self.body_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
                           for i in range(self.model.nbody)]
        self.geom_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
                           for i in range(self.model.ngeom)]
        if not MOVING_ROBOT_BODIES.issubset(set(self.body_names)):
            raise ValueError("The expected repaired robot bodies are missing")
        self.object_geom_ids = [np.flatnonzero(self.model.geom_bodyid == body_id)
                                for body_id in self.object_body_ids]
        if any(len(ids) == 0 for ids in self.object_geom_ids):
            raise ValueError("Each object must have compiled geometry")

    def sample(self) -> dict:
        import mujoco

        if self.env.model is not self.model:
            raise RuntimeError("Create a new observer after an environment reset")
        # mj_step advances qpos after computing derived geometry. Refresh only
        # the copy so scoring refers to the actual state at the recorded time.
        mujoco.mj_copyData(self.data, self.model, self.env.data)
        mujoco.mj_forward(self.model, self.data)
        data, model = self.data, self.model
        if not all(np.all(np.isfinite(value)) for value in (data.qpos, data.qvel, data.qacc, data.ctrl)):
            raise ValueError("Nonfinite simulator state")
        objects = []
        for body_id, geom_ids in zip(self.object_body_ids, self.object_geom_ids):
            tops, radii = [], []
            for geom_id in geom_ids:
                extent, radius = _shape_extent(int(model.geom_type[geom_id]), model.geom_size[geom_id],
                                               data.geom_xmat[geom_id].reshape(3, 3))
                tops.append(float(data.geom_xpos[geom_id, 2]) + extent)
                radii.append(float(np.linalg.norm(model.geom_pos[geom_id])) + radius)
            objects.append({"name": self.body_names[body_id],
                            "position": data.xpos[body_id].tolist(),
                            "quaternion": data.xquat[body_id].tolist(),
                            "radius_bound_m": max(radii), "top_z_m": max(tops)})
        contacts = []
        for contact in data.contact[:data.ncon]:
            first, second = int(contact.geom1), int(contact.geom2)
            body1, body2 = self.body_names[model.geom_bodyid[first]], self.body_names[model.geom_bodyid[second]]
            contacts.append({"geom1": self.geom_names[first], "geom2": self.geom_names[second],
                             "body1": body1, "body2": body2,
                             "moving_robot": body1 in MOVING_ROBOT_BODIES or body2 in MOVING_ROBOT_BODIES,
                             "signed_distance_m": float(contact.dist)})
        return {"time_seconds": float(data.time), "ee_position": data.site_xpos[self.ee_site_id].tolist(),
                "objects": objects, "contacts": contacts}


class StationaryMonitor:
    """Pure deterministic metric monitor; pass consecutive 2 ms samples.

    Sample 0 is supplied at construction. A run starting inside the radius
    still needs 500 subsequent intervals; arriving at sample k needs safety
    and proximity through sample k+500. Any safety violation ends the episode.
    """

    def __init__(self, initial_sample: dict, target_index: int):
        self.initial = deepcopy(initial_sample)
        objects = self.initial["objects"]
        if type(target_index) is not int or not 0 <= target_index < len(objects):
            raise ValueError("Target index must identify an initial object")
        names = [obj["name"] for obj in objects]
        if len(set(names)) != len(names):
            raise ValueError("Object names must be unique")
        target = objects[target_index]
        self._goal = _vector(target["position"], 3).copy()
        self._goal[2] = float(target["top_z_m"]) + GOAL_CLEARANCE
        _vector(self._goal, 3)
        self._start_time = float(initial_sample["time_seconds"])
        if not math.isfinite(self._start_time):
            raise ValueError("Initial simulation time must be finite")
        self._index = -1
        self._first_inside = None
        self._max_pose_bound = 0.
        self._min_distance = math.inf
        self._initially_inside = bool(np.linalg.norm(_vector(initial_sample["ee_position"], 3) - self._goal) < GOAL_RADIUS)
        self._result = {"done": False}
        self._consume(0, self.initial)

    @property
    def goal(self) -> list[float]:
        return self._goal.tolist()

    @property
    def result(self) -> dict:
        return deepcopy(self._result)

    def consume(self, sample_index: int, sample: dict) -> dict:
        if self._result["done"]:
            raise RuntimeError("Cannot consume samples after the episode has ended")
        return self._consume(sample_index, sample)

    def _consume(self, sample_index: int, sample: dict) -> dict:
        if type(sample_index) is not int or sample_index != self._index + 1 or sample_index > HORIZON_INTERVALS:
            raise ValueError("Expected the next consecutive physics sample index")
        timestamp = float(sample["time_seconds"])
        if not math.isfinite(timestamp) or not math.isclose(
                timestamp, self._start_time + sample_index * PHYSICS_DT, rel_tol=0., abs_tol=1e-9):
            raise ValueError("Sample time disagrees with the 2 ms protocol")
        ee = _vector(sample["ee_position"], 3)
        if len(sample["objects"]) != len(self.initial["objects"]):
            raise ValueError("Object count changed during the episode")
        bounds = [pose_displacement_bound(old, new) for old, new in zip(self.initial["objects"], sample["objects"])]
        violations = []
        for contact in sample["contacts"]:
            distance = float(contact["signed_distance_m"])
            if not math.isfinite(distance) or type(contact["moving_robot"]) is not bool:
                raise ValueError("Invalid contact sample")
            if contact["moving_robot"] and distance <= 0.:
                violations.append({"type": "forbidden_contact", "contact": deepcopy(contact)})
        for obj, bound in zip(sample["objects"], bounds):
            if bound > MAX_OBJECT_POSE_DISPLACEMENT:
                violations.append({"type": "object_motion", "object_name": obj["name"], "bound_m": bound})
        distance = float(np.linalg.norm(ee - self._goal))
        self._min_distance = min(self._min_distance, distance)
        if distance >= GOAL_RADIUS:
            self._first_inside = None
        elif self._first_inside is None:
            self._first_inside = sample_index
        dwell = 0 if self._first_inside is None else sample_index - self._first_inside
        self._max_pose_bound = max(self._max_pose_bound, *bounds)
        # Safety takes priority even on the first completed hold or last sample.
        success = not violations and dwell >= HOLD_INTERVALS
        stop = (violations[0]["type"] if violations else "held_goal" if success
                else "horizon" if sample_index == HORIZON_INTERVALS else None)
        self._index = sample_index
        self._result = {"sample_index": sample_index, "time_seconds": timestamp,
                        "current_distance_m": distance, "min_distance_m": self._min_distance,
                        "goal_position": self.goal,
                        "pose_displacement_bounds_m": bounds,
                        "max_motion_bound_m": self._max_pose_bound,
                        "initially_within_tolerance": self._initially_inside,
                        "first_in_radius_sample": self._first_inside,
                        "dwell_intervals": dwell, "dwell_seconds": dwell * PHYSICS_DT,
                        "success": bool(success), "done": stop is not None,
                        "stop_reason": stop, "violations": violations}
        return self.result
