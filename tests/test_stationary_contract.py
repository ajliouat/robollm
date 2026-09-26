"""Adversarial metric checks and development-only observer checks."""
from copy import deepcopy
import math

import mujoco
import numpy as np
import pytest

from envs.multi_object_env import MultiObjectEnv
from evaluation.stationary_contract import (
    GOAL_CLEARANCE, GOAL_RADIUS, HOLD_INTERVALS, HORIZON_INTERVALS,
    PHYSICS_DT, MujocoObserver, StationaryMonitor, pose_displacement_bound,
)


def sample(index=0, ee=(0., 0., 0.)):
    return {"time_seconds": index * PHYSICS_DT, "ee_position": list(ee),
            "objects": [{"name": f"object{i}", "position": [float(i), 0., -0.12],
                         "quaternion": [1., 0., 0., 0.], "radius_bound_m": 0.03,
                         "top_z_m": -GOAL_CLEARANCE} for i in range(3)],
            "contacts": []}


def contact(*, moving=True, distance=0.):
    return {"geom1": "link2_geom" if moving else "object_geom", "geom2": "table_top",
            "body1": "link2" if moving else "object0", "body2": "table",
            "moving_robot": moving, "signed_distance_m": distance}


def test_initial_inside_requires_five_hundred_elapsed_intervals():
    monitor = StationaryMonitor(sample(), 0)
    assert monitor.result["initially_within_tolerance"] and not monitor.result["success"]
    for index in range(1, HOLD_INTERVALS):
        assert not monitor.consume(index, sample(index))["done"]
    result = monitor.consume(500, sample(500))
    assert result["success"] and result["dwell_intervals"] == 500
    assert result["dwell_seconds"] == 1.


def test_arriving_after_reset_requires_501_qualifying_samples():
    monitor = StationaryMonitor(sample(ee=(0.1, 0., 0.)), 0)
    for index in range(1, 501):
        assert not monitor.consume(index, sample(index))["success"]
    assert monitor.result["dwell_intervals"] == 499
    assert monitor.consume(501, sample(501))["success"]


def test_transient_distance_excursion_resets_dwell():
    monitor = StationaryMonitor(sample(), 0)
    for index in range(1, 251):
        monitor.consume(index, sample(index))
    assert monitor.consume(251, sample(251, (GOAL_RADIUS, 0., 0.)))["dwell_intervals"] == 0
    for index in range(252, 752):
        assert not monitor.consume(index, sample(index))["success"]
    assert monitor.consume(752, sample(752))["success"]


@pytest.mark.parametrize("violation", ["contact", "motion"])
def test_safety_has_priority_on_the_success_sample(violation):
    monitor = StationaryMonitor(sample(), 0)
    for index in range(1, 500):
        monitor.consume(index, sample(index))
    last = sample(500)
    if violation == "contact":
        last["contacts"] = [contact()]
    else:
        last["objects"][2]["position"][1] = 0.006
    result = monitor.consume(500, last)
    assert result["done"] and not result["success"]
    assert result["stop_reason"] == ("forbidden_contact" if violation == "contact" else "object_motion")


def test_initial_contact_is_failure_without_actions():
    initial = sample()
    initial["contacts"] = [contact()]
    result = StationaryMonitor(initial, 0).result
    assert result["sample_index"] == 0 and result["done"] and not result["success"]


def test_transient_forbidden_contact_cannot_be_erased_by_a_later_safe_sample():
    monitor = StationaryMonitor(sample(), 0)
    first = sample(1)
    first["contacts"] = [contact(distance=-1e-12)]
    assert monitor.consume(1, first)["stop_reason"] == "forbidden_contact"
    with pytest.raises(RuntimeError, match="ended"):
        monitor.consume(2, sample(2))


def test_allowed_object_contact_and_separated_robot_contact_do_not_fail():
    initial = sample()
    initial["contacts"] = [contact(moving=False, distance=-0.0001), contact(distance=1e-12)]
    assert not StationaryMonitor(initial, 0).result["done"]


def test_all_objects_and_translation_boundary_are_checked():
    monitor = StationaryMonitor(sample(), 0)
    first = sample(1)
    first["objects"][2]["position"][1] = 0.005
    assert not monitor.consume(1, first)["done"]
    second = sample(2)
    second["objects"][2]["position"][1] = 0.005001
    result = monitor.consume(2, second)
    assert result["stop_reason"] == "object_motion"
    assert result["violations"][0]["object_name"] == "object2"


def test_rotation_is_counted_and_quaternion_sign_is_equivalent():
    initial = sample()["objects"][0]
    current = deepcopy(initial)
    current["quaternion"] = [-1., 0., 0., 0.]
    assert pose_displacement_bound(initial, current) == 0.
    angle = 0.2
    current["quaternion"] = [math.cos(angle / 2), 0., 0., math.sin(angle / 2)]
    expected = 2 * initial["radius_bound_m"] * math.sin(angle / 2)
    assert pose_displacement_bound(initial, current) == pytest.approx(expected)
    monitor = StationaryMonitor(sample(), 0)
    rotated = sample(1)
    rotated["objects"][0] = current
    assert monitor.consume(1, rotated)["stop_reason"] == "object_motion"


def test_translation_and_rotation_bounds_add():
    initial = sample()["objects"][0]
    current = deepcopy(initial)
    current["position"][0] += .003
    current["quaternion"] = [math.cos(.05), math.sin(.05), 0., 0.]
    assert pose_displacement_bound(initial, current) == pytest.approx(.003 + .06 * math.sin(.05))


@pytest.mark.parametrize("arrival,success", [(4500, True), (4501, False)])
def test_horizon_counts_hold_time_and_final_sample_can_complete_it(arrival, success):
    monitor = StationaryMonitor(sample(ee=(.1, 0., 0.)), 0)
    for index in range(1, HORIZON_INTERVALS + 1):
        ee = (.1, 0., 0.) if index < arrival else (0., 0., 0.)
        result = monitor.consume(index, sample(index, ee))
    assert result["success"] is success
    assert result["stop_reason"] == ("held_goal" if success else "horizon")


@pytest.mark.parametrize("index", [0, 2, 1.0, True])
def test_missing_duplicate_or_noninteger_samples_are_rejected(index):
    with pytest.raises(ValueError, match="consecutive"):
        StationaryMonitor(sample(), 0).consume(index, sample(1))


@pytest.mark.parametrize("field", ["time", "ee", "quaternion", "contact", "identity", "geometry"])
def test_invalid_or_changed_observations_fail_closed(field):
    monitor = StationaryMonitor(sample(), 0)
    current = sample(1)
    if field == "time":
        current["time_seconds"] = .05
    elif field == "ee":
        current["ee_position"][0] = math.nan
    elif field == "quaternion":
        current["objects"][0]["quaternion"] = [0., 0., 0., 0.]
    elif field == "contact":
        current["contacts"] = [contact(distance=math.inf)]
    elif field == "identity":
        current["objects"][0]["name"] = "different"
    else:
        current["objects"][0]["radius_bound_m"] = .031
    with pytest.raises(ValueError):
        monitor.consume(1, current)


def test_fixed_goal_and_initial_snapshot_do_not_alias_caller_state():
    initial = sample(ee=(.1, 0., 0.))
    monitor = StationaryMonitor(initial, 0)
    initial["objects"][0]["top_z_m"] = 5.
    monitor.goal[0] = 12.
    current = sample(1, (.03, 0., 0.))
    current["objects"][0]["top_z_m"] = 8.
    result = monitor.consume(1, current)
    assert result["goal_position"] == [0., 0., 0.]
    assert result["min_distance_m"] == pytest.approx(.03)


def integration_state(env):
    signature = mujoco.mjtState.mjSTATE_INTEGRATION
    values = np.zeros(mujoco.mj_stateSize(env.model, signature))
    mujoco.mj_getState(env.model, env.data, values, signature)
    return values


@pytest.fixture
def env():
    instance = MultiObjectEnv(n_objects=3)
    instance.reset(seed=17)
    try:
        yield instance
    finally:
        instance.close()


def test_observer_refreshes_geometry_without_altering_live_integration_state(env):
    observer = MujocoObserver(env)
    initial = observer.sample()
    body = env._obj_body_ids[0]
    address = env.model.jnt_qposadr[env.model.body_jntadr[body]]
    old_position = env.data.xpos[body].copy()
    # Deliberately leave live derived positions stale, as integration does.
    env.data.qpos[address] += .001
    before = integration_state(env)
    current = observer.sample()
    np.testing.assert_array_equal(integration_state(env), before)
    np.testing.assert_array_equal(env.data.xpos[body], old_position)
    assert current["objects"][0]["position"][0] == pytest.approx(initial["objects"][0]["position"][0] + .001)
    assert observer.data is not env.data


@pytest.mark.parametrize("shape,radius", [("box", math.sqrt(3) * .02), ("sphere", .02), ("cylinder", math.sqrt(2) * .02)])
def test_compiled_object_top_and_radius_for_each_supported_shape(shape, radius):
    instance = MultiObjectEnv(n_objects=3, allowed_shapes=[shape])
    try:
        instance.reset(seed=17)
        observed = MujocoObserver(instance).sample()
        for obj in observed["objects"]:
            assert obj["top_z_m"] == pytest.approx(.455)
            assert obj["radius_bound_m"] == pytest.approx(radius)
        assert StationaryMonitor(observed, 1).goal[2] == pytest.approx(.555)
    finally:
        instance.close()


def test_real_folded_robot_contacts_are_forbidden(env):
    env.data.qpos[:7] = [0., -.785, 0., -2.356, 0., 1.571, .785]
    observed = MujocoObserver(env).sample()
    assert any(c["moving_robot"] and c["signed_distance_m"] < 0 for c in observed["contacts"])
    assert StationaryMonitor(observed, 1).result["stop_reason"] == "forbidden_contact"


def test_observer_rejects_model_replacement_after_reset(env):
    observer = MujocoObserver(env)
    env.reset(seed=18)
    with pytest.raises(RuntimeError, match="new observer"):
        observer.sample()
