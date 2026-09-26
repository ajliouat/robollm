"""Bounded development-only checks for the privileged joint-path controller."""
import json

import mujoco
import numpy as np
import pytest

from envs.multi_object_env import MultiObjectEnv
from evaluation.stationary_contract import MujocoObserver, StationaryMonitor
from policies.stationary_reach import StationaryReachController


@pytest.fixture
def scene():
    env = MultiObjectEnv(n_objects=3)
    env.reset(seed=17)
    yield env
    env.close()


def integration_state(env):
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(env.model, spec))
    mujoco.mj_getState(env.model, env.data, state, spec)
    return state


def goal(env):
    return StationaryMonitor(MujocoObserver(env).sample(), 1).goal


@pytest.mark.parametrize('value', [[1, 2], [1, 2, 3, 4], [np.nan, 0, 0], [np.inf, 0, 0]])
def test_rejects_invalid_goal(scene, value):
    with pytest.raises(ValueError, match='finite 3-vector'):
        StationaryReachController(scene, value)


@pytest.mark.parametrize('seed', [True, -1, 2.5])
def test_rejects_invalid_planner_seed(scene, seed):
    with pytest.raises(ValueError, match='seed'):
        StationaryReachController(scene, goal(scene), seed)


def test_planning_and_commands_leave_live_state_and_model_untouched(scene):
    before = integration_state(scene)
    original_margins = scene.model.geom_margin.copy()
    original_mass = scene.model.body_mass.copy()
    controller = StationaryReachController(scene, goal(scene), 1729)
    assert controller.metadata['planning_status'] == 'ready'
    assert controller.metadata['collision_free_paths'] > 0
    first = controller.command()
    np.testing.assert_array_equal(first, controller.command())
    np.testing.assert_array_equal(integration_state(scene), before)
    np.testing.assert_array_equal(scene.model.geom_margin, original_margins)
    np.testing.assert_array_equal(scene.model.body_mass, original_mass)
    assert np.any(controller._model.geom_margin != original_margins)
    assert controller._model is not scene.model
    assert controller._data is not scene.data
    json.dumps(controller.metadata, allow_nan=False)


def test_same_reset_and_planner_seed_reproduce_plan_and_counters(scene):
    first = StationaryReachController(scene, goal(scene), 1729)
    second = StationaryReachController(scene, goal(scene), 1729)
    assert first.metadata == second.metadata
    np.testing.assert_array_equal(first.command(), second.command())
    assert first.metadata['starts_attempted'] == 64
    assert first.metadata['ik_iterations'] <= 64 * 250


def test_unreachable_goal_reports_budgeted_failure_and_does_not_move(scene):
    before = integration_state(scene)
    controller = StationaryReachController(scene, [100., 100., 100.], 1729)
    assert controller.metadata['planning_status'] == 'failed'
    assert controller.metadata['failure'] == 'no_collision_free_path'
    assert controller.metadata['starts_attempted'] == 64
    assert controller.metadata['ik_iterations'] == 64 * 250
    np.testing.assert_array_equal(controller.command(), scene.data.qpos[:7])
    np.testing.assert_array_equal(integration_state(scene), before)


def test_clear_endpoint_does_not_allow_a_colliding_connector(scene, monkeypatch):
    # A dev17 IK solution with a clear endpoint but a self-colliding straight
    # connector. Endpoint-only collision tests would incorrectly accept it.
    candidate = np.array([-.3660297387255242, -1.798, -2.017443340846346,
                          -1.9659203143038555, -1.9445241031297134,
                          1.2635964695734732, -.4201752137878563])
    monkeypatch.setattr(StationaryReachController, 'N_STARTS', 1)
    monkeypatch.setattr(StationaryReachController, '_ik', lambda self, start: candidate.copy())
    controller = StationaryReachController(scene, goal(scene), 1729)
    assert controller.metadata['collision_free_endpoints'] == 1
    assert controller.metadata['collision_free_paths'] == 0
    assert controller.metadata['failure'] == 'no_collision_free_path'


def test_observed_tracking_drift_is_rechecked_before_another_command(scene):
    controller = StationaryReachController(scene, goal(scene), 1729)
    scene.data.qpos[:7] = [0, -.785, 0, -2.356, 0, 1.571, .785]
    mujoco.mj_forward(scene.model, scene.data)
    before = integration_state(scene)
    command = controller.command()
    assert controller.metadata['planning_status'] == 'failed'
    assert controller.metadata['failure'] == 'tracking_clearance'
    np.testing.assert_array_equal(command, scene.data.qpos[:7])
    np.testing.assert_array_equal(integration_state(scene), before)


def test_reset_invalidates_the_plan(scene):
    controller = StationaryReachController(scene, goal(scene), 1729)
    scene.reset(seed=17)
    with pytest.raises(ValueError, match='reset'):
        controller.command()


def test_nonfinite_clock_cannot_jump_to_a_reference(scene):
    controller = StationaryReachController(scene, goal(scene), 1729)
    scene.data.time = np.nan
    with pytest.raises(ValueError, match='time'):
        controller.command()


def test_actual_servos_complete_a_safe_stationary_hold_on_development_scene(scene):
    observer = MujocoObserver(scene)
    monitor = StationaryMonitor(observer.sample(), 1)
    controller = StationaryReachController(scene, monitor.goal, 1729)
    assert controller.metadata['planning_status'] == 'ready'
    assert monitor.result['current_distance_m'] > .2  # No trivial reset success.
    sample = 0
    for _ in range(200):
        command = controller.command()
        assert controller.metadata['planning_status'] == 'ready'
        assert command.shape == (7,) and np.isfinite(command).all()
        assert np.all(command >= scene._jnt_lo) and np.all(command <= scene._jnt_hi)
        scene.data.ctrl[scene._arm_act_ids] = command
        scene.data.ctrl[[scene._finger_l_act, scene._finger_r_act]] = .04
        for _ in range(25):
            mujoco.mj_step(scene.model, scene.data)
            sample += 1
            result = monitor.consume(sample, observer.sample())
            if result['done']:
                break
        if result['done']:
            break
    assert result['success']
    assert result['stop_reason'] == 'held_goal'
    assert result['dwell_intervals'] == 500
    assert result['max_motion_bound_m'] <= .005
    assert not result['violations']
    assert result['time_seconds'] <= 10.
    assert np.all(scene.data.warning.number == 0)
