"""Target identity and lifecycle regressions, without learning or rendering."""
from types import SimpleNamespace

import numpy as np
import pytest

from evaluation.pipeline import HierarchicalExecutor
from planner.task_parser import SubTask, TaskPlan
from policies.scripted import Phase, ScriptedMoveTo, ScriptedPickPlace


class TinyScene:
    """Deterministic state transitions make false-success oracles testable."""
    def __init__(self, callback=None, auto_gripper=True):
        self._obj_specs = [
            SimpleNamespace(name="red_box_0", color_name="red", shape="box"),
            SimpleNamespace(name="blue_box_1", color_name="blue", shape="box"),
        ]
        self.positions = np.array([[0.2, 0, 0.435], [0.6, 0, 0.435]])
        self.ee_pos = np.array([0.4, 0, 0.435])
        self.data = SimpleNamespace(qpos=np.zeros(9))
        self.data.qpos[7:9] = 0.04
        self._max_episode_steps = 5
        self._elapsed_steps = 0
        self.terminated = False
        self.actions = []
        self.callback = callback
        self.auto_gripper = auto_gripper

    def object_pos(self, index=0):
        return self.positions[index].copy()

    def _check_terminated(self):
        return self.terminated

    def step(self, action):
        if self.terminated:
            raise AssertionError("Stepped an ended environment")
        action = np.asarray(action).copy()
        self.actions.append(action)
        self._elapsed_steps += 1
        self.ee_pos += action[:3] * 0.05
        if self.auto_gripper:
            self.data.qpos[7:9] = 0.02 * (1 + action[3])
        if self.callback:
            self.callback(self, action)
        truncated = self._elapsed_steps >= self._max_episode_steps
        return np.zeros(1), -1.0, self.terminated, truncated, {}


def plan(*steps):
    return TaskPlan([SubTask(primitive, target) for primitive, target in steps], valid=True)


class PickReleasePolicy:
    """Minimal controller stub; tests concern outcome verification, not tuning."""
    approach_height = 0.15
    grasp_height_offset = 0.02
    lift_height = 0.20

    def __init__(self):
        self.infos = []
        self.reset()

    def reset(self):
        self.phase = Phase.APPROACH

    def act(self, info):
        self.infos.append({key: value.copy() if isinstance(value, np.ndarray) else value
                           for key, value in info.items()})
        if self.phase == Phase.APPROACH:
            self.phase = Phase.LIFT
            return np.array([0., 0., 0., -1.])
        if self.phase == Phase.MOVE:
            self.phase = Phase.RELEASE
            return np.array([0., 0., 0., -1.])
        if self.phase in {Phase.RELEASE, Phase.DONE}:
            self.phase = Phase.DONE
            return np.array([0., 0., 0., 1.])
        return np.array([0., 0., 0., -1.])


def test_nonfirst_target_drives_action():
    """This assertion also runs against the old public executor API."""
    env = TinyScene()
    executor = HierarchicalExecutor(max_steps_per_subtask=1)
    executor.execute(env, plan(("move_to", "blue_box_1")))
    # Blue is right of EE; index0/red is left. Object0 substitution reverses it.
    assert env.actions[0][0] > 0, "Controller drove toward object0 instead of grounded blue target"


def test_live_named_target_is_refreshed_for_actions_and_success():
    def move_target_then_reach(env, action):
        if len(env.actions) == 1:
            env.ee_pos = env.positions[1].copy()
            env.positions[1][0] += 0.05
        else:
            env.ee_pos = env.positions[1].copy()

    env = TinyScene(move_target_then_reach)
    executor = HierarchicalExecutor(max_steps_per_subtask=3)
    result = executor.execute(env, plan(("move_to", "blue_box_1")))
    step = result.step_results[0]
    assert len(env.actions) == 2  # Being at the stale snapshot must not succeed.
    assert env.actions[1][0] > 0
    assert step.target_name == "blue_box_1"
    assert step.initial_distance == pytest.approx(0.2)
    assert step.final_distance == 0
    assert step.success and result.overall_success


def test_reaching_already_satisfied_has_no_new_success_credit():
    env = TinyScene()
    env.ee_pos = env.positions[1].copy()
    result = HierarchicalExecutor().execute(env, plan(("move_to", "blue_box_1")))
    step = result.step_results[0]
    assert step.already_satisfied and step.completed
    assert not step.success and not result.overall_success
    assert step.n_steps == 0 and not env.actions
    assert result.sub_task_success_rate == 0


def test_initial_reaching_precondition_can_be_followed_by_new_pick():
    def lift_selected(env, action):
        env.positions[1][2] += 0.1
        env.ee_pos = env.positions[1].copy()

    env = TinyScene(lift_selected)
    env.ee_pos = env.positions[1].copy()
    executor = HierarchicalExecutor(max_steps_per_subtask=2)
    executor._pick_place = PickReleasePolicy()
    result = executor.execute(env, plan(("move_to", "blue_box_1"), ("pick", "blue_box_1")))
    assert result.step_results[0].already_satisfied
    assert result.step_results[1].success
    assert result.overall_success


@pytest.mark.parametrize("policy_type", [ScriptedMoveTo, ScriptedPickPlace])
def test_named_mapping_wins_over_conflicting_generic_position(policy_type):
    info = {
        "ee_pos": np.array([0.4, 0, 0.435]),
        "obj_pos": np.array([0.2, 0, 0.435]),
        "target_name": "blue",
        "obj_positions": {"red": np.array([0.2, 0, 0.435]), "blue": np.array([0.6, 0, 0.435])},
    }
    assert policy_type().act(info)[0] > 0


@pytest.mark.parametrize("policy_type", [ScriptedMoveTo, ScriptedPickPlace])
def test_ambiguous_multiobject_input_is_not_silently_object0(policy_type):
    with pytest.raises(ValueError, match="unambiguous"):
        policy_type().act({"ee_pos": np.zeros(3), "obj_positions": {"a": np.zeros(3), "b": np.ones(3)}})


def test_missing_named_mapping_does_not_fall_back_to_generic_position():
    with pytest.raises(ValueError, match="absent"):
        ScriptedMoveTo().act({"ee_pos": np.zeros(3), "obj_pos": np.ones(3),
                              "target_name": "missing", "obj_positions": {"a": np.zeros(3)}})


def test_pick_does_not_credit_first_object_or_initial_absolute_height():
    env = TinyScene()
    env.positions[0][2] = 0.9  # Old first-object/absolute-height oracle succeeds falsely.
    env.positions[1][2] = 0.8  # Selected object is already high, but never rises.
    env.ee_pos = env.positions[1].copy()
    executor = HierarchicalExecutor(max_steps_per_subtask=2)
    executor._pick_place = PickReleasePolicy()
    step = executor.execute(env, plan(("pick", "blue_box_1"))).step_results[0]
    assert not step.success
    assert step.source_name == "blue_box_1"


def test_pick_tracks_relative_lift_of_nonfirst_source():
    def lift_blue(env, action):
        env.positions[1][2] += 0.09
        env.ee_pos = env.positions[1].copy()

    env = TinyScene(lift_blue)
    env.positions[1][2] = 0.1  # A relative lift can succeed below the old absolute0.5 threshold.
    executor = HierarchicalExecutor(max_steps_per_subtask=2)
    policy = PickReleasePolicy()
    executor._pick_place = policy
    step = executor.execute(env, plan(("pick", "blue_box_1"))).step_results[0]
    assert step.success
    np.testing.assert_allclose(policy.infos[0]["obj_pos"], [0.6, 0, 0.1])
    assert policy.infos[0]["target_name"] == "blue_box_1"


def test_pick_requires_observed_closed_gripper_not_just_rising_object():
    def lift_blue(env, action):
        env.positions[1][2] += 0.1
        env.ee_pos = env.positions[1].copy()

    env = TinyScene(lift_blue, auto_gripper=False)
    executor = HierarchicalExecutor(max_steps_per_subtask=2)
    executor._pick_place = PickReleasePolicy()
    assert not executor.execute(env, plan(("pick", "blue_box_1"))).overall_success


def test_pick_does_not_credit_a_raised_object_far_from_closed_gripper():
    def raise_remote_object(env, action):
        env.positions[1][2] += 0.1
        # The object's motion is not accompanied by the end effector.

    env = TinyScene(raise_remote_object)
    executor = HierarchicalExecutor(max_steps_per_subtask=2)
    executor._pick_place = PickReleasePolicy()
    result = executor.execute(env, plan(("pick", "blue_box_1")))
    assert not result.overall_success
    assert np.all(env.data.qpos[7:9] == 0)


def pick_then_place_callback(env, action):
    if len(env.actions) == 1:
        env.positions[1][2] += 0.1
    else:
        # Source is already beside the destination while the gripper is still closed.
        env.positions[1] = env.positions[0] + np.array([0, 0, 0.04])
    env.ee_pos = env.positions[1].copy()


def test_place_keeps_picked_source_and_waits_for_release():
    env = TinyScene(pick_then_place_callback)
    executor = HierarchicalExecutor(max_steps_per_subtask=3)
    policy = PickReleasePolicy()
    executor._pick_place = policy
    result = executor.execute(env, plan(("pick", "blue_box_1"), ("place", "red_box_0")))
    assert result.overall_success
    place = result.step_results[1]
    assert place.source_name == "blue_box_1" and place.target_name == "red_box_0"
    assert place.n_steps == 2  # First at-target state was still held.
    assert env.actions[-1][3] > 0
    assert policy.infos[1]["target_name"] == "blue_box_1"
    np.testing.assert_allclose(policy.infos[1]["obj_pos"], [0.6, 0, 0.535])
    np.testing.assert_allclose(policy.infos[1]["goal_pos"], [0.2, 0, 0.435])


def test_place_release_command_without_observed_opening_does_not_succeed():
    def jam_after_pick(env, action):
        pick_then_place_callback(env, action)
        env.data.qpos[7:9] = 0.0

    env = TinyScene(jam_after_pick)
    executor = HierarchicalExecutor(max_steps_per_subtask=3)
    executor._pick_place = PickReleasePolicy()
    result = executor.execute(env, plan(("pick", "blue_box_1"), ("place", "red_box_0")))
    assert result.step_results[0].success
    assert not result.step_results[1].success


def test_place_compares_3d_source_position_not_only_xy():
    def source_too_high(env, action):
        pick_then_place_callback(env, action)
        if len(env.actions) > 1:
            env.positions[1][2] += 0.5

    env = TinyScene(source_too_high)
    executor = HierarchicalExecutor(max_steps_per_subtask=3)
    executor._pick_place = PickReleasePolicy()
    result = executor.execute(env, plan(("pick", "blue_box_1"), ("place", "red_box_0")))
    assert not result.step_results[1].success


def test_place_without_successful_pick_cannot_reuse_scene_geometry():
    env = TinyScene()
    env.positions[1] = env.positions[0].copy()
    step = HierarchicalExecutor().execute(env, plan(("place", "blue_box_1"))).step_results[0]
    assert not step.success and step.n_steps == 0
    assert "successful pick" in step.error
    assert not env.actions


def test_failed_prerequisite_skips_remaining_steps_and_restores_horizon():
    env = TinyScene()
    result = HierarchicalExecutor(max_steps_per_subtask=1).execute(
        env, plan(("move_to", "blue_box_1"), ("pick", "blue_box_1")))
    assert not result.step_results[0].success
    assert result.step_results[1].skipped
    assert len(env.actions) == 1
    assert env._max_episode_steps == 5


@pytest.mark.parametrize("end_kind", ["terminate", "truncate"])
def test_environment_end_stops_following_steps_even_after_reaching(end_kind):
    def end_after_reach(env, action):
        env.ee_pos = env.positions[1].copy()
        if end_kind == "terminate":
            env.terminated = True
        else:
            env._elapsed_steps = env._max_episode_steps

    env = TinyScene(end_after_reach)
    result = HierarchicalExecutor(max_steps_per_subtask=3).execute(
        env, plan(("move_to", "blue_box_1"), ("pick", "blue_box_1")))
    assert result.step_results[0].success
    assert result.step_results[1].skipped
    assert len(env.actions) == 1
    assert not result.overall_success
    assert env._max_episode_steps == 5


def test_exception_restores_horizon_and_does_not_leave_held_source():
    def crash(env, action):
        raise RuntimeError("simulator failure")

    env = TinyScene(crash)
    executor = HierarchicalExecutor(max_steps_per_subtask=10)
    with pytest.raises(RuntimeError, match="simulator failure"):
        executor.execute(env, plan(("move_to", "blue_box_1")))
    assert env._max_episode_steps == 5
    assert executor._held_object_name is None


def test_executor_reuse_does_not_carry_source_between_plans():
    env = TinyScene(pick_then_place_callback)
    executor = HierarchicalExecutor(max_steps_per_subtask=3)
    executor._pick_place = PickReleasePolicy()
    assert executor.execute(env, plan(("pick", "blue_box_1"))).overall_success
    actions_before = len(env.actions)
    result = executor.execute(env, plan(("place", "red_box_0")))
    assert not result.overall_success and len(env.actions) == actions_before


@pytest.mark.parametrize("next_step", [
    ("move_to", "red_box_0"), ("pick", "red_box_0"), ("place", "blue_box_1"),
])
def test_held_source_cannot_be_dropped_overwritten_or_placed_on_itself(next_step):
    env = TinyScene(pick_then_place_callback)
    executor = HierarchicalExecutor(max_steps_per_subtask=3)
    executor._pick_place = PickReleasePolicy()
    result = executor.execute(env, plan(("pick", "blue_box_1"), next_step))
    assert result.step_results[0].success
    assert not result.step_results[1].success
    assert result.step_results[1].error
    assert len(env.actions) == 1


def test_already_ended_environment_is_not_extended_and_restarted():
    env = TinyScene()
    env._elapsed_steps = env._max_episode_steps
    result = HierarchicalExecutor(max_steps_per_subtask=100).execute(env, plan(("move_to", "blue_box_1")))
    assert result.step_results[0].skipped and result.step_results[0].truncated
    assert not env.actions and env._max_episode_steps == 5


def test_previously_terminated_environment_is_not_restarted():
    env = TinyScene()
    env.terminated = True
    env._elapsed_steps = 1
    result = HierarchicalExecutor(max_steps_per_subtask=100).execute(env, plan(("move_to", "blue_box_1")))
    assert result.step_results[0].skipped and result.step_results[0].terminated
    assert not env.actions and env._max_episode_steps == 5


@pytest.mark.parametrize("steps", [0, -1, True, 1.5])
def test_invalid_horizon_is_rejected(steps):
    with pytest.raises(ValueError, match="positive integer"):
        HierarchicalExecutor(max_steps_per_subtask=steps)


def test_real_mujoco_nonfirst_target_passes_live_identity_to_controller():
    from envs.multi_object_env import MultiObjectEnv

    class CapturePolicy:
        def __init__(self):
            self.infos = []

        def act(self, info):
            self.infos.append(info)
            return np.array([0., 0., 0., 1.])

    env = MultiObjectEnv(n_objects=3)
    try:
        env.reset(seed=17)
        policy = CapturePolicy()
        name = env._obj_specs[2].name
        target = env.object_pos(2)
        executor = HierarchicalExecutor(max_steps_per_subtask=1, move_to_policy=policy)
        result = executor.execute(env, plan(("move_to", name)))
        assert result.step_results[0].target_name == name
        np.testing.assert_array_equal(policy.infos[0]["obj_pos"], target)
        assert policy.infos[0]["target_name"] == name
        assert env._max_episode_steps == 200
    finally:
        env.close()
