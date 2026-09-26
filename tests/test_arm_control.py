"""Physical reset/control contracts; development seeds only, no training.

These checks establish local simulator properties, not grasping performance or
hardware validity. Historical inertials come from the frozen v1 MJCF snapshot.
"""
from pathlib import Path

import mujoco
import numpy as np
import pytest

from envs.arm_control import HOME_QPOS
from envs.multi_object_env import MultiObjectEnv
from envs.place import PlaceEnv
from envs.tabletop import TabletopEnv
from policies.scripted import ScriptedMoveTo


ROBOT_BODIES = [f"link{i}" for i in range(1, 8)] + ["gripper_base", "finger_l", "finger_r"]
BASELINE_XML = Path(__file__).resolve().parents[1] / "evaluation/frozen_v1/assets/tabletop_scene.xml"


def body_id(model, name):
    identifier = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert identifier >= 0
    return identifier


def arm_contacts(env):
    robot_ids = {body_id(env.model, name) for name in ROBOT_BODIES}
    return [contact for contact in env.data.contact[:env.data.ncon]
            if int(env.model.geom_bodyid[contact.geom1]) in robot_ids
            or int(env.model.geom_bodyid[contact.geom2]) in robot_ids]


@pytest.fixture(params=[TabletopEnv, MultiObjectEnv], ids=["single", "multi"])
def env(request):
    instance = request.param()
    try:
        yield instance
    finally:
        instance.close()


@pytest.mark.parametrize("seed", [17, 18])
def test_reset_has_no_arm_penetration_and_matching_servo_targets(env, seed):
    env.reset(seed=seed)
    assert not arm_contacts(env)
    np.testing.assert_array_equal(env.data.qpos[:9], HOME_QPOS)
    np.testing.assert_array_equal(env.data.ctrl[:9], HOME_QPOS)
    key = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_KEY, "home")
    np.testing.assert_array_equal(env.model.key_qpos[key, :9], HOME_QPOS)
    np.testing.assert_array_equal(env.model.key_ctrl[key], HOME_QPOS)
    assert np.all(HOME_QPOS[:7] > env._jnt_lo)
    assert np.all(HOME_QPOS[:7] < env._jnt_hi)
    # Known FK of the custom collision-free pose, not a task-success target.
    np.testing.assert_allclose(env.ee_pos, [0.5, 0.2501496025963956, 0.7554183534005166], atol=1e-8)


@pytest.mark.parametrize("n_objects", [1, 3, 6])
def test_injected_keyframe_dimensions_and_robot_controls(n_objects):
    env = MultiObjectEnv(n_objects=n_objects)
    try:
        env.reset(seed=17)
        assert env.model.key_qpos.shape == (1, 9 + 7 * n_objects)
        assert env.model.key_ctrl.shape == (1, 9)
        np.testing.assert_array_equal(env.model.key_qpos[0, :9], HOME_QPOS)
        np.testing.assert_array_equal(env.model.key_ctrl[0], HOME_QPOS)
    finally:
        env.close()


def test_place_reset_preserves_its_closed_finger_pose_in_servo_targets():
    env = PlaceEnv()
    try:
        env.reset(seed=17)
        np.testing.assert_array_equal(env.data.qpos[:7], HOME_QPOS[:7])
        np.testing.assert_array_equal(env.data.ctrl[env._arm_act_ids], env.data.qpos[:7])
        np.testing.assert_array_equal(env.data.qpos[7:9], [0.005, 0.005])
        np.testing.assert_array_equal(
            env.data.ctrl[[env._finger_l_act, env._finger_r_act]], env.data.qpos[7:9],
        )
        # At reset the position servos must not actively reopen the gripper.
        np.testing.assert_allclose(
            env.data.actuator_force[[env._finger_l_act, env._finger_r_act]], 0., atol=1e-12,
        )
    finally:
        env.close()


def test_collider_changes_preserve_all_robot_inertials_from_frozen_v1(env):
    baseline = mujoco.MjModel.from_xml_path(str(BASELINE_XML))
    env.reset(seed=17)
    for name in ROBOT_BODIES:
        old, new = body_id(baseline, name), body_id(env.model, name)
        for field in ("body_mass", "body_ipos", "body_iquat", "body_inertia"):
            np.testing.assert_array_equal(getattr(env.model, field)[new], getattr(baseline, field)[old],
                                          err_msg=f"Changed {field} for {name}")


def test_only_robot_bodies_receive_ideal_gravity_compensation(env):
    env.reset(seed=17)
    np.testing.assert_array_equal(env.model.opt.gravity, [0, 0, -9.81])
    robot_ids = {body_id(env.model, name) for name in ROBOT_BODIES}
    for identifier in range(env.model.nbody):
        assert env.model.body_gravcomp[identifier] == (1.0 if identifier in robot_ids else 0.0)
    assert not env.model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONTACT
    for index in range(1, 8):
        assert mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, f"link{index}_geom") >= 0


@pytest.mark.parametrize("seed", [17, 18])
def test_zero_command_holds_the_physical_home_for_ten_seconds(env, seed):
    env.reset(seed=seed)
    position = env.ee_pos.copy()
    joints = env.data.qpos[:7].copy()
    for _ in range(200):
        env.step(np.zeros(4))  # Half-open fingers match their reset target.
    np.testing.assert_allclose(env.ee_pos, position, atol=1e-7, rtol=0)
    np.testing.assert_allclose(env.data.qpos[:7], joints, atol=1e-7, rtol=0)
    assert not arm_contacts(env)


def test_uncompensated_gravity_is_detected_by_the_hold_contract(env):
    env.reset(seed=17)
    position = env.ee_pos.copy()
    env.model.body_gravcomp[:] = 0
    for _ in range(40):
        env.step(np.zeros(4))
    assert np.linalg.norm(env.ee_pos - position) > 0.01


def test_objects_still_fall_and_are_supported_by_table_contacts(env):
    env.reset(seed=17)
    object_identifier = (env._obj_body_ids[0] if isinstance(env, MultiObjectEnv)
                         else env._block_body_id)
    joint = env.model.body_jntadr[object_identifier]
    qpos_address = env.model.jnt_qposadr[joint]
    initial_z = float(env.data.qpos[qpos_address + 2]) + 0.1
    env.data.qpos[qpos_address + 2] = initial_z
    mujoco.mj_forward(env.model, env.data)
    env.step(np.zeros(4))
    assert env.data.xpos[object_identifier, 2] < initial_z - 0.003
    for _ in range(49):
        env.step(np.zeros(4))
    assert env.data.xpos[object_identifier, 2] == pytest.approx(0.435, abs=0.005)
    table = body_id(env.model, "table")
    assert any(
        {int(env.model.geom_bodyid[contact.geom1]), int(env.model.geom_bodyid[contact.geom2])}
        == {object_identifier, table}
        for contact in env.data.contact[:env.data.ncon]
    )


def test_real_self_collisions_remain_active_after_clearance_repair(env):
    env.reset(seed=17)
    # The original invalid folding is still invalid: this is no collision-mask workaround.
    env.data.qpos[:7] = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
    mujoco.mj_forward(env.model, env.data)
    pair = {body_id(env.model, "link2"), body_id(env.model, "link4")}
    assert any(
        {int(env.model.geom_bodyid[contact.geom1]), int(env.model.geom_bodyid[contact.geom2])} == pair
        and contact.dist < -0.005
        for contact in env.data.contact[:env.data.ncon]
    )


@pytest.mark.parametrize("yaw_delta", [-0.12, 0.12])
def test_controller_reaches_local_forward_kinematic_targets(env, yaw_delta):
    env.reset(seed=17)
    reference = mujoco.MjData(env.model)
    reference.qpos[:] = env.data.qpos
    reference.qpos[0] += yaw_delta
    mujoco.mj_forward(env.model, reference)
    target = reference.site_xpos[env._ee_site_id].copy()
    assert np.linalg.norm(env.ee_pos - target) > 0.03
    controller = ScriptedMoveTo()  # Existing gain8 and threshold3cm, without tuning.
    for _ in range(200):
        action = controller.act({"ee_pos": env.ee_pos, "obj_pos": target})
        env.step(action)
        if np.linalg.norm(env.ee_pos - target) < 0.03:
            break
    assert np.linalg.norm(env.ee_pos - target) < 0.03
