"""Development-only crossed probes of geometry, reset pose and gravity support.

All eight variants retain collisions, object gravity, original link inertials,
the gain-8 policy and original DLS implementation. These probes continue for the
full horizon to inspect contacts; their minimum-distance hits are NOT v2 scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from types import ModuleType
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from envs.arm_control import HOME_QPOS
from evaluation.frozen_v1 import ROOT, environment_class, manifest
from evaluation.grounded_control import save_report
from evaluation.benchmark import _runtime_provenance
from policies.scripted import ScriptedMoveTo

SEEDS = (17, 18)
ROBOT_BODIES = tuple(f"link{i}" for i in range(1, 8)) + (
    "gripper_base", "finger_l", "finger_r",
)


def _variant(geometry: bool, home: bool, compensation: bool, asset: Path):
    """Compile one declared intervention, with the original implementation."""
    environment_class()  # Verify and load the isolated original dependency.
    tree = ET.fromstring((ROOT / "assets/tabletop_scene.xml").read_bytes())
    original = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
    if geometry:
        for name, extent in (("link1", "0 0 0.03 0 0 0.12"),
                             ("link2", "0 0 0 0 0 0.28")):
            body = tree.find(f'.//body[@name="{name}"]')
            body.find("geom").set("fromto", extent)
            bid = mujoco.mj_name2id(original, mujoco.mjtObj.mjOBJ_BODY, name)
            ET.SubElement(body, "inertial", {
                "pos": " ".join(map(str, original.body_ipos[bid])),
                "quat": " ".join(map(str, original.body_iquat[bid])),
                "mass": str(original.body_mass[bid]),
                "diaginertia": " ".join(map(str, original.body_inertia[bid])),
            })
    if compensation:
        for name in ROBOT_BODIES:
            tree.find(f'.//body[@name="{name}"]').set("gravcomp", "1")
    payload = ET.tostring(tree)
    asset.write_bytes(payload)
    # The one import rewrite isolates the historical spawner. The original
    # reset, stepping, IK and observation code are otherwise executed verbatim.
    source = (ROOT / "multi_object_env.py").read_text().replace(
        "from envs.object_spawner import", "from evaluation.frozen_v1._spawner import"
    )
    mod = ModuleType("motor_diagnostic_variant")
    mod.__file__ = str(ROOT / "multi_object_env.py")
    exec(compile(source, mod.__file__, "exec"), mod.__dict__)
    mod._SCENE_XML = asset
    if home:
        mod._HOME_QPOS = HOME_QPOS.copy()
    return mod.MultiObjectEnv, hashlib.sha256(payload).hexdigest()


def _state(env):
    m, d = env.model, env.data
    jacp = np.zeros((3, m.nv)); jacr = np.zeros_like(jacp)
    mujoco.mj_jacSite(m, d, jacp, jacr, env._ee_site_id)
    contacts = []
    for index, contact in enumerate(d.contact):
        force = np.zeros(6)
        mujoco.mj_contactForce(m, d, index, force)
        bodies = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                                    int(m.geom_bodyid[g]))
                  for g in (contact.geom1, contact.geom2)]
        contacts.append({"bodies": bodies, "distance_m": float(contact.dist),
                         "normal_force_n": float(force[0])})
    return {
        "ee": env.ee_pos.tolist(), "target": env.object_pos(1).tolist(),
        "qpos": d.qpos[:9].tolist(), "qvel": d.qvel[:9].tolist(),
        "ctrl": d.ctrl.tolist(), "bias": d.qfrc_bias[:9].tolist(),
        "actuator_force": d.qfrc_actuator[:9].tolist(),
        "passive_force": d.qfrc_passive[:9].tolist(),
        "constraint_force": d.qfrc_constraint[:9].tolist(),
        "jacobian_singular_values": np.linalg.svd(jacp[:, :7], compute_uv=False).tolist(),
        "contacts": contacts,
    }


def run_diagnostics():
    report = {"kind": "development_diagnostics", "seeds": list(SEEDS),
              "baseline": manifest(), "provenance": _runtime_provenance(),
              "horizon_steps": 200, "episodes": []}
    with tempfile.TemporaryDirectory() as tmp:
        for geometry in (False, True):
            for home in (False, True):
                for compensation in (False, True):
                    tag = f"geometry{int(geometry)}_home{int(home)}_gravity{int(compensation)}"
                    cls, asset_hash = _variant(geometry, home, compensation, Path(tmp) / "scene.xml")
                    for seed in SEEDS:
                        env = cls()
                        try:
                            env.reset(seed=seed)
                            if home:
                                env.data.ctrl[:9] = env.data.qpos[:9]
                                mujoco.mj_forward(env.model, env.data)
                            trace = [_state(env)]
                            policy = ScriptedMoveTo(gain=8)
                            for _ in range(200):
                                env.step(policy.act({"ee_pos": env.ee_pos, "obj_pos": env.object_pos(1)}))
                                trace.append(_state(env))
                            distances = [float(np.linalg.norm(np.array(t["ee"]) - t["target"])) for t in trace]
                            report["episodes"].append({
                                "variant": tag, "seed": seed, "model_xml_sha256": asset_hash,
                                "minimum_distance_m": min(distances[1:]),
                                "first_proximity_step": next((i for i, d in enumerate(distances) if i and d < .03), None),
                                "trace": trace,
                            })
                        finally:
                            env.close()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run_diagnostics()
    save_report(result, args.output)
    for episode in result["episodes"]:
        print(episode["variant"], episode["seed"], episode["minimum_distance_m"], episode["first_proximity_step"])
