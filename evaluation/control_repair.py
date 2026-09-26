"""Paired frozen-v1/repaired-system study; development only until source freeze.

The common object scene is paired. Robot model, home and servo may differ, so
this measures the declared system bundle, not an isolated causal component.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from datetime import datetime, timezone
from typing import Callable, Sequence

import numpy as np

from evaluation import frozen_v1
from evaluation.benchmark import _runtime_provenance, wilson_interval
from evaluation.grounded_control import (
    _canonical, _snapshot, _vector, save_report, semantic_digest, semantic_payload,
)
from evaluation.pipeline import HierarchicalExecutor
from planner.grounder import SimGrounder
from planner.task_parser import SubTask, TaskPlan
from policies.scripted import ScriptedMoveTo

PROTOCOL_PATH = Path(__file__).with_name("control_repair_protocol.json")
REPO = Path(__file__).resolve().parents[1]
VARIANTS = ("frozen_v1", "repaired_system")
DEVELOPMENT_SEEDS = [17, 18, *range(101, 111)]
HELD_OUT_SEEDS = list(range(260927000, 260927100))
TRACE_COLUMNS = [
    "step", "action_x", "action_y", "action_z", "gripper",
    "ee_x", "ee_y", "ee_z", "target_x", "target_y", "target_z",
    "object0_x", "object0_y", "object0_z", "named_target_distance_m",
    "object0_distance_m", "target_displacement_m", "terminated", "truncated",
]
DIAGNOSTIC_COLUMNS = [
    "step", "arm_qpos", "arm_qvel", "arm_actuator_targets", "arm_qfrc_actuator",
    "arm_qfrc_bias", "arm_qfrc_constraint", "arm_qfrc_applied", "arm_qacc",
    "arm_qfrc_passive", "arm_qfrc_gravcomp", "contacts", "simulation_time",
]
CONTACT_COLUMNS = ["geom1", "geom2", "signed_distance_m", "force_torque_6d"]


class StudyExecutionError(RuntimeError):
    """An aborted run retains inspected scenes/partial traces, without a summary."""
    def __init__(self, message: str, report: dict):
        super().__init__(message)
        self.report = report


class _EpisodeExecutionError(RuntimeError):
    def __init__(self, message: str, episode: dict):
        super().__init__(message)
        self.episode = episode


def _sha(value) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def load_protocol(path=PROTOCOL_PATH) -> dict:
    p = json.loads(Path(path).read_text())
    expected = {
        "schema_version": 2, "study_id": "robollm-control-repair-v2",
        "development_seeds": DEVELOPMENT_SEEDS, "held_out_seeds": HELD_OUT_SEEDS,
        "previously_observed_evaluation_seeds": list(range(260926000, 260926100)),
        "variants": list(VARIANTS), "baseline_source_revision": frozen_v1.REVISION,
        "controller": {"gain": 8.0, "threshold_m": 0.03, "horizon_steps": 200, "gripper_action": 1.0},
        "environment": {"n_objects": 3, "render_mode": None, "max_episode_steps": 200,
                        "settling_steps": 0, "control_dt_seconds": 0.05},
    }
    for key, value in expected.items():
        if p.get(key) != value:
            raise ValueError(f"Protocol field {key!r} differs from the reserved study")
    if p.get("target_selection", {}).get("indices") != [1, 2]:
        raise ValueError("Targets must alternate indices 1 and 2")
    return p


def select_seeds(protocol: dict, split: str, seeds=None) -> list[tuple[int, int]]:
    if split not in ("development", "held-out"):
        raise ValueError("Unknown split")
    declared = protocol["development_seeds" if split == "development" else "held_out_seeds"]
    chosen = list(declared if seeds is None else seeds)
    if (not chosen or any(type(s) is not int for s in chosen)
            or len(set(chosen)) != len(chosen) or any(s not in declared for s in chosen)):
        raise ValueError("Seeds must be distinct integers from the declared split")
    if split == "held-out" and chosen != declared:
        raise ValueError("Held-out evaluation must retain all 100 reserved seeds in order")
    return [(declared.index(seed), seed) for seed in chosen]


def _file_hashes(paths) -> dict:
    return {path: hashlib.sha256((REPO / path).read_bytes()).hexdigest() for path in paths}


def _real_variants() -> tuple[dict[str, Callable], dict]:
    from envs.multi_object_env import MultiObjectEnv
    manifests = {
        "frozen_v1": frozen_v1.manifest(),
        "repaired_system": {"files": _file_hashes(sorted(
            str(path.relative_to(REPO)) for path in (REPO / "envs").rglob("*")
            if path.is_file() and path.suffix in (".py", ".xml")
        ))},
        "shared_evaluator": {"files": _file_hashes([
            "evaluation/control_repair.py", "evaluation/pipeline.py", "planner/grounder.py",
            "policies/scripted.py", "evaluation/control_repair_protocol.json",
        ])},
    }
    return {
        "frozen_v1": frozen_v1.create_environment,
        "repaired_system": lambda: MultiObjectEnv(n_objects=3, render_mode=None, max_episode_steps=200),
    }, manifests


def _initial_state(env, observation, target_index: int, fixture: bool) -> tuple[dict, dict]:
    full = _snapshot(env, observation)
    common = {"objects": [dict(obj) for obj in full["objects"]],
              "target_index": target_index, "target_name": full["objects"][target_index]["name"]}
    if fixture:
        common["world"] = {"fixture": True}
    else:
        import mujoco
        model, data = env.model, env.data
        common["world"] = {name: np.asarray(getattr(model.opt, name)).tolist()
                           for name in ("gravity", "timestep", "integrator", "solver", "iterations", "tolerance")}

        def geoms(ids):
            return [{"name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(i)),
                     **{name: np.asarray(getattr(model, "geom_" + name)[i]).tolist()
                        for name in ("type", "size", "pos", "quat", "friction", "solimp", "solref",
                                     "contype", "conaffinity", "condim")}}
                    for i in ids]

        table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
        if table < 0:
            raise RuntimeError("Study requires the declared shared table")
        common["world"]["table_position"] = data.xpos[table].tolist()
        common["world"]["table_quaternion"] = data.xquat[table].tolist()
        common["world"]["static_geoms"] = geoms(np.flatnonzero(np.isin(model.geom_bodyid, [0, table])))
        for obj, body_id in zip(common["objects"], env._obj_body_ids):
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 0)
            obj["quaternion"] = data.xquat[body_id].tolist()
            obj["velocity_6d"] = velocity.tolist()
            obj["actual_mass"] = float(model.body_mass[body_id])
            obj["actual_inertia"] = model.body_inertia[body_id].tolist()
            obj["geoms"] = geoms(np.flatnonzero(model.geom_bodyid == body_id))
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.zeros(mujoco.mj_stateSize(model, spec))
        mujoco.mj_getState(model, data, state, spec)
        full["integration_state"] = {"spec": int(spec), "values": state.tolist()}
        arrays = (
            "qpos0", "body_pos", "body_quat", "body_mass", "body_inertia", "body_gravcomp",
            "jnt_type", "jnt_pos", "jnt_axis", "jnt_range", "dof_damping", "dof_armature",
            "geom_type", "geom_size", "geom_pos", "geom_quat", "geom_friction", "geom_solimp",
            "geom_solref", "geom_contype", "geom_conaffinity", "geom_condim", "exclude_signature",
            "actuator_trnid", "actuator_gear", "actuator_gainprm", "actuator_biasprm",
            "actuator_ctrlrange", "actuator_forcerange",
        )
        model_parameters = {name: np.asarray(getattr(model, name)).tolist() for name in arrays}
        model_parameters["options"] = {name: np.asarray(getattr(model.opt, name)).tolist()
                                        for name in ("gravity", "timestep", "integrator", "solver", "iterations", "tolerance")}
        full["model_parameters_sha256"] = _sha(model_parameters)
        full["model_parameters"] = model_parameters
        full["geom_names"] = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(model.ngeom)]
        full["common_scene"] = common
    _canonical(full)
    _canonical(common)
    return full, common


def _diagnostics(env, step: int):
    import mujoco
    model, data = env.model, env.data
    joint_ids = np.asarray(env._arm_jnt_ids)
    qpos_ids, dof_ids = model.jnt_qposadr[joint_ids], model.jnt_dofadr[joint_ids]
    contacts = []
    for i in range(data.ncon):
        contact = data.contact[i]
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        contacts.append([int(contact.geom1), int(contact.geom2), float(contact.dist), force.tolist()])
    row = [step, data.qpos[qpos_ids].tolist(), data.qvel[dof_ids].tolist(),
           data.ctrl[env._arm_act_ids].tolist()]
    row.extend(np.asarray(getattr(data, name))[dof_ids].tolist() for name in (
        "qfrc_actuator", "qfrc_bias", "qfrc_constraint", "qfrc_applied", "qacc",
    ))
    row.append(data.qfrc_passive[dof_ids].tolist())
    row.append(data.qfrc_gravcomp[dof_ids].tolist() if hasattr(data, "qfrc_gravcomp") else None)
    row.extend([contacts, float(data.time)])
    _canonical(row)
    return row


class _TraceEnv:
    def __init__(self, env, target_index: int, fixture: bool):
        self.env, self.target_index, self.fixture = env, target_index, fixture
        self.target_initial = _vector(env.object_pos(target_index))
        self.rows, self.diagnostics = [], []
        self.ended = False
        self.attempted_steps = 0
        self.pending_action = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def _max_episode_steps(self):
        return self.env._max_episode_steps

    @_max_episode_steps.setter
    def _max_episode_steps(self, value):
        self.env._max_episode_steps = min(value, 200)

    def step(self, action):
        if self.ended or len(self.rows) >= 200:
            raise RuntimeError("Attempt to step an ended episode")
        action = _vector(action, 4)
        if np.any(abs(action) > 1) or action[3] != 1:
            raise ValueError("Study actions must be bounded, with the gripper open")
        self.pending_action = action.tolist()
        self.attempted_steps += 1
        before = None if self.fixture else float(self.env.data.time)
        output = self.env.step(action)
        _, reward, terminated, truncated, _ = output
        if not np.isfinite(reward):
            raise ValueError("Nonfinite reward")
        if not self.fixture and not np.isclose(self.env.data.time - before, 0.05, rtol=0, atol=1e-10):
            raise RuntimeError("Simulator control duration differs from the frozen 20Hz protocol")
        ee, target, object0 = _vector(self.env.ee_pos), _vector(self.env.object_pos(self.target_index)), _vector(self.env.object_pos(0))
        self.rows.append([len(self.rows) + 1, *action.tolist(), *ee.tolist(), *target.tolist(),
                          *object0.tolist(), float(np.linalg.norm(ee - target)),
                          float(np.linalg.norm(ee - object0)), float(np.linalg.norm(target - self.target_initial)),
                          bool(terminated), bool(truncated)])
        if not self.fixture:
            self.diagnostics.append(_diagnostics(self.env, len(self.rows)))
        self.ended = bool(terminated or truncated)
        self.pending_action = None
        return output


def _episode(env, target_index, snapshot, fixture: bool) -> dict:
    traced = _TraceEnv(env, target_index, fixture)
    try:
        return _execute_episode(env, target_index, snapshot, traced)
    except Exception as exc:
        partial = {"status": "aborted", "target_name": snapshot["objects"][target_index]["name"],
                   "target_index": target_index, "steps": len(traced.rows),
                   "attempted_steps": traced.attempted_steps, "pending_action": traced.pending_action,
                   "trace": traced.rows, "diagnostics": traced.diagnostics,
                   "error_type": type(exc).__name__, "error": str(exc)}
        raise _EpisodeExecutionError(str(exc), partial) from exc


def _execute_episode(env, target_index, snapshot, traced) -> dict:
    target_name = snapshot["objects"][target_index]["name"]
    executor = HierarchicalExecutor(SimGrounder(), 200, move_to_policy=ScriptedMoveTo(gain=8.0, threshold=0.03))
    if executor.REACH_DISTANCE != 0.03:
        raise RuntimeError("Executor success threshold differs from frozen protocol")
    result = executor.execute(traced, TaskPlan([SubTask("move_to", target_name)], valid=True))
    if len(result.step_results) != 1:
        raise RuntimeError("Expected exactly one reaching result")
    step = result.step_results[0]
    if (step.grounding is None or not step.grounding.success or step.grounding.matched is None
            or step.grounding.matched.name != target_name or step.target_name != target_name):
        raise RuntimeError("Grounding did not preserve the requested identity")
    if step.error or step.skipped or step.n_steps != len(traced.rows):
        raise RuntimeError(f"Invalid executor result: {step.error}")
    initial = float(np.linalg.norm(np.asarray(snapshot["ee_position"]) - traced.target_initial))
    distances = [initial] + [row[14] for row in traced.rows]
    pre_satisfied = initial < 0.03
    success = not pre_satisfied and any(d < 0.03 for d in distances[1:])
    if step.already_satisfied != pre_satisfied or step.success != success:
        raise RuntimeError("Executor outcome disagrees with independently recorded geometry")
    if pre_satisfied and traced.rows:
        raise RuntimeError("Initially satisfied scene performed actions")
    if any(row[14] < 0.03 or row[17] or row[18] for row in traced.rows[:-1]):
        raise RuntimeError("Trace continued after its first hit or episode boundary")
    if not (pre_satisfied or success or step.terminated or step.truncated):
        raise RuntimeError("Episode ended without a protocol stopping condition")
    return {
        "status": "complete",
        "target_name": target_name, "target_index": target_index, "grounded_name": step.grounding.matched.name,
        "success": success, "initially_satisfied": pre_satisfied, "steps": len(traced.rows),
        "initial_distance_m": initial, "final_distance_m": distances[-1], "min_distance_m": min(distances),
        "final_target_displacement_m": traced.rows[-1][16] if traced.rows else 0.0,
        "max_target_displacement_m": max([0.0] + [row[16] for row in traced.rows]),
        "terminated": bool(step.terminated), "truncated": bool(step.truncated),
        "stop_reason": "initially_satisfied" if pre_satisfied else "target_reached" if success else "environment_done",
        "trace": traced.rows, "diagnostics": traced.diagnostics,
    }


def summarize(episodes: list[dict], seeds: list[int]) -> dict:
    by_variant, totals = {}, {}
    for variant in VARIANTS:
        rows = [e for e in episodes if e["variant"] == variant]
        indexed = {e["env_seed"]: e for e in rows}
        if len(rows) != len(seeds) or set(indexed) != set(seeds):
            raise RuntimeError("Missing or duplicate paired episode")
        by_variant[variant] = indexed
        count = sum(e["success"] for e in rows)
        low, high = wilson_interval(count, len(seeds))
        totals[variant] = {"n_episodes": len(seeds), "n_successes": count,
            "n_initially_satisfied": sum(e["initially_satisfied"] for e in rows),
            "success_rate": count / len(seeds), "wilson95_low": low, "wilson95_high": high,
            **{f"mean_{key}": float(np.mean([e[key] for e in rows])) for key in
               ("steps", "initial_distance_m", "final_distance_m", "min_distance_m")}}
    paired = dict.fromkeys(("both_success", "both_failure", "repaired_only", "frozen_only"), 0)
    for seed in seeds:
        old, new = (by_variant[v][seed] for v in VARIANTS)
        if old["shared_scene_sha256"] != new["shared_scene_sha256"] or old["target_name"] != new["target_name"]:
            raise RuntimeError("Summary received unpaired scenes")
        a, b = old["success"], new["success"]
        paired["both_success" if a and b else "frozen_only" if a else "repaired_only" if b else "both_failure"] += 1
    paired["n_pairs"] = len(seeds)
    paired["success_rate_difference_repaired_minus_frozen"] = (paired["repaired_only"] - paired["frozen_only"]) / len(seeds)
    return {"variants": totals, "paired": paired}


def run_study(split="development", seeds: Sequence[int] | None = None, *, fixture_factories=None) -> dict:
    protocol = load_protocol()
    selection = select_seeds(protocol, split, seeds)
    fixture = fixture_factories is not None
    provenance = _runtime_provenance()
    if split == "held-out" and not fixture and (
            not provenance.get("git_revision") or provenance.get("git_dirty") is not False):
        raise RuntimeError("Real held-out evaluation requires a clean recorded Git revision")
    if fixture:
        if set(fixture_factories) != set(VARIANTS):
            raise ValueError("Fixtures must supply both declared variants")
        factories, manifests = fixture_factories, {"test_fixture": True}
    else:
        factories, manifests = _real_variants()
    started = time.perf_counter()
    report = {"schema_version": 2, "study_id": protocol["study_id"], "split": split,
        "test_fixture": fixture, "status": "running", "protocol": protocol, "protocol_sha256": _sha(protocol),
        "selected_seeds": [seed for _, seed in selection], "provenance": provenance,
        "implementation_manifests": manifests, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "trace_columns": TRACE_COLUMNS, "diagnostic_columns": DIAGNOSTIC_COLUMNS,
        "contact_columns": CONTACT_COLUMNS, "scenes": [], "episodes": []}
    for index, seed in selection:
        target_index = 1 + index % 2
        expected = None
        for variant in VARIANTS:
            env = None
            try:
                env = factories[variant]()
                observation, _ = env.reset(seed=seed)
                env.action_space.seed(seed)
                snapshot, common = _initial_state(env, observation, target_index, fixture)
                scene_hash = _sha(common)
                report["scenes"].append({"variant": variant, "env_seed": seed, "episode_index": index,
                    "shared_scene_sha256": scene_hash, "shared_scene": common,
                    "full_initial_state_sha256": _sha(snapshot), "full_initial_state": snapshot})
                if expected is not None and scene_hash != expected:
                    raise RuntimeError(f"Shared object scene differs for seed {seed}, variant {variant}")
                expected = scene_hash
                episode = _episode(env, target_index, snapshot, fixture)
                episode.update({"variant": variant, "env_seed": seed, "action_seed": seed,
                                "episode_index": index, "shared_scene_sha256": scene_hash})
                report["episodes"].append(episode)
            except Exception as exc:
                if isinstance(exc, _EpisodeExecutionError):
                    exc.episode.update({"variant": variant, "env_seed": seed, "action_seed": seed,
                                        "episode_index": index, "shared_scene_sha256": expected})
                    report["episodes"].append(exc.episode)
                report["status"] = "aborted"
                report["failure"] = {"variant": variant, "env_seed": seed, "episode_index": index,
                                     "error_type": type(exc).__name__, "error": str(exc)}
                report["elapsed_seconds"] = time.perf_counter() - started
                raise StudyExecutionError(f"Study aborted: {exc}", report) from exc
            finally:
                if env is not None:
                    env.close()
    if split == "held-out" and not fixture and _runtime_provenance() != provenance:
        report["status"] = "aborted"
        report["failure"] = {"error_type": "ProvenanceChanged",
                             "error": "Source/runtime provenance changed during evaluation"}
        report["elapsed_seconds"] = time.perf_counter() - started
        raise StudyExecutionError(report["failure"]["error"], report)
    report["summary"] = summarize(report["episodes"], report["selected_seeds"])
    report["elapsed_seconds"] = time.perf_counter() - started
    report["status"] = "complete"
    _canonical(report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("development", "held-out"), default="development")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        select_seeds(load_protocol(), args.split, args.seeds)
        if args.output.exists():
            raise ValueError("Output already exists; use a new path")
        if not (args.output.name.endswith(".json") or args.output.name.endswith(".json.gz")):
            raise ValueError("Output must end in .json or .json.gz")
    except ValueError as error:
        parser.error(str(error))
    try:
        report = run_study(args.split, args.seeds)
    except StudyExecutionError as exc:
        save_report(exc.report, args.output)
        parser.exit(2, f"{exc}; incomplete evidence saved without aggregate results\n")
    save_report(report, args.output)
    print(json.dumps({"semantic_sha256": semantic_digest(report), "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
