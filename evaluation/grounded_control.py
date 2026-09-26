"""Frozen, paired CPU study of instantaneous proximity to a named live object.

Every arm uses HierarchicalExecutor's same target identity, success and stopping
logic. Only action selection changes. No model, renderer or GPU is involved.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from evaluation.benchmark import _runtime_provenance, wilson_interval
from evaluation.pipeline import HierarchicalExecutor
from planner.grounder import SimGrounder
from planner.task_parser import SubTask, TaskPlan
from policies.scripted import ScriptedMoveTo

PROTOCOL_PATH = Path(__file__).with_name("control_protocol.json")
ARMS = (
    "grounded_executor",
    "first_object_action_ablation",
    "random_xyz_open",
    "zero_xyz_open",
)
TRACE_COLUMNS = [
    "step",
    "action_x",
    "action_y",
    "action_z",
    "gripper",
    "ee_x",
    "ee_y",
    "ee_z",
    "target_x",
    "target_y",
    "target_z",
    "object0_x",
    "object0_y",
    "object0_z",
    "named_target_distance_m",
    "object0_distance_m",
    "target_displacement_m",
    "terminated",
    "truncated",
]


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def load_protocol(path: str | Path = PROTOCOL_PATH) -> dict:
    protocol = json.loads(Path(path).read_text())
    expected = {
        "schema_version": 1,
        "study_id": "robollm-named-target-proximity-v1",
        "development_seeds": [17, 18],
        "held_out_seeds": list(range(260926000, 260926100)),
        "arms": list(ARMS),
        "controller": {
            "gain": 8.0,
            "threshold_m": 0.03,
            "horizon_steps": 200,
            "gripper_action": 1.0,
        },
        "environment": {
            "class": "envs.multi_object_env.MultiObjectEnv",
            "n_objects": 3,
            "render_mode": None,
            "max_episode_steps": 200,
            "settling_steps": 0,
            "control_dt_seconds": 0.05,
        },
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"Protocol field {key!r} does not match the frozen study")
    if protocol.get("target_selection", {}).get("indices") != [1, 2]:
        raise ValueError("Frozen targets must alternate indices 1 and 2")
    return protocol


def select_seeds(
    protocol: dict, split: str, seeds: Sequence[int] | None = None
) -> list[tuple[int, int]]:
    """Return (original split index, seed); held-out runs always use all 100."""
    if split not in ("smoke", "held-out"):
        raise ValueError("split must be 'smoke' or 'held-out'")
    declared = protocol["development_seeds" if split == "smoke" else "held_out_seeds"]
    selected = list(declared if seeds is None else seeds)
    if not selected or any(type(seed) is not int for seed in selected):
        raise ValueError("seeds must be a nonempty list of integers")
    if len(set(selected)) != len(selected) or any(
        seed not in declared for seed in selected
    ):
        raise ValueError("seeds must be unique members of the declared split")
    if split == "held-out" and selected != declared:
        raise ValueError("A held-out run must retain all 100 frozen seeds in order")
    return [(declared.index(seed), seed) for seed in selected]


def _vector(value: Any, size: int = 3) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"Expected a finite {size}-vector, got {result}")
    return result.copy()


def _snapshot(env: Any, observation: Any) -> dict:
    objects = []
    for index, spec in enumerate(env._obj_specs):
        objects.append(
            {
                "index": index,
                "name": spec.name,
                "color": spec.color_name,
                "shape": spec.shape,
                "position": _vector(env.object_pos(index)).tolist(),
                "size_attr": getattr(spec, "size_attr", None),
                "mass": getattr(spec, "mass", None),
            }
        )
    if len(objects) != 3 or len({obj["name"] for obj in objects}) != 3:
        raise ValueError("Study requires exactly three uniquely named objects")
    snapshot = {
        "objects": objects,
        "ee_position": _vector(env.ee_pos).tolist(),
        "observation": np.asarray(observation).tolist(),
        "simulator_state": {},
    }
    if hasattr(env, "data"):
        for name in (
            "qpos",
            "qvel",
            "act",
            "ctrl",
            "qacc_warmstart",
            "mocap_pos",
            "mocap_quat",
        ):
            snapshot["simulator_state"][name] = np.asarray(
                getattr(env.data, name)
            ).tolist()
        snapshot["simulator_state"]["time"] = float(env.data.time)
    _canonical(snapshot)  # Reject nonfinite snapshot values before any actions.
    return snapshot


class _TraceEnv:
    """Observe calls without introducing a second success/reward implementation."""

    def __init__(self, env: Any, target_index: int, horizon: int):
        self._env = env
        self.target_index = target_index
        self.horizon = horizon
        self.target_initial = _vector(env.object_pos(target_index))
        self.rows: list[list] = []
        self._ended = False

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def _max_episode_steps(self):
        return self._env._max_episode_steps

    @_max_episode_steps.setter
    def _max_episode_steps(self, value):
        # The executor may expand a multi-subtask horizon. This study is one
        # subtask and keeps its predeclared 200-step environment limit intact.
        self._env._max_episode_steps = min(value, self.horizon)

    def step(self, action):
        if self._ended or len(self.rows) >= self.horizon:
            raise RuntimeError("Attempted to step after the study episode ended")
        action = _vector(action, 4)
        if np.any(np.abs(action) > 1) or action[3] != 1.0:
            raise ValueError("Study actions must be bounded and keep gripper open")
        observation, reward, terminated, truncated, info = self._env.step(action)
        if not math.isfinite(float(reward)):
            raise ValueError("Environment returned a nonfinite reward")
        ee = _vector(self._env.ee_pos)
        target = _vector(self._env.object_pos(self.target_index))
        object0 = _vector(self._env.object_pos(0))
        self.rows.append(
            [
                len(self.rows) + 1,
                *action.tolist(),
                *ee.tolist(),
                *target.tolist(),
                *object0.tolist(),
                float(np.linalg.norm(ee - target)),
                float(np.linalg.norm(ee - object0)),
                float(np.linalg.norm(target - self.target_initial)),
                bool(terminated),
                bool(truncated),
            ]
        )
        self._ended = bool(terminated or truncated)
        return observation, reward, terminated, truncated, info


class _FirstObjectActions:
    def __init__(self, env: Any, controller: dict):
        self.env = env
        self.policy = ScriptedMoveTo(
            gain=controller["gain"], threshold=controller["threshold_m"]
        )

    def reset(self):
        self.policy.reset()

    def act(self, info):
        return self.policy.act(
            {"ee_pos": info["ee_pos"], "obj_pos": self.env.object_pos(0)}
        )


class _RandomOpenActions:
    def __init__(self, env: Any):
        self.env = env

    def act(self, info):
        action = np.asarray(self.env.action_space.sample(), dtype=np.float64).copy()
        action[3] = 1.0
        return action


class _ZeroOpenActions:
    def act(self, info):
        return np.array([0.0, 0.0, 0.0, 1.0])


def _episode(
    env: Any, arm: str, target_index: int, snapshot: dict, controller: dict
) -> dict:
    target_name = snapshot["objects"][target_index]["name"]
    traced = _TraceEnv(env, target_index, controller["horizon_steps"])
    policy = {
        "grounded_executor": None,
        "first_object_action_ablation": _FirstObjectActions(traced, controller),
        "random_xyz_open": _RandomOpenActions(traced),
        "zero_xyz_open": _ZeroOpenActions(),
    }[arm]
    executor = HierarchicalExecutor(
        grounder=SimGrounder(),
        max_steps_per_subtask=controller["horizon_steps"],
        move_to_policy=policy,
    )
    if executor.REACH_DISTANCE != controller["threshold_m"]:
        raise RuntimeError("Executor success threshold differs from frozen protocol")
    if arm == "grounded_executor":
        if (
            executor._move_to.gain != controller["gain"]
            or executor._move_to.threshold != controller["threshold_m"]
        ):
            raise RuntimeError(
                "Executor defaults differ from frozen controller parameters"
            )
    plan = TaskPlan(sub_tasks=[SubTask("move_to", target_name)], valid=True)
    result = executor.execute(traced, plan, instruction=f"move to {target_name}")
    if len(result.step_results) != 1:
        raise RuntimeError("Expected exactly one reaching subtask result")
    step = result.step_results[0]
    if (
        step.grounding is None
        or not step.grounding.success
        or step.grounding.matched.name != target_name
    ):
        raise RuntimeError(
            "Grounder did not resolve the exact declared target identity"
        )
    if step.target_name != target_name or step.error:
        raise RuntimeError(f"Executor target failure: {step.error or step.target_name}")
    if step.n_steps != len(traced.rows) or result.total_steps != len(traced.rows):
        raise RuntimeError("Executor step count disagrees with observed actions")

    ee_initial = np.asarray(snapshot["ee_position"])
    initial = float(np.linalg.norm(ee_initial - traced.target_initial))
    initial_object0 = float(
        np.linalg.norm(ee_initial - snapshot["objects"][0]["position"])
    )
    distances = [initial] + [row[14] for row in traced.rows]
    initial_predicate = initial < controller["threshold_m"]
    observed_success = not initial_predicate and any(
        distance < controller["threshold_m"] for distance in distances[1:]
    )
    if (
        bool(step.already_satisfied) != initial_predicate
        or bool(step.success) != observed_success
    ):
        raise RuntimeError(
            "Executor success disagrees with the recorded named-target trace"
        )
    for expected, actual in (
        (initial, step.initial_distance),
        (distances[-1], step.final_distance),
    ):
        if actual is None or not math.isclose(
            expected, actual, rel_tol=0, abs_tol=1e-12
        ):
            raise RuntimeError("Executor distances disagree with recorded positions")
    if initial_predicate and traced.rows:
        raise RuntimeError("Initially satisfied episodes must not perform actions")
    return {
        "arm": arm,
        "target_index": target_index,
        "target_name": target_name,
        "grounded_name": step.grounding.matched.name,
        "success": bool(step.success),
        "initially_satisfied": initial_predicate,
        "initial_distance_m": initial,
        "final_distance_m": distances[-1],
        "min_distance_m": min(distances),
        "initial_object0_distance_m": initial_object0,
        "final_object0_distance_m": traced.rows[-1][15]
        if traced.rows
        else initial_object0,
        "min_object0_distance_m": min(
            [initial_object0] + [row[15] for row in traced.rows]
        ),
        "final_target_displacement_m": traced.rows[-1][16] if traced.rows else 0.0,
        "max_target_displacement_m": max([0.0] + [row[16] for row in traced.rows]),
        "steps": len(traced.rows),
        "terminated": bool(step.terminated),
        "truncated": bool(step.truncated),
        "stop_reason": (
            "initially_satisfied"
            if initial_predicate
            else "target_reached"
            if step.success
            else "environment_done"
            if step.terminated or step.truncated
            else "horizon"
        ),
        "trace": traced.rows,
    }


def _summary(episodes: list[dict], n_episodes: int) -> dict:
    arms = {}
    for arm in ARMS:
        selected = [episode for episode in episodes if episode["arm"] == arm]
        if len(selected) != n_episodes:
            raise RuntimeError("Missing paired episode in summary")
        successes = sum(episode["success"] for episode in selected)
        low, high = wilson_interval(successes, n_episodes)
        arms[arm] = {
            "n_episodes": n_episodes,
            "n_successes": successes,
            "n_initially_satisfied": sum(
                episode["initially_satisfied"] for episode in selected
            ),
            "success_rate": successes / n_episodes,
            "wilson95_low": low,
            "wilson95_high": high,
            "mean_steps": float(np.mean([episode["steps"] for episode in selected])),
            "mean_final_distance_m": float(
                np.mean([episode["final_distance_m"] for episode in selected])
            ),
            "mean_min_distance_m": float(
                np.mean([episode["min_distance_m"] for episode in selected])
            ),
        }
    paired = dict.fromkeys(
        ("both_success", "both_failure", "grounded_only", "ablation_only"), 0
    )
    ground = {
        episode["env_seed"]: episode["success"]
        for episode in episodes
        if episode["arm"] == ARMS[0]
    }
    ablation = {
        episode["env_seed"]: episode["success"]
        for episode in episodes
        if episode["arm"] == ARMS[1]
    }
    for seed, success in ground.items():
        key = (
            "both_success"
            if success and ablation[seed]
            else "grounded_only"
            if success
            else "ablation_only"
            if ablation[seed]
            else "both_failure"
        )
        paired[key] += 1
    paired["n_pairs"] = n_episodes
    paired["success_rate_difference"] = (
        paired["grounded_only"] - paired["ablation_only"]
    ) / n_episodes
    return {"arms": arms, "grounded_vs_first_object_action_ablation": paired}


def run_study(
    split: str = "smoke",
    seeds: Sequence[int] | None = None,
    *,
    env_factory: Callable | None = None,
) -> dict:
    """Run a bounded frozen split. An injected environment is labeled a fixture."""
    protocol = load_protocol()
    selection = select_seeds(protocol, split, seeds)
    is_fixture = env_factory is not None
    environment_snapshot = None
    if env_factory is None:
        # v1 keeps its original simulator even after the default environment is
        # repaired. Its historical protocol/results must not silently change.
        from evaluation.frozen_v1 import create_environment, manifest

        env_factory = create_environment
        environment_snapshot = manifest()

    started = time.perf_counter()
    provenance = _runtime_provenance()
    if (
        split == "held-out"
        and not is_fixture
        and (
            not provenance.get("git_revision")
            or provenance.get("git_dirty") is not False
        )
    ):
        raise RuntimeError(
            "Real held-out evaluation requires a clean recorded Git revision"
        )
    report = {
        "schema_version": 1,
        "study_id": protocol["study_id"],
        "split": split,
        "test_fixture": is_fixture,
        "environment_snapshot": environment_snapshot,
        "protocol": protocol,
        "protocol_sha256": hashlib.sha256(_canonical(protocol)).hexdigest(),
        "selected_seeds": [seed for _, seed in selection],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance,
        "trace_columns": TRACE_COLUMNS,
        "scenes": [],
        "episodes": [],
    }
    for episode_index, seed in selection:
        target_index = 1 + episode_index % 2
        expected_scene = None
        for arm in ARMS:
            env = env_factory()
            try:
                observation, _ = env.reset(seed=seed)
                env.action_space.seed(seed)
                snapshot = _snapshot(env, observation)
                scene_hash = hashlib.sha256(_canonical(snapshot)).hexdigest()
                if expected_scene is None:
                    expected_scene = scene_hash
                    report["scenes"].append(
                        {
                            "episode_index": episode_index,
                            "env_seed": seed,
                            "target_index": target_index,
                            "target_name": snapshot["objects"][target_index]["name"],
                            "initial_scene_sha256": scene_hash,
                            "initial_snapshot": snapshot,
                        }
                    )
                elif scene_hash != expected_scene:
                    raise RuntimeError(
                        f"Paired initial scene mismatch for seed {seed}, arm {arm}"
                    )
                episode = _episode(
                    env, arm, target_index, snapshot, protocol["controller"]
                )
                episode.update(
                    {
                        "episode_index": episode_index,
                        "env_seed": seed,
                        "action_seed": seed,
                        "initial_scene_sha256": scene_hash,
                    }
                )
                report["episodes"].append(episode)
            finally:
                env.close()
    report["summary"] = _summary(report["episodes"], len(selection))
    report["elapsed_seconds"] = time.perf_counter() - started
    _canonical(report)
    return report


def semantic_payload(report: dict) -> dict:
    """Exact replay content, excluding only timing and machine/code metadata."""
    return {
        key: value
        for key, value in report.items()
        if key not in {"timestamp_utc", "elapsed_seconds", "provenance"}
    }


def semantic_digest(report: dict) -> str:
    return hashlib.sha256(_canonical(semantic_payload(report))).hexdigest()


def save_report(report: dict, path: str | Path) -> None:
    """Exclusively publish a complete JSON artifact; never overwrite prior runs."""
    path = Path(path)
    if not (path.name.endswith(".json") or path.name.endswith(".json.gz")):
        raise ValueError("Output must end in .json or .json.gz")
    payload = _canonical(report) + b"\n"
    if path.name.endswith(".gz"):
        payload = gzip.compress(payload, mtime=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            staged = Path(handle.name)
            handle.write(payload)
        os.link(staged, path)  # Atomic and fails if destination already exists.
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("smoke", "held-out"), default="smoke")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Smoke only: subset of 17,18; target index remains fixed",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New .json or .json.gz artifact; existing files are protected",
    )
    args = parser.parse_args(argv)
    try:
        select_seeds(load_protocol(), args.split, args.seeds)
        if args.output.exists():
            raise ValueError("Output already exists; choose a new artifact path")
        if not (
            args.output.name.endswith(".json") or args.output.name.endswith(".json.gz")
        ):
            raise ValueError("Output must end in .json or .json.gz")
    except ValueError as error:
        parser.error(str(error))
    report = run_study(args.split, args.seeds)
    save_report(report, args.output)
    print(
        json.dumps(
            {
                "split": args.split,
                "semantic_sha256": semantic_digest(report),
                "summary": report["summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
