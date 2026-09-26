"""Execute grounded primitive plans with explicit target and outcome semantics.

Controllers and grounding use privileged simulator state. The pick/place checks
are motion/release heuristics, not proof of a secure grasp or a stable stack.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

import numpy as np

from planner.grounder import GrounderBase, SimGrounder, GroundingResult
from planner.task_parser import TaskPlan, SubTask
from policies.scripted import Phase, ScriptedPickPlace, ScriptedMoveTo


@dataclass
class StepResult:
    """A newly achieved success is separate from a pre-satisfied predicate."""
    sub_task: SubTask
    success: bool
    total_reward: float
    n_steps: int
    grounding: GroundingResult | None = None
    error: str = ""
    target_name: str | None = None
    source_name: str | None = None
    already_satisfied: bool = False
    terminated: bool = False
    truncated: bool = False
    skipped: bool = False
    initial_distance: float | None = None
    final_distance: float | None = None

    @property
    def completed(self) -> bool:
        """A satisfied precondition can permit the next step without new credit."""
        return self.success or self.already_satisfied


@dataclass
class ExecutionResult:
    instruction: str
    step_results: list[StepResult] = field(default_factory=list)
    overall_success: bool = False
    total_reward: float = 0.0
    total_steps: int = 0

    @property
    def n_sub_tasks(self) -> int:
        return len(self.step_results)

    @property
    def sub_task_success_rate(self) -> float:
        if not self.step_results:
            return 0.0
        return sum(s.success for s in self.step_results) / len(self.step_results)


class HierarchicalExecutor:
    """Execute named-object plans on an already-reset MultiObjectEnv.

    Reaching means EE within 3 cm of the named object's current centre. A
    predicate true before any action is returned as ``already_satisfied``, not
    ``success``. Pick/place retain the source identity within one execute call.
    A failure or terminal environment stops the plan; later steps are skipped.
    """
    REACH_DISTANCE = 0.03
    PICK_RISE = 0.08
    PICK_EE_DISTANCE = 0.06
    PLACE_DISTANCE = 0.05
    CLOSED_OPENING = 0.015
    RELEASED_OPENING = 0.025

    def __init__(
        self,
        grounder: GrounderBase | None = None,
        max_steps_per_subtask: int = 200,
        *,
        move_to_policy: Any | None = None,
    ):
        if (isinstance(max_steps_per_subtask, bool)
                or not isinstance(max_steps_per_subtask, Integral)
                or max_steps_per_subtask <= 0):
            raise ValueError("max_steps_per_subtask must be a positive integer")
        self.grounder = grounder or SimGrounder()
        self.max_steps = int(max_steps_per_subtask)
        self._pick_place = ScriptedPickPlace()
        self._move_to = ScriptedMoveTo() if move_to_policy is None else move_to_policy
        if not callable(getattr(self._move_to, "act", None)):
            raise TypeError("move_to_policy must provide act(info)")
        self._held_object_name: str | None = None

    def execute(self, env: Any, plan: TaskPlan, instruction: str = "") -> ExecutionResult:
        """Execute one plan; restore the environment horizon even on exceptions.

        Overall success requires every step completed and at least one new
        success. An entirely pre-satisfied plan therefore gains no control credit.
        Exceptions from the environment propagate; they are never success rows.
        """
        result = ExecutionResult(instruction=instruction)
        self._held_object_name = None
        if not plan.valid or plan.n_steps == 0:
            return result

        original_max_steps = getattr(env, "_max_episode_steps", None)
        elapsed = getattr(env, "_elapsed_steps", 0)
        was_truncated = original_max_steps is not None and elapsed >= original_max_steps
        was_terminated = bool(
            elapsed > 0 and callable(getattr(env, "_check_terminated", None))
            and env._check_terminated()
        )
        stop_reason = "Environment already ended; reset before executing" if (
            was_truncated or was_terminated
        ) else ""
        try:
            if original_max_steps is not None and not stop_reason:
                env._max_episode_steps = max(
                    original_max_steps, elapsed + plan.n_steps * self.max_steps,
                )
            for sub_task in plan.sub_tasks:
                if stop_reason:
                    step_result = StepResult(
                        sub_task, False, 0.0, 0, error=stop_reason, skipped=True,
                        terminated=was_terminated, truncated=was_truncated,
                    )
                else:
                    step_result = self._execute_subtask(env, sub_task)
                    was_terminated, was_truncated = step_result.terminated, step_result.truncated
                    if was_terminated or was_truncated:
                        stop_reason = "Skipped because the environment ended"
                    elif not step_result.completed:
                        stop_reason = "Skipped after a failed prerequisite"
                result.step_results.append(step_result)
                result.total_reward += step_result.total_reward
                result.total_steps += step_result.n_steps
        finally:
            if original_max_steps is not None:
                env._max_episode_steps = original_max_steps
            self._held_object_name = None

        result.overall_success = (
            all(s.completed for s in result.step_results)
            and any(s.success for s in result.step_results)
        )
        return result

    def _execute_subtask(self, env: Any, sub_task: SubTask) -> StepResult:
        if sub_task.primitive not in {"move_to", "pick", "place"}:
            return StepResult(sub_task, False, 0.0, 0, error=f"Unknown primitive: {sub_task.primitive}")
        if sub_task.primitive == "place" and self._held_object_name is None:
            return StepResult(sub_task, False, 0.0, 0, error="Place requires a successful pick in this plan")
        if sub_task.primitive == "pick" and self._held_object_name is not None:
            return StepResult(sub_task, False, 0.0, 0, error="Place the held object before picking another")
        if sub_task.primitive == "move_to" and self._held_object_name is not None:
            return StepResult(sub_task, False, 0.0, 0, error="Open-gripper move_to cannot carry a picked object; use place")

        grounding = self.grounder.ground(sub_task.target, self._build_scene_info(env))
        if not grounding.success or grounding.matched is None:
            return StepResult(sub_task, False, 0.0, 0, grounding,
                              error=f"Grounding failed: {grounding.error}")
        name = grounding.matched.name
        # Resolve name against the live scene, not a stale grounding snapshot.
        try:
            self._object_position(env, name)
        except LookupError as exc:
            return StepResult(sub_task, False, 0.0, 0, grounding,
                              error=str(exc), target_name=name)
        if sub_task.primitive == "move_to":
            return self._run_move_to(env, sub_task, name, grounding)
        if sub_task.primitive == "pick":
            return self._run_pick(env, sub_task, name, grounding)
        return self._run_place(env, sub_task, name, grounding)

    def _run_move_to(self, env: Any, sub_task: SubTask, name: str,
                     grounding: GroundingResult) -> StepResult:
        if callable(getattr(self._move_to, "reset", None)):
            self._move_to.reset()
        distance = self._distance_to_object(env, name)
        result = StepResult(sub_task, False, 0.0, 0, grounding, target_name=name,
                            initial_distance=distance, final_distance=distance)
        if distance < self.REACH_DISTANCE:
            result.already_satisfied = True
            return result

        for _ in range(self.max_steps):
            target_pos = self._object_position(env, name)
            info = self._get_policy_info(env, target_pos, target_name=name)
            _, reward, term, trunc, _ = env.step(self._move_to.act(info))
            self._record_step(result, reward, term, trunc)
            result.final_distance = self._distance_to_object(env, name)
            result.success = result.final_distance < self.REACH_DISTANCE
            if result.success or term or trunc:
                return result
        result.error = "Reaching step budget exhausted"
        return result

    def _run_pick(self, env: Any, sub_task: SubTask, name: str,
                  grounding: GroundingResult) -> StepResult:
        self._pick_place.reset()
        start_z = float(self._object_position(env, name)[2])
        result = StepResult(sub_task, False, 0.0, 0, grounding,
                            target_name=name, source_name=name)
        if self._gripper_opening(env) is None:
            result.error = "Pick requires observable gripper opening"
            return result
        close_commanded = False
        for _ in range(self.max_steps):
            obj_pos = self._object_position(env, name)
            info = self._get_policy_info(env, obj_pos, target_name=name)
            info["approach_z"] = start_z + self._pick_place.approach_height
            info["lift_target_z"] = start_z + self.PICK_RISE + self._pick_place.grasp_height_offset + 0.02
            action = self._pick_place.act(info)
            close_commanded |= bool(action[3] < 0)
            _, reward, term, trunc, _ = env.step(action)
            self._record_step(result, reward, term, trunc)
            obj_pos = self._object_position(env, name)
            opening = self._gripper_opening(env)
            result.success = bool(
                close_commanded and opening is not None and opening < self.CLOSED_OPENING
                and obj_pos[2] - start_z >= self.PICK_RISE
                and np.linalg.norm(self._get_ee_pos(env) - obj_pos) < self.PICK_EE_DISTANCE
            )
            if result.success:
                self._held_object_name = name
            if result.success or term or trunc:
                return result
        result.error = "Pick motion/gripper criteria were not achieved"
        return result

    def _run_place(self, env: Any, sub_task: SubTask, target_name: str,
                   grounding: GroundingResult) -> StepResult:
        source_name = self._held_object_name
        result = StepResult(sub_task, False, 0.0, 0, grounding,
                            target_name=target_name, source_name=source_name)
        if source_name == target_name:
            result.error = "Cannot place an object relative to itself"
            return result
        if self._gripper_opening(env) is None:
            result.error = "Place requires observable gripper opening"
            return result
        self._pick_place.reset()
        self._pick_place.phase = Phase.MOVE
        released = False
        for _ in range(self.max_steps):
            source_pos = self._object_position(env, source_name)
            target_pos = self._object_position(env, target_name)
            info = self._get_policy_info(env, source_pos, goal=target_pos,
                                         target_name=source_name)
            # Stay above the current source/destination while translating.
            info["lift_target_z"] = max(source_pos[2], target_pos[2] + self._pick_place.lift_height)
            phase_before = self._pick_place.phase
            action = self._pick_place.act(info)
            released |= bool(phase_before in {Phase.RELEASE, Phase.DONE} and action[3] > 0)
            _, reward, term, trunc, _ = env.step(action)
            self._record_step(result, reward, term, trunc)
            source_pos = self._object_position(env, source_name)
            target_pos = self._object_position(env, target_name)
            opening = self._gripper_opening(env)
            result.success = bool(
                released and opening is not None and opening > self.RELEASED_OPENING
                and np.linalg.norm(source_pos - target_pos) < self.PLACE_DISTANCE
            )
            if result.success:
                self._held_object_name = None
            if result.success or term or trunc:
                return result
        result.error = "Source placement/release criteria were not achieved"
        return result

    @staticmethod
    def _record_step(result: StepResult, reward: float, terminated: bool, truncated: bool) -> None:
        if not np.isfinite(reward):
            raise ValueError("Non-finite reward during plan execution")
        result.total_reward += float(reward)
        result.n_steps += 1
        result.terminated = bool(terminated)
        result.truncated = bool(truncated)

    @staticmethod
    def _position(value: Any) -> np.ndarray:
        pos = np.asarray(value, dtype=float)
        if pos.shape != (3,) or not np.all(np.isfinite(pos)):
            raise ValueError("Expected a finite 3D simulator position")
        return pos.copy()

    @classmethod
    def _get_ee_pos(cls, env: Any) -> np.ndarray:
        if not hasattr(env, "ee_pos"):
            raise ValueError("Environment does not expose ee_pos")
        return cls._position(env.ee_pos)

    @classmethod
    def _object_position(cls, env: Any, name: str) -> np.ndarray:
        indices = [i for i, spec in enumerate(getattr(env, "_obj_specs", [])) if spec.name == name]
        if len(indices) != 1 or not callable(getattr(env, "object_pos", None)):
            raise LookupError(f"Expected one live object named {name!r}; found {len(indices)}")
        return cls._position(env.object_pos(indices[0]))

    @classmethod
    def _distance_to_object(cls, env: Any, name: str) -> float:
        return float(np.linalg.norm(cls._get_ee_pos(env) - cls._object_position(env, name)))

    @staticmethod
    def _gripper_opening(env: Any) -> float | None:
        if hasattr(env, "gripper_opening"):
            opening = float(env.gripper_opening)
        elif hasattr(env, "data") and hasattr(env.data, "qpos") and len(env.data.qpos) >= 9:
            opening = float(np.mean(env.data.qpos[7:9]))
        else:
            return None
        return opening if np.isfinite(opening) else None

    def _get_policy_info(self, env: Any, target_pos: np.ndarray,
                         goal: np.ndarray | None = None,
                         target_name: str | None = None) -> dict:
        info = {"ee_pos": self._get_ee_pos(env), "obj_pos": target_pos.copy()}
        if target_name is not None:
            info["target_name"] = target_name
        if goal is not None:
            info["goal_pos"] = goal.copy()
        return info

    @classmethod
    def _build_scene_info(cls, env: Any) -> dict:
        return {"objects": [
            {"name": spec.name, "color": spec.color_name, "shape": spec.shape,
             "position": cls._position(env.object_pos(i)).tolist()}
            for i, spec in enumerate(getattr(env, "_obj_specs", []))
        ]}
