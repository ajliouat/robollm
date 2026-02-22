"""
Hierarchical executor — runs sub-task plans on MuJoCo environments.

Takes a TaskPlan from the Planner and executes each sub-task using
the appropriate RL policy or scripted baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from planner.grounder import GrounderBase, SimGrounder, GroundingResult
from planner.task_parser import TaskPlan, SubTask
from policies.scripted import ScriptedPickPlace, ScriptedMoveTo


@dataclass
class StepResult:
    """Result of executing a single sub-task."""
    sub_task: SubTask
    success: bool
    total_reward: float
    n_steps: int
    grounding: GroundingResult | None = None
    error: str = ""


@dataclass
class ExecutionResult:
    """Result of executing a full task plan."""
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
        return sum(1 for s in self.step_results if s.success) / len(self.step_results)


class HierarchicalExecutor:
    """Execute sub-task plans on a multi-object MuJoCo environment.

    Combines planner output with object grounding and primitive policies
    to form the complete instruction → execution pipeline.

    Usage:
        from planner import Planner, MockVLM, SimGrounder
        from evaluation.pipeline import HierarchicalExecutor

        env = MultiObjectEnv(n_objects=3)
        executor = HierarchicalExecutor(grounder=SimGrounder())
        planner = Planner(MockVLM())

        plan = planner.plan("pick up the red block and place it on the blue one")
        result = executor.execute(env, plan, "pick up the red block...")
    """

    def __init__(
        self,
        grounder: GrounderBase | None = None,
        max_steps_per_subtask: int = 200,
    ):
        self.grounder = grounder or SimGrounder()
        self.max_steps = max_steps_per_subtask

        # Policies for each primitive
        self._pick_place = ScriptedPickPlace()
        self._move_to = ScriptedMoveTo()

    def execute(
        self,
        env: Any,
        plan: TaskPlan,
        instruction: str = "",
    ) -> ExecutionResult:
        """Execute a task plan on the environment.

        Args:
            env: Gymnasium env (should already be reset).
            plan: Validated task plan from Planner.
            instruction: Original instruction string.

        Returns:
            ExecutionResult with per-step and overall metrics.
        """
        result = ExecutionResult(instruction=instruction)

        if not plan.valid or plan.n_steps == 0:
            result.overall_success = False
            return result

        for sub_task in plan.sub_tasks:
            step_result = self._execute_subtask(env, sub_task)
            result.step_results.append(step_result)
            result.total_reward += step_result.total_reward
            result.total_steps += step_result.n_steps

            if not step_result.success:
                # Continue attempting remaining sub-tasks
                pass

        # Overall success: all sub-tasks succeeded
        result.overall_success = all(
            s.success for s in result.step_results
        )
        return result

    def _execute_subtask(
        self, env: Any, sub_task: SubTask,
    ) -> StepResult:
        """Execute a single sub-task on the environment."""

        # Build scene info for grounding
        scene_info = self._build_scene_info(env)

        # Ground the target
        grounding = self.grounder.ground(
            sub_task.target, scene_info,
        )

        if not grounding.success:
            return StepResult(
                sub_task=sub_task,
                success=False,
                total_reward=0.0,
                n_steps=0,
                grounding=grounding,
                error=f"Grounding failed: {grounding.error}",
            )

        target_pos = grounding.matched.position.copy()

        # Select policy based on primitive
        if sub_task.primitive == "move_to":
            return self._run_move_to(env, sub_task, target_pos, grounding)
        elif sub_task.primitive == "pick":
            return self._run_pick(env, sub_task, target_pos, grounding)
        elif sub_task.primitive == "place":
            return self._run_place(env, sub_task, target_pos, grounding)
        else:
            return StepResult(
                sub_task=sub_task,
                success=False,
                total_reward=0.0,
                n_steps=0,
                error=f"Unknown primitive: {sub_task.primitive}",
            )

    def _run_move_to(
        self, env: Any, sub_task: SubTask,
        target_pos: np.ndarray, grounding: GroundingResult,
    ) -> StepResult:
        """Execute move_to primitive."""
        self._move_to.reset()
        total_reward = 0.0

        for step in range(self.max_steps):
            info = self._get_policy_info(env, target_pos)
            action = self._move_to.act(info)
            _, reward, term, trunc, info_step = env.step(action)
            total_reward += reward

            # Check success: EE within 3cm of target
            ee_pos = self._get_ee_pos(env)
            dist = float(np.linalg.norm(ee_pos - target_pos))
            if dist < 0.03 or term or trunc:
                return StepResult(
                    sub_task=sub_task,
                    success=dist < 0.03,
                    total_reward=total_reward,
                    n_steps=step + 1,
                    grounding=grounding,
                )

        return StepResult(
            sub_task=sub_task,
            success=False,
            total_reward=total_reward,
            n_steps=self.max_steps,
            grounding=grounding,
        )

    def _run_pick(
        self, env: Any, sub_task: SubTask,
        target_pos: np.ndarray, grounding: GroundingResult,
    ) -> StepResult:
        """Execute pick primitive (approach + grasp + lift)."""
        self._pick_place.reset()
        total_reward = 0.0

        for step in range(self.max_steps):
            info = self._get_policy_info(env, target_pos)
            action = self._pick_place.act(info)
            _, reward, term, trunc, info_step = env.step(action)
            total_reward += reward

            # Consider pick successful if object is lifted
            obj_pos = self._get_first_obj_pos(env)
            if obj_pos is not None and obj_pos[2] > 0.50:
                return StepResult(
                    sub_task=sub_task,
                    success=True,
                    total_reward=total_reward,
                    n_steps=step + 1,
                    grounding=grounding,
                )

            if term or trunc:
                break

        return StepResult(
            sub_task=sub_task,
            success=False,
            total_reward=total_reward,
            n_steps=min(step + 1, self.max_steps),
            grounding=grounding,
        )

    def _run_place(
        self, env: Any, sub_task: SubTask,
        target_pos: np.ndarray, grounding: GroundingResult,
    ) -> StepResult:
        """Execute place primitive — move to target, lower, release."""
        # Reuse pick_place controller (it has move and release phases)
        # Set it to MOVE phase since we should already be holding
        from policies.scripted import Phase
        self._pick_place.phase = Phase.MOVE
        total_reward = 0.0

        for step in range(self.max_steps):
            info = self._get_policy_info(env, target_pos, goal=target_pos)
            action = self._pick_place.act(info)
            _, reward, term, trunc, info_step = env.step(action)
            total_reward += reward

            # Place success: object near target and on table
            obj_pos = self._get_first_obj_pos(env)
            if obj_pos is not None:
                dist = float(np.linalg.norm(obj_pos[:2] - target_pos[:2]))
                if dist < 0.05 and obj_pos[2] < 0.50:
                    return StepResult(
                        sub_task=sub_task,
                        success=True,
                        total_reward=total_reward,
                        n_steps=step + 1,
                        grounding=grounding,
                    )

            if term or trunc:
                break

        return StepResult(
            sub_task=sub_task,
            success=False,
            total_reward=total_reward,
            n_steps=min(step + 1, self.max_steps),
            grounding=grounding,
        )

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _get_ee_pos(env: Any) -> np.ndarray:
        """Get end-effector position from env."""
        if hasattr(env, "ee_pos"):
            return env.ee_pos
        return np.zeros(3)

    @staticmethod
    def _get_first_obj_pos(env: Any) -> np.ndarray | None:
        """Get first object position from env."""
        if hasattr(env, "object_pos"):
            return env.object_pos(0)
        return None

    def _get_policy_info(
        self, env: Any, target_pos: np.ndarray,
        goal: np.ndarray | None = None,
    ) -> dict:
        """Build info dict for scripted policies."""
        info: dict[str, Any] = {
            "ee_pos": self._get_ee_pos(env),
            "obj_pos": target_pos.copy(),
        }
        if goal is not None:
            info["goal_pos"] = goal.copy()
        elif hasattr(env, "object_pos"):
            info["obj_pos"] = env.object_pos(0)
        return info

    @staticmethod
    def _build_scene_info(env: Any) -> dict:
        """Extract scene info for grounder from env."""
        scene: dict[str, Any] = {"objects": []}

        if hasattr(env, "_obj_specs"):
            for i, spec in enumerate(env._obj_specs):
                pos = env.object_pos(i) if hasattr(env, "object_pos") else np.zeros(3)
                scene["objects"].append({
                    "name": spec.name,
                    "color": spec.color_name,
                    "shape": spec.shape,
                    "position": pos.tolist(),
                })

        return scene
