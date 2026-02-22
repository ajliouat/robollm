"""
High-level planner: instruction → sub-task plan.

Ties together VLM wrapper, prompt templates, and task parser.
"""

from __future__ import annotations

from typing import Any

from planner.task_parser import TaskPlan, parse_subtasks
from planner.vlm_wrapper import VLMBase, VLMResponse


class Planner:
    """Orchestrates VLM-based task decomposition.

    Usage:
        vlm = MockVLM()   # or TransformersVLM(...)
        planner = Planner(vlm)
        plan = planner.plan("pick up the red block and place it on the blue one")
        for step in plan.sub_tasks:
            print(step.primitive, step.target)
    """

    def __init__(self, vlm: VLMBase):
        self.vlm = vlm
        self._history: list[dict] = []

    def plan(
        self,
        instruction: str,
        image: Any | None = None,
    ) -> TaskPlan:
        """Decompose instruction into a validated sub-task plan.

        Args:
            instruction: Natural language instruction.
            image: Optional scene image (numpy H×W×3 RGB).

        Returns:
            TaskPlan with validated sub-tasks.
        """
        response: VLMResponse = self.vlm.decompose(image, instruction)

        task_plan = parse_subtasks(response.sub_tasks)

        self._history.append({
            "instruction": instruction,
            "vlm_model": response.model_name,
            "raw_text": response.raw_text,
            "plan": task_plan.to_dict(),
            "valid": task_plan.valid,
        })

        return task_plan

    @property
    def history(self) -> list[dict]:
        """Return planning history for logging/debugging."""
        return list(self._history)

    def clear_history(self):
        self._history.clear()
