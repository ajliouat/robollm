"""
Task parser — validate and normalise VLM output into executable sub-tasks.

Handles malformed VLM output, validates primitive names,
and produces a clean sequence for the executor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from planner.prompt_templates import VALID_PRIMITIVES

_COLOR_ALIASES = {
    "crimson": "red",
    "scarlet": "red",
    "azure": "blue",
    "navy": "blue",
    "cyan": "blue",
    "lime": "green",
    "emerald": "green",
    "amber": "yellow",
    "gold": "yellow",
    "tangerine": "orange",
    "violet": "purple",
    "magenta": "purple",
    "lavender": "purple",
}

VALID_COLORS = {"red", "green", "blue", "yellow", "orange", "purple"}


@dataclass
class SubTask:
    """A single validated sub-task in the execution plan."""
    primitive: str          # "move_to", "pick", or "place"
    target: str             # object color/name or position description
    description: str = ""   # human-readable step description
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "primitive": self.primitive,
            "target": self.target,
            "description": self.description,
        }


@dataclass
class TaskPlan:
    """Validated sequence of sub-tasks forming a complete plan."""
    sub_tasks: list[SubTask]
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def n_steps(self) -> int:
        return len(self.sub_tasks)

    def primitives(self) -> list[str]:
        return [st.primitive for st in self.sub_tasks]

    def targets(self) -> list[str]:
        return [st.target for st in self.sub_tasks]

    def to_dict(self) -> dict:
        return {
            "sub_tasks": [st.to_dict() for st in self.sub_tasks],
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def normalise_color(name: str) -> str:
    """Normalise color name, handling aliases."""
    lower = name.lower().strip()
    if lower in VALID_COLORS:
        return lower
    return _COLOR_ALIASES.get(lower, lower)


def parse_subtasks(raw_subtasks: list[dict[str, Any]]) -> TaskPlan:
    """Parse and validate a list of raw sub-task dicts.

    Args:
        raw_subtasks: List of dicts from VLM output, each with
                      "primitive", "target", and optionally "description".

    Returns:
        TaskPlan with validated sub-tasks and any errors.
    """
    if not raw_subtasks:
        return TaskPlan(
            sub_tasks=[], valid=False,
            errors=["Empty sub-task list"],
        )

    sub_tasks: list[SubTask] = []
    errors: list[str] = []
    warnings: list[str] = []

    for i, raw in enumerate(raw_subtasks):
        if not isinstance(raw, dict):
            errors.append(f"Step {i}: not a dict, got {type(raw).__name__}")
            continue

        # Validate primitive
        primitive = raw.get("primitive", "").lower().strip()
        if primitive not in VALID_PRIMITIVES:
            errors.append(
                f"Step {i}: unknown primitive '{primitive}'. "
                f"Valid: {sorted(VALID_PRIMITIVES)}"
            )
            continue

        # Validate target
        target = str(raw.get("target", "")).strip()
        if not target:
            errors.append(f"Step {i}: missing target")
            continue

        # Normalise color targets
        normalised = normalise_color(target)
        if normalised != target.lower():
            warnings.append(f"Step {i}: normalised '{target}' → '{normalised}'")
            target = normalised
        else:
            target = target.lower()

        description = str(raw.get("description", ""))

        sub_tasks.append(SubTask(
            primitive=primitive,
            target=target,
            description=description,
        ))

    # Validate ordering constraints
    if sub_tasks:
        ordering_errors = _validate_ordering(sub_tasks)
        warnings.extend(ordering_errors)

    valid = len(errors) == 0 and len(sub_tasks) > 0
    return TaskPlan(
        sub_tasks=sub_tasks,
        valid=valid,
        errors=errors,
        warnings=warnings,
    )


def _validate_ordering(sub_tasks: list[SubTask]) -> list[str]:
    """Check that sub-task ordering makes physical sense."""
    warnings: list[str] = []
    holding: str | None = None

    for i, st in enumerate(sub_tasks):
        if st.primitive == "pick":
            if holding is not None:
                warnings.append(
                    f"Step {i}: picking '{st.target}' but already "
                    f"holding '{holding}'"
                )
            holding = st.target

        elif st.primitive == "place":
            if holding is None:
                warnings.append(
                    f"Step {i}: placing but not holding anything"
                )
            holding = None

        elif st.primitive == "move_to":
            pass  # Always valid

    return warnings
