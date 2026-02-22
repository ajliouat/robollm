"""
Prompt templates for VLM task decomposition.

Structured prompts that convert natural language instructions into
a format the VLM can reliably decompose into sub-task sequences.
"""

from __future__ import annotations

VALID_PRIMITIVES = {"move_to", "pick", "place"}

DECOMPOSITION_SYSTEM = """\
You are a robotic task planner. Given a natural language instruction about \
manipulating objects on a tabletop, decompose it into a sequence of primitive \
sub-tasks.

Available primitives:
- move_to: Move the robot gripper close to a target object
- pick: Grasp and lift a target object
- place: Place the currently held object at a target position

Available object colors: red, green, blue, yellow, orange, purple
Available object shapes: box, cylinder, sphere

Output format: JSON array where each element has:
- "primitive": one of "move_to", "pick", "place"
- "target": object color/name or position description
- "description": brief English description of the step

Rules:
1. Always approach an object (move_to) before picking it
2. Always pick an object before placing it
3. One object manipulation at a time
4. Output ONLY the JSON array, no other text
"""


def build_decomposition_prompt(instruction: str) -> str:
    """Build the full prompt for task decomposition.

    Args:
        instruction: e.g. "pick up the red block and put it next to the blue one"

    Returns:
        Formatted prompt string with system context + instruction.
    """
    return f"{DECOMPOSITION_SYSTEM}\n\nInstruction: {instruction}\n\nSub-tasks:"


# ── Scenario library for evaluation ──────────────────────────────

EVAL_SCENARIOS: list[dict] = [
    # Simple single-object tasks
    {
        "instruction": "pick up the red block",
        "expected_primitives": ["move_to", "pick"],
        "expected_targets": ["red", "red"],
    },
    {
        "instruction": "move to the blue cylinder",
        "expected_primitives": ["move_to"],
        "expected_targets": ["blue"],
    },
    {
        "instruction": "grab the green sphere",
        "expected_primitives": ["move_to", "pick"],
        "expected_targets": ["green", "green"],
    },
    # Pick-and-place tasks
    {
        "instruction": "pick up the red block and place it on the blue block",
        "expected_primitives": ["move_to", "pick", "place"],
        "expected_targets": ["red", "red", "blue"],
    },
    {
        "instruction": "put the yellow cylinder next to the green one",
        "expected_primitives": ["move_to", "pick", "place"],
        "expected_targets": ["yellow", "yellow", "green"],
    },
    {
        "instruction": "take the orange sphere and drop it by the purple box",
        "expected_primitives": ["move_to", "pick", "place"],
        "expected_targets": ["orange", "orange", "purple"],
    },
    # Move-only tasks
    {
        "instruction": "approach the red object",
        "expected_primitives": ["move_to"],
        "expected_targets": ["red"],
    },
    {
        "instruction": "go to the blue block",
        "expected_primitives": ["move_to"],
        "expected_targets": ["blue"],
    },
    {
        "instruction": "reach for the green cylinder",
        "expected_primitives": ["move_to"],
        "expected_targets": ["green"],
    },
    # Stacking tasks
    {
        "instruction": "stack the red block on the blue block",
        "expected_primitives": None,  # Stack: 2 colors × 3 steps
        "expected_targets": None,
    },
    {
        "instruction": "stack red green and blue blocks into a tower",
        "expected_primitives": None,  # Variable: 3 colors × 3 steps each
        "expected_targets": None,
    },
    # Sorting tasks
    {
        "instruction": "sort the red and blue objects into separate bins",
        "expected_primitives": None,  # Variable: depends on detected colors
        "expected_targets": None,
    },
    # Ambiguous / challenging
    {
        "instruction": "clean up the table",
        "expected_primitives": None,  # Any valid decomposition
        "expected_targets": None,
    },
    {
        "instruction": "pick up everything",
        "expected_primitives": None,
        "expected_targets": None,
    },
    {
        "instruction": "move the red one to where the blue one is",
        "expected_primitives": ["move_to"],
        "expected_targets": ["red"],
    },
    # Multi-step complex instructions
    {
        "instruction": "pick up the red block, then pick up the blue block, and stack them",
        "expected_primitives": None,  # At least 6 steps
        "expected_targets": None,
    },
    {
        "instruction": "grab the yellow sphere and put it on the green cylinder",
        "expected_primitives": ["move_to", "pick", "place"],
        "expected_targets": ["yellow", "yellow", "green"],
    },
    {
        "instruction": "take the purple box and place it on the table center",
        "expected_primitives": ["move_to", "pick", "place"],
        "expected_targets": ["purple", "purple", None],
    },
    {
        "instruction": "move the robot arm to the orange object",
        "expected_primitives": ["move_to"],
        "expected_targets": ["orange"],
    },
    {
        "instruction": "get the red cylinder and bring it to the goal",
        "expected_primitives": ["move_to", "pick", "place"],
        "expected_targets": ["red", "red", None],
    },
]
