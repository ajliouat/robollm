"""
Dynamic multi-object spawning for MuJoCo tabletop scenes.

Generates randomized objects (cubes, cylinders, spheres) in 6 colors,
injects them into the MJCF XML at load time, and provides utilities
for querying object metadata during simulation.
"""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

# ── Object catalogue ─────────────────────────────────────────────────────────

SHAPES = ("box", "cylinder", "sphere")

COLORS: dict[str, tuple[float, float, float]] = {
    "red":    (0.85, 0.15, 0.15),
    "green":  (0.15, 0.75, 0.15),
    "blue":   (0.15, 0.25, 0.85),
    "yellow": (0.90, 0.85, 0.10),
    "orange": (0.95, 0.55, 0.10),
    "purple": (0.60, 0.15, 0.75),
}

COLOR_NAMES = list(COLORS.keys())

# ── Size parameters ──────────────────────────────────────────────────────────

#: Half-extent for cubes, radius for cylinders/spheres
_OBJ_SIZE = 0.02  # 4 cm objects (2 cm half-extent)
_CYL_HALF_HEIGHT = 0.02

# ── Spawn region (table surface) ─────────────────────────────────────────────

_TABLE_X_RANGE = (0.30, 0.70)
_TABLE_Y_RANGE = (-0.10, 0.25)
_TABLE_Z = 0.415  # table top surface height

_MIN_OBJ_SPACING = 0.06  # minimum distance between object centres (m)


@dataclass
class ObjectSpec:
    """Metadata for a single spawned object."""

    name: str
    shape: str  # "box" | "cylinder" | "sphere"
    color_name: str
    rgba: tuple[float, float, float, float]
    size_attr: str  # MuJoCo geom `size` attribute value
    mass: float = 0.05

    # Set at spawn time
    init_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def qpos_len(self) -> int:
        """Length of free-joint qpos (3 pos + 4 quat = 7)."""
        return 7

    @property
    def qvel_len(self) -> int:
        """Length of free-joint qvel (3 lin + 3 ang = 6)."""
        return 6


def _make_geom_size(shape: str) -> str:
    """Return the MuJoCo `size` string for a given shape type."""
    if shape == "box":
        return f"{_OBJ_SIZE} {_OBJ_SIZE} {_OBJ_SIZE}"
    elif shape == "cylinder":
        return f"{_OBJ_SIZE} {_CYL_HALF_HEIGHT}"
    elif shape == "sphere":
        return f"{_OBJ_SIZE}"
    raise ValueError(f"Unknown shape: {shape}")


def _surface_z(shape: str) -> float:
    """Z position so the object rests on the table surface."""
    if shape == "box":
        return _TABLE_Z + _OBJ_SIZE
    elif shape == "cylinder":
        return _TABLE_Z + _CYL_HALF_HEIGHT
    elif shape == "sphere":
        return _TABLE_Z + _OBJ_SIZE
    raise ValueError(f"Unknown shape: {shape}")


# ── Spawning logic ───────────────────────────────────────────────────────────


def generate_object_specs(
    n_objects: int,
    rng: np.random.Generator,
    *,
    allowed_shapes: Sequence[str] = SHAPES,
    allowed_colors: Sequence[str] | None = None,
    unique_colors: bool = True,
) -> list[ObjectSpec]:
    """Create specs for *n_objects* with randomized shape/color/position.

    Parameters
    ----------
    n_objects : int
        Number of objects to generate (1–6).
    rng : numpy Generator
        Random number generator for reproducibility.
    allowed_shapes : sequence of str
        Subset of SHAPES to sample from.
    allowed_colors : sequence of str or None
        Subset of COLOR_NAMES. None → all 6 colours.
    unique_colors : bool
        If True, no two objects share a colour (requires n <= len(colors)).
    """
    if allowed_colors is None:
        allowed_colors = list(COLOR_NAMES)
    if unique_colors and n_objects > len(allowed_colors):
        raise ValueError(
            f"Cannot pick {n_objects} unique colors from {len(allowed_colors)} options"
        )

    # Pick colours
    if unique_colors:
        color_choices = list(
            rng.choice(allowed_colors, size=n_objects, replace=False)
        )
    else:
        color_choices = list(
            rng.choice(allowed_colors, size=n_objects, replace=True)
        )

    # Pick shapes
    shape_choices = list(
        rng.choice(list(allowed_shapes), size=n_objects, replace=True)
    )

    # Generate non-overlapping positions
    positions = _sample_non_overlapping_positions(n_objects, rng)

    specs: list[ObjectSpec] = []
    for i, (shape, color_name) in enumerate(zip(shape_choices, color_choices)):
        r, g, b = COLORS[color_name]
        name = f"obj_{color_name}_{shape}"
        # Ensure unique names if same color+shape appears (non-unique case)
        existing_names = {s.name for s in specs}
        if name in existing_names:
            name = f"{name}_{i}"

        pos = positions[i].copy()
        pos[2] = _surface_z(shape)

        spec = ObjectSpec(
            name=name,
            shape=shape,
            color_name=color_name,
            rgba=(r, g, b, 1.0),
            size_attr=_make_geom_size(shape),
            init_pos=pos,
        )
        specs.append(spec)

    return specs


def _sample_non_overlapping_positions(
    n: int,
    rng: np.random.Generator,
    max_attempts: int = 200,
) -> list[np.ndarray]:
    """Sample *n* positions on the table that are at least _MIN_OBJ_SPACING apart."""
    positions: list[np.ndarray] = []
    for _ in range(n):
        for _attempt in range(max_attempts):
            x = rng.uniform(*_TABLE_X_RANGE)
            y = rng.uniform(*_TABLE_Y_RANGE)
            pos = np.array([x, y, 0.0])
            if all(
                np.linalg.norm(pos[:2] - p[:2]) >= _MIN_OBJ_SPACING
                for p in positions
            ):
                positions.append(pos)
                break
        else:
            # Fallback: place anyway (won't overlap much at 6 objects)
            x = rng.uniform(*_TABLE_X_RANGE)
            y = rng.uniform(*_TABLE_Y_RANGE)
            positions.append(np.array([x, y, 0.0]))
    return positions


# ── MJCF injection ───────────────────────────────────────────────────────────


def inject_objects_into_xml(
    xml_path: str | Path,
    specs: list[ObjectSpec],
    *,
    remove_existing_block: bool = True,
) -> str:
    """Read the base scene XML and inject object bodies.

    Returns the modified XML as a string (does NOT write to disk).
    The original file is never modified.
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    worldbody = root.find("worldbody")
    assert worldbody is not None, "No <worldbody> in MJCF"

    # Optionally remove the hardcoded red block from v1.0.0 scene
    if remove_existing_block:
        for body in list(worldbody):
            if body.get("name") == "block_red":
                worldbody.remove(body)

    # Also update the keyframe to remove the old block qpos if we removed it
    keyframe_elem = root.find(".//keyframe/key[@name='home']")

    # Inject new object bodies
    for spec in specs:
        body = ET.SubElement(worldbody, "body")
        body.set("name", spec.name)
        px, py, pz = spec.init_pos
        body.set("pos", f"{px:.4f} {py:.4f} {pz:.4f}")

        joint = ET.SubElement(body, "joint")
        joint.set("name", f"{spec.name}_free")
        joint.set("type", "free")

        geom = ET.SubElement(body, "geom")
        geom.set("name", f"{spec.name}_geom")
        geom.set("type", spec.shape)
        geom.set("size", spec.size_attr)
        r, g, b, a = spec.rgba
        geom.set("rgba", f"{r:.2f} {g:.2f} {b:.2f} {a:.1f}")
        geom.set("mass", str(spec.mass))
        geom.set("class", "object")

    # Rebuild keyframe qpos: arm(7) + fingers(2) + N objects * 7
    if keyframe_elem is not None:
        arm_fingers = "0 -0.785 0 -2.356 0 1.571 0.785 0.02 0.02"
        obj_qpos_parts: list[str] = []
        for spec in specs:
            px, py, pz = spec.init_pos
            obj_qpos_parts.append(
                f"{px:.4f} {py:.4f} {pz:.4f} 1 0 0 0"
            )
        full_qpos = arm_fingers + " " + " ".join(obj_qpos_parts)
        keyframe_elem.set("qpos", full_qpos)

    return ET.tostring(root, encoding="unicode")


# ── Utilities for environments ───────────────────────────────────────────────


def get_object_qpos_slice(
    obj_index: int,
    n_arm_dof: int = 7,
    n_finger_dof: int = 2,
) -> slice:
    """Return the qpos slice for object *obj_index* (0-based)."""
    base = n_arm_dof + n_finger_dof + obj_index * 7
    return slice(base, base + 7)


def get_object_qvel_slice(
    obj_index: int,
    n_arm_dof: int = 7,
    n_finger_dof: int = 2,
) -> slice:
    """Return the qvel slice for object *obj_index* (0-based)."""
    base = n_arm_dof + n_finger_dof + obj_index * 6
    return slice(base, base + 6)
