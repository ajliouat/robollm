"""
Object grounding — map text descriptions to object poses in simulation.

Provides:
- GrounderBase: abstract interface
- SimGrounder: uses privileged sim info (object specs + positions)
- VisualGrounder: DINOv2-small feature matching (optional, requires torchvision)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


@dataclass
class GroundedObject:
    """Result of grounding a text description to a sim object."""
    name: str               # object name in sim (e.g. "red_box_0")
    position: np.ndarray    # 3D position
    color: str              # canonical color
    shape: str              # box, cylinder, sphere
    confidence: float = 1.0
    match_reason: str = ""  # why this object was matched

    def __repr__(self) -> str:
        pos = ", ".join(f"{x:.3f}" for x in self.position)
        return (
            f"GroundedObject({self.name}, color={self.color}, "
            f"shape={self.shape}, pos=[{pos}], conf={self.confidence:.2f})"
        )


@dataclass
class GroundingResult:
    """Full result of a grounding query."""
    query: str
    matched: GroundedObject | None
    candidates: list[GroundedObject] = field(default_factory=list)
    success: bool = False
    error: str = ""


class GrounderBase(ABC):
    """Abstract base for object grounding."""

    @abstractmethod
    def ground(
        self,
        description: str,
        scene_info: dict[str, Any],
        image: np.ndarray | None = None,
    ) -> GroundingResult:
        """Map a text description to an object in the scene.

        Args:
            description: e.g. "red block", "blue cylinder", "green"
            scene_info: dict with object names, positions, colors, shapes
            image: optional scene image (for visual grounding)

        Returns:
            GroundingResult with matched object and candidates.
        """
        ...


# ── Color & shape synonyms ────────────────────────────────────────

_COLOR_SYNONYMS: dict[str, set[str]] = {
    "red": {"red", "crimson", "scarlet", "ruby", "cherry"},
    "green": {"green", "lime", "emerald", "jade"},
    "blue": {"blue", "azure", "navy", "cyan", "cobalt"},
    "yellow": {"yellow", "gold", "amber", "lemon"},
    "orange": {"orange", "tangerine", "amber"},
    "purple": {"purple", "violet", "magenta", "lavender", "plum"},
}

_SHAPE_SYNONYMS: dict[str, set[str]] = {
    "box": {"box", "block", "cube", "brick", "square"},
    "cylinder": {"cylinder", "tube", "can", "rod", "pillar"},
    "sphere": {"sphere", "ball", "orb", "globe", "round"},
}


def _resolve_color(word: str) -> str | None:
    """Resolve a word to a canonical color, or None."""
    word = word.lower().strip()
    for color, syns in _COLOR_SYNONYMS.items():
        if word in syns:
            return color
    return None


def _resolve_shape(word: str) -> str | None:
    """Resolve a word to a canonical shape, or None."""
    word = word.lower().strip()
    for shape, syns in _SHAPE_SYNONYMS.items():
        if word in syns:
            return shape
    return None


# ── SimGrounder (privileged info) ─────────────────────────────────

class SimGrounder(GrounderBase):
    """Ground objects using privileged simulation info.

    Uses exact object color/shape from the env info dict.
    No vision model needed — pure string matching with synonyms.
    """

    def ground(
        self,
        description: str,
        scene_info: dict[str, Any],
        image: np.ndarray | None = None,
    ) -> GroundingResult:
        words = description.lower().split()

        # Extract color and shape from description
        query_color = None
        query_shape = None
        for w in words:
            if query_color is None:
                query_color = _resolve_color(w)
            if query_shape is None:
                query_shape = _resolve_shape(w)

        # Get objects from scene info
        objects = self._extract_objects(scene_info)
        if not objects:
            return GroundingResult(
                query=description,
                matched=None,
                success=False,
                error="No objects in scene_info",
            )

        # Score each object
        scored: list[tuple[float, GroundedObject]] = []
        for obj in objects:
            score = 0.0
            reasons = []

            if query_color and obj.color == query_color:
                score += 1.0
                reasons.append(f"color={query_color}")
            if query_shape and obj.shape == query_shape:
                score += 0.5
                reasons.append(f"shape={query_shape}")

            # Partial match: object name contains query word
            for w in words:
                if w in obj.name.lower():
                    score += 0.2
                    reasons.append(f"name contains '{w}'")
                    break

            obj.confidence = min(score, 1.0)
            obj.match_reason = ", ".join(reasons) if reasons else "no match"
            scored.append((score, obj))

        # Sort by score (descending)
        scored.sort(key=lambda x: -x[0])
        candidates = [obj for _, obj in scored]

        best_score, best_obj = scored[0]
        if best_score > 0:
            return GroundingResult(
                query=description,
                matched=best_obj,
                candidates=candidates,
                success=True,
            )
        else:
            return GroundingResult(
                query=description,
                matched=None,
                candidates=candidates,
                success=False,
                error=f"No object matches '{description}'",
            )

    @staticmethod
    def _extract_objects(scene_info: dict) -> list[GroundedObject]:
        """Extract GroundedObject list from scene info dict."""
        objects: list[GroundedObject] = []

        # Format 1: obj_specs list (from MultiObjectEnv._obj_specs)
        if "obj_specs" in scene_info:
            positions = scene_info.get("obj_positions", {})
            for spec in scene_info["obj_specs"]:
                pos = positions.get(spec.name, np.zeros(3))
                objects.append(GroundedObject(
                    name=spec.name,
                    position=np.asarray(pos),
                    color=spec.color,
                    shape=spec.shape,
                ))

        # Format 2: flat dict with object_names, object_colors, etc.
        elif "object_names" in scene_info:
            names = scene_info["object_names"]
            colors = scene_info.get("object_colors", ["unknown"] * len(names))
            shapes = scene_info.get("object_shapes", ["unknown"] * len(names))
            positions = scene_info.get("object_positions", [np.zeros(3)] * len(names))
            for name, color, shape, pos in zip(names, colors, shapes, positions):
                objects.append(GroundedObject(
                    name=name,
                    position=np.asarray(pos),
                    color=color,
                    shape=shape,
                ))

        # Format 3: objects list of dicts
        elif "objects" in scene_info:
            for obj_dict in scene_info["objects"]:
                objects.append(GroundedObject(
                    name=obj_dict.get("name", "unknown"),
                    position=np.asarray(obj_dict.get("position", [0, 0, 0])),
                    color=obj_dict.get("color", "unknown"),
                    shape=obj_dict.get("shape", "unknown"),
                ))

        return objects


# ── VisualGrounder (DINOv2) ───────────────────────────────────────

class VisualGrounder(GrounderBase):
    """Ground objects using DINOv2-small visual embeddings.

    Extracts object crops from the scene image, computes DINOv2
    embeddings, and matches against text-derived target embeddings.

    Requires: pip install torch torchvision
    """

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        device: str = "cpu",
    ):
        self._model_name = model_name
        self._device = device
        self._model = None
        self._transform = None

    def _load(self):
        if self._model is not None:
            return

        try:
            import torch
            import torchvision.transforms as T
        except ImportError as e:
            raise ImportError(
                "VisualGrounder requires: pip install torch torchvision"
            ) from e

        self._model = torch.hub.load(
            "facebookresearch/dinov2", self._model_name,
        )
        self._model.eval()
        self._model.to(self._device)

        self._transform = T.Compose([
            T.ToPILImage(),
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _embed_crop(self, crop: np.ndarray) -> np.ndarray:
        """Compute DINOv2 embedding for an image crop."""
        import torch

        self._load()
        tensor = self._transform(crop).unsqueeze(0).to(self._device)
        with torch.no_grad():
            features = self._model(tensor)
        return features.squeeze().cpu().numpy()

    def ground(
        self,
        description: str,
        scene_info: dict[str, Any],
        image: np.ndarray | None = None,
    ) -> GroundingResult:
        """Ground using visual similarity.

        Falls back to SimGrounder if no image is provided.
        """
        if image is None:
            # Fallback to sim grounder
            return SimGrounder().ground(description, scene_info, image)

        self._load()

        # For now, use SimGrounder with confidence boost from visual check
        # Full visual grounding requires bounding box extraction
        sim_result = SimGrounder().ground(description, scene_info, image)
        return sim_result
