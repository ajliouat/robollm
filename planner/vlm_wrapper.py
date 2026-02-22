"""
VLM wrapper — abstract interface for Vision-Language Models.

Provides:
- VLMBase: abstract base class
- MockVLM: deterministic rule-based planner for testing
- TransformersVLM: real VLM via HuggingFace transformers (optional)
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class VLMResponse:
    """Structured response from a VLM query."""
    raw_text: str
    sub_tasks: list[dict[str, Any]]
    model_name: str
    confidence: float = 1.0


class VLMBase(ABC):
    """Abstract base class for Vision-Language Model wrappers."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier."""
        ...

    @abstractmethod
    def generate(self, image: Any, instruction: str) -> str:
        """Generate raw text response from image + instruction.

        Args:
            image: RGB image (numpy array H×W×3) or None for text-only.
            instruction: Natural language task instruction.

        Returns:
            Raw text response from the model.
        """
        ...

    def decompose(self, image: Any, instruction: str) -> VLMResponse:
        """Decompose instruction into sub-task sequence.

        Calls generate() then parses the output.

        Args:
            image: RGB image or None.
            instruction: e.g. "pick up the red block and place it on the blue one"

        Returns:
            VLMResponse with parsed sub-task list.
        """
        from planner.prompt_templates import build_decomposition_prompt

        prompt = build_decomposition_prompt(instruction)
        raw = self.generate(image, prompt)
        sub_tasks = self._parse_json_subtasks(raw)
        return VLMResponse(
            raw_text=raw,
            sub_tasks=sub_tasks,
            model_name=self.model_name,
        )

    @staticmethod
    def _parse_json_subtasks(text: str) -> list[dict[str, Any]]:
        """Extract JSON array of sub-tasks from model output."""
        # Try direct JSON parse first
        text = text.strip()
        if text.startswith("["):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

        # Search for JSON array in text
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Fallback: try to extract from markdown code block
        match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        return []


# ── Mock VLM ──────────────────────────────────────────────────────

# Keywords that map to primitives
_PICK_WORDS = {"pick", "grab", "grasp", "take", "get"}
_PLACE_WORDS = {"place", "put", "set", "drop", "release", "bring"}
_MOVE_WORDS = {"move", "go", "approach", "reach", "touch"}
_STACK_WORDS = {"stack", "pile", "tower"}
_SORT_WORDS = {"sort", "arrange", "organize", "group"}

_COLORS = {"red", "green", "blue", "yellow", "orange", "purple"}


class MockVLM(VLMBase):
    """Rule-based mock VLM for testing without GPU.

    Parses keywords from instructions to produce deterministic
    sub-task decompositions. Useful for pipeline integration tests.
    """

    @property
    def model_name(self) -> str:
        return "mock-vlm-v1"

    def generate(self, image: Any, instruction: str) -> str:
        """Generate deterministic response by keyword matching."""
        lower = instruction.lower()
        sub_tasks = self._keyword_decompose(lower)
        return json.dumps(sub_tasks, indent=2)

    def decompose(self, image: Any, instruction: str) -> VLMResponse:
        """Override decompose to use raw instruction (skip prompt wrapping).

        The base class wraps the instruction in a system prompt before
        calling generate(). Since MockVLM uses keyword matching, the
        system prompt words (pick, place, move_to) would always match.
        """
        sub_tasks = self._keyword_decompose(instruction.lower())
        raw = json.dumps(sub_tasks, indent=2)
        return VLMResponse(
            raw_text=raw,
            sub_tasks=sub_tasks,
            model_name=self.model_name,
        )

    def _keyword_decompose(self, text: str) -> list[dict[str, Any]]:
        """Decompose instruction text into sub-task list via keywords."""
        words = set(text.split())
        sub_tasks: list[dict[str, Any]] = []

        # Detect target colors
        colors = [c for c in _COLORS if c in words]
        target_color = colors[0] if colors else "red"
        dest_color = colors[1] if len(colors) > 1 else None

        # Detect task type
        has_pick = bool(words & _PICK_WORDS)
        has_place = bool(words & _PLACE_WORDS)
        has_move = bool(words & _MOVE_WORDS)
        has_stack = bool(words & _STACK_WORDS)
        has_sort = bool(words & _SORT_WORDS)

        if has_stack:
            # Stack task: move_to → pick → place for each object
            for i, c in enumerate(colors if colors else ["red", "blue"]):
                sub_tasks.append({
                    "primitive": "move_to",
                    "target": c,
                    "description": f"approach the {c} object",
                })
                sub_tasks.append({
                    "primitive": "pick",
                    "target": c,
                    "description": f"pick up the {c} object",
                })
                sub_tasks.append({
                    "primitive": "place",
                    "target": "stack_position",
                    "description": f"place {c} object at stack position {i}",
                })
        elif has_sort:
            for c in colors if colors else ["red", "blue", "green"]:
                sub_tasks.append({
                    "primitive": "move_to",
                    "target": c,
                    "description": f"approach the {c} object",
                })
                sub_tasks.append({
                    "primitive": "pick",
                    "target": c,
                    "description": f"pick up the {c} object",
                })
                sub_tasks.append({
                    "primitive": "place",
                    "target": f"{c}_bin",
                    "description": f"place {c} object in {c} bin",
                })
        elif has_pick and has_place:
            sub_tasks.append({
                "primitive": "move_to",
                "target": target_color,
                "description": f"approach the {target_color} object",
            })
            sub_tasks.append({
                "primitive": "pick",
                "target": target_color,
                "description": f"pick up the {target_color} object",
            })
            dest = dest_color if dest_color else "goal"
            sub_tasks.append({
                "primitive": "place",
                "target": dest,
                "description": f"place object at {dest} position",
            })
        elif has_place:
            # 'put'/'set'/'drop' implies pick then place
            sub_tasks.append({
                "primitive": "move_to",
                "target": target_color,
                "description": f"approach the {target_color} object",
            })
            sub_tasks.append({
                "primitive": "pick",
                "target": target_color,
                "description": f"pick up the {target_color} object",
            })
            dest = dest_color if dest_color else "goal"
            sub_tasks.append({
                "primitive": "place",
                "target": dest,
                "description": f"place object at {dest} position",
            })
        elif has_pick:
            sub_tasks.append({
                "primitive": "move_to",
                "target": target_color,
                "description": f"approach the {target_color} object",
            })
            sub_tasks.append({
                "primitive": "pick",
                "target": target_color,
                "description": f"pick up the {target_color} object",
            })
        elif has_move:
            sub_tasks.append({
                "primitive": "move_to",
                "target": target_color,
                "description": f"move to the {target_color} object",
            })
        else:
            # Default: try pick and place
            sub_tasks.append({
                "primitive": "move_to",
                "target": target_color,
                "description": f"approach the {target_color} object",
            })
            sub_tasks.append({
                "primitive": "pick",
                "target": target_color,
                "description": f"pick up the {target_color} object",
            })

        return sub_tasks


# ── Transformers VLM (optional) ───────────────────────────────────

class TransformersVLM(VLMBase):
    """HuggingFace Transformers VLM wrapper.

    Supports PaliGemma, Phi-3-Vision, or any vision-language model
    with a text generation pipeline.

    Requires: pip install transformers torch accelerate
    """

    def __init__(
        self,
        model_id: str = "google/paligemma-3b-mix-224",
        device: str = "auto",
        max_new_tokens: int = 512,
        quantize_4bit: bool = False,
    ):
        self._model_id = model_id
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._quantize_4bit = quantize_4bit
        self._pipeline = None
        self._processor = None

    @property
    def model_name(self) -> str:
        return self._model_id.split("/")[-1]

    def _load(self):
        """Lazy-load model on first use."""
        if self._pipeline is not None:
            return

        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
            import torch
        except ImportError as e:
            raise ImportError(
                "TransformersVLM requires: "
                "pip install transformers torch accelerate"
            ) from e

        kwargs: dict[str, Any] = {}
        if self._quantize_4bit:
            try:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            except ImportError:
                pass

        self._processor = AutoProcessor.from_pretrained(self._model_id)
        self._pipeline = AutoModelForVision2Seq.from_pretrained(
            self._model_id,
            device_map=self._device,
            torch_dtype=torch.float16,
            **kwargs,
        )

    def generate(self, image: Any, instruction: str) -> str:
        self._load()

        from PIL import Image as PILImage
        import numpy as np

        # Convert numpy array to PIL if needed
        if isinstance(image, np.ndarray):
            pil_image = PILImage.fromarray(image.astype(np.uint8))
        elif image is None:
            # Create dummy image for text-only queries
            pil_image = PILImage.new("RGB", (224, 224), (128, 128, 128))
        else:
            pil_image = image

        inputs = self._processor(
            text=instruction,
            images=pil_image,
            return_tensors="pt",
        ).to(self._pipeline.device)

        import torch
        with torch.no_grad():
            outputs = self._pipeline.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
            )

        decoded = self._processor.decode(
            outputs[0], skip_special_tokens=True,
        )
        # Remove the input prompt from output
        if instruction in decoded:
            decoded = decoded[decoded.index(instruction) + len(instruction):]
        return decoded.strip()
