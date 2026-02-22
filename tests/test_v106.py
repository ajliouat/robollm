"""Tests for v1.0.6 — Object grounding."""

from __future__ import annotations

import numpy as np
import pytest

from planner.grounder import (
    GroundedObject,
    GroundingResult,
    SimGrounder,
    _resolve_color,
    _resolve_shape,
)


# ── Synonym resolution ────────────────────────────────────────────

class TestSynonyms:
    def test_resolve_color_canonical(self):
        assert _resolve_color("red") == "red"
        assert _resolve_color("blue") == "blue"
        assert _resolve_color("green") == "green"

    def test_resolve_color_synonyms(self):
        assert _resolve_color("crimson") == "red"
        assert _resolve_color("azure") == "blue"
        assert _resolve_color("lime") == "green"
        assert _resolve_color("gold") == "yellow"
        assert _resolve_color("tangerine") == "orange"
        assert _resolve_color("violet") == "purple"

    def test_resolve_color_case_insensitive(self):
        assert _resolve_color("RED") == "red"
        assert _resolve_color("Blue") == "blue"

    def test_resolve_color_unknown(self):
        assert _resolve_color("pink") is None
        assert _resolve_color("xyz") is None

    def test_resolve_shape_canonical(self):
        assert _resolve_shape("box") == "box"
        assert _resolve_shape("cylinder") == "cylinder"
        assert _resolve_shape("sphere") == "sphere"

    def test_resolve_shape_synonyms(self):
        assert _resolve_shape("block") == "box"
        assert _resolve_shape("cube") == "box"
        assert _resolve_shape("ball") == "sphere"
        assert _resolve_shape("tube") == "cylinder"
        assert _resolve_shape("orb") == "sphere"

    def test_resolve_shape_unknown(self):
        assert _resolve_shape("pyramid") is None


# ── SimGrounder ───────────────────────────────────────────────────

def _make_scene_info(objects: list[dict]) -> dict:
    """Helper to create scene_info dict in 'objects' format."""
    return {"objects": objects}


def _standard_scene() -> dict:
    """Standard 3-object scene for testing."""
    return _make_scene_info([
        {
            "name": "red_box_0",
            "color": "red",
            "shape": "box",
            "position": [0.5, 0.0, 0.435],
        },
        {
            "name": "blue_cylinder_1",
            "color": "blue",
            "shape": "cylinder",
            "position": [0.4, 0.1, 0.435],
        },
        {
            "name": "green_sphere_2",
            "color": "green",
            "shape": "sphere",
            "position": [0.6, -0.1, 0.435],
        },
    ])


class TestSimGrounder:
    def setup_method(self):
        self.grounder = SimGrounder()
        self.scene = _standard_scene()

    def test_ground_by_color(self):
        result = self.grounder.ground("red", self.scene)
        assert result.success
        assert result.matched is not None
        assert result.matched.color == "red"

    def test_ground_by_color_blue(self):
        result = self.grounder.ground("blue", self.scene)
        assert result.success
        assert result.matched.color == "blue"

    def test_ground_by_color_green(self):
        result = self.grounder.ground("green", self.scene)
        assert result.success
        assert result.matched.color == "green"

    def test_ground_by_shape(self):
        result = self.grounder.ground("sphere", self.scene)
        assert result.success
        assert result.matched.shape == "sphere"

    def test_ground_by_color_and_shape(self):
        result = self.grounder.ground("red box", self.scene)
        assert result.success
        assert result.matched.color == "red"
        assert result.matched.shape == "box"

    def test_ground_by_synonym(self):
        result = self.grounder.ground("crimson cube", self.scene)
        assert result.success
        assert result.matched.color == "red"
        assert result.matched.shape == "box"

    def test_ground_ball(self):
        result = self.grounder.ground("green ball", self.scene)
        assert result.success
        assert result.matched.color == "green"
        assert result.matched.shape == "sphere"

    def test_ground_tube(self):
        result = self.grounder.ground("blue tube", self.scene)
        assert result.success
        assert result.matched.color == "blue"
        assert result.matched.shape == "cylinder"

    def test_ground_no_match(self):
        result = self.grounder.ground("yellow pyramid", self.scene)
        assert not result.success

    def test_ground_empty_scene(self):
        result = self.grounder.ground("red", _make_scene_info([]))
        assert not result.success
        assert "No objects" in result.error

    def test_candidates_sorted_by_score(self):
        result = self.grounder.ground("red box", self.scene)
        assert len(result.candidates) == 3
        # Best match should be red box
        assert result.candidates[0].color == "red"

    def test_position_preserved(self):
        result = self.grounder.ground("red box", self.scene)
        np.testing.assert_array_almost_equal(
            result.matched.position, [0.5, 0.0, 0.435])

    def test_grounded_object_repr(self):
        obj = GroundedObject(
            name="red_box_0",
            position=np.array([0.5, 0.0, 0.435]),
            color="red", shape="box",
        )
        repr_str = repr(obj)
        assert "red_box_0" in repr_str
        assert "red" in repr_str

    def test_result_fields(self):
        result = self.grounder.ground("red", self.scene)
        assert result.query == "red"
        assert isinstance(result.matched, GroundedObject)
        assert isinstance(result.candidates, list)


# ── Scene info formats ────────────────────────────────────────────

class TestSceneInfoFormats:
    def setup_method(self):
        self.grounder = SimGrounder()

    def test_object_names_format(self):
        scene = {
            "object_names": ["red_box_0", "blue_cyl_1"],
            "object_colors": ["red", "blue"],
            "object_shapes": ["box", "cylinder"],
            "object_positions": [
                np.array([0.5, 0.0, 0.435]),
                np.array([0.4, 0.1, 0.435]),
            ],
        }
        result = self.grounder.ground("blue", scene)
        assert result.success
        assert result.matched.color == "blue"

    def test_no_objects_format(self):
        """Scene with no recognizable format returns error."""
        result = self.grounder.ground("red", {"random_key": 123})
        assert not result.success


# ── Grounding accuracy on standard queries ────────────────────────

_GROUNDING_QUERIES = [
    ("red block", "red"),
    ("the blue cylinder", "blue"),
    ("green sphere", "green"),
    ("red cube", "red"),
    ("blue tube", "blue"),
    ("green ball", "green"),
    ("crimson box", "red"),
    ("azure cylinder", "blue"),
    ("emerald sphere", "green"),
    ("the block", "box"),  # shape-only query
    ("red", "red"),
    ("blue", "blue"),
    ("green", "green"),
    ("red box on the table", "red"),
    ("big blue cylinder", "blue"),
    ("small green ball", "green"),
    ("ruby brick", "red"),
    ("cobalt rod", "blue"),
    ("jade orb", "green"),
    ("scarlet cube", "red"),
]


class TestGroundingAccuracy:
    def setup_method(self):
        self.grounder = SimGrounder()
        self.scene = _standard_scene()

    @pytest.mark.parametrize("query,expected", _GROUNDING_QUERIES,
                             ids=[q[0] for q in _GROUNDING_QUERIES])
    def test_query(self, query: str, expected: str):
        result = self.grounder.ground(query, self.scene)
        assert result.success, f"Failed to ground '{query}'"
        # Expected is either a color or a shape
        if expected in {"red", "green", "blue", "yellow", "orange", "purple"}:
            assert result.matched.color == expected, (
                f"Query '{query}': expected color={expected}, "
                f"got {result.matched.color}"
            )
        else:
            assert result.matched.shape == expected

    def test_overall_accuracy(self):
        total = len(_GROUNDING_QUERIES)
        correct = 0
        for query, expected in _GROUNDING_QUERIES:
            result = self.grounder.ground(query, self.scene)
            if result.success:
                if expected in {"red", "green", "blue"}:
                    if result.matched.color == expected:
                        correct += 1
                elif result.matched.shape == expected:
                    correct += 1
        accuracy = correct / total
        print(f"\nGrounding accuracy: {correct}/{total} = {accuracy:.0%}")
        assert accuracy >= 0.80, f"Accuracy {accuracy:.0%} < 80%"
