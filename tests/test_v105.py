"""Tests for v1.0.5 — VLM planner, task parser, and decomposition accuracy."""

from __future__ import annotations

import json

import pytest

from planner.vlm_wrapper import MockVLM, VLMBase, VLMResponse
from planner.task_parser import (
    SubTask, TaskPlan, parse_subtasks, normalise_color,
)
from planner.prompt_templates import (
    EVAL_SCENARIOS, VALID_PRIMITIVES, build_decomposition_prompt,
)
from planner.planner import Planner


# ── VLMBase & MockVLM ──────────────────────────────────────────────

class TestMockVLM:
    def setup_method(self):
        self.vlm = MockVLM()

    def test_model_name(self):
        assert self.vlm.model_name == "mock-vlm-v1"

    def test_generate_returns_json_string(self):
        result = self.vlm.generate(None, "pick up the red block")
        parsed = json.loads(result)
        assert isinstance(parsed, list)

    def test_pick_instruction(self):
        result = self.vlm.generate(None, "pick up the red block")
        tasks = json.loads(result)
        primitives = [t["primitive"] for t in tasks]
        assert "move_to" in primitives
        assert "pick" in primitives

    def test_place_instruction(self):
        result = self.vlm.generate(
            None, "pick up the red block and place it on the blue one")
        tasks = json.loads(result)
        primitives = [t["primitive"] for t in tasks]
        assert "place" in primitives

    def test_move_instruction(self):
        result = self.vlm.generate(None, "move to the green sphere")
        tasks = json.loads(result)
        assert tasks[0]["primitive"] == "move_to"
        assert tasks[0]["target"] == "green"

    def test_stack_instruction(self):
        result = self.vlm.generate(
            None, "stack the red and blue blocks")
        tasks = json.loads(result)
        primitives = [t["primitive"] for t in tasks]
        assert primitives.count("pick") >= 2

    def test_sort_instruction(self):
        result = self.vlm.generate(
            None, "sort the red and blue objects into bins")
        tasks = json.loads(result)
        assert len(tasks) >= 4  # 2 objects × (move_to + pick + place)

    def test_color_detection(self):
        result = self.vlm.generate(None, "grab the yellow cylinder")
        tasks = json.loads(result)
        targets = [t["target"] for t in tasks]
        assert "yellow" in targets

    def test_decompose_returns_vlm_response(self):
        resp = self.vlm.decompose(None, "pick up the red block")
        assert isinstance(resp, VLMResponse)
        assert resp.model_name == "mock-vlm-v1"
        assert len(resp.sub_tasks) > 0


# ── JSON parsing ──────────────────────────────────────────────────

class TestJSONParsing:
    def test_parse_raw_json(self):
        text = '[{"primitive": "pick", "target": "red"}]'
        result = VLMBase._parse_json_subtasks(text)
        assert len(result) == 1
        assert result[0]["primitive"] == "pick"

    def test_parse_json_in_text(self):
        text = 'Here are the steps:\n[{"primitive": "pick", "target": "red"}]\nDone.'
        result = VLMBase._parse_json_subtasks(text)
        assert len(result) == 1

    def test_parse_code_block(self):
        text = '```json\n[{"primitive": "pick", "target": "red"}]\n```'
        result = VLMBase._parse_json_subtasks(text)
        assert len(result) == 1

    def test_parse_malformed_returns_empty(self):
        result = VLMBase._parse_json_subtasks("this is not json at all")
        assert result == []

    def test_parse_empty_string(self):
        result = VLMBase._parse_json_subtasks("")
        assert result == []


# ── Task Parser ───────────────────────────────────────────────────

class TestTaskParser:
    def test_valid_subtask(self):
        raw = [{"primitive": "move_to", "target": "red", "description": "go"}]
        plan = parse_subtasks(raw)
        assert plan.valid
        assert plan.n_steps == 1
        assert plan.sub_tasks[0].primitive == "move_to"

    def test_invalid_primitive(self):
        raw = [{"primitive": "fly", "target": "red"}]
        plan = parse_subtasks(raw)
        assert not plan.valid
        assert len(plan.errors) > 0

    def test_missing_target(self):
        raw = [{"primitive": "pick", "target": ""}]
        plan = parse_subtasks(raw)
        assert not plan.valid

    def test_empty_list(self):
        plan = parse_subtasks([])
        assert not plan.valid
        assert "Empty" in plan.errors[0]

    def test_non_dict_item(self):
        raw = ["not a dict"]
        plan = parse_subtasks(raw)
        assert not plan.valid

    def test_color_normalisation(self):
        assert normalise_color("crimson") == "red"
        assert normalise_color("azure") == "blue"
        assert normalise_color("RED") == "red"
        assert normalise_color("unknown") == "unknown"

    def test_ordering_warning_double_pick(self):
        raw = [
            {"primitive": "pick", "target": "red"},
            {"primitive": "pick", "target": "blue"},
        ]
        plan = parse_subtasks(raw)
        assert plan.valid  # valid structure, but warning
        assert len(plan.warnings) > 0

    def test_ordering_warning_place_without_pick(self):
        raw = [
            {"primitive": "place", "target": "goal"},
        ]
        plan = parse_subtasks(raw)
        assert plan.valid  # valid structure, but warning
        assert any("not holding" in w for w in plan.warnings)

    def test_full_pick_place_sequence(self):
        raw = [
            {"primitive": "move_to", "target": "red", "description": "approach"},
            {"primitive": "pick", "target": "red", "description": "grasp"},
            {"primitive": "place", "target": "blue", "description": "place"},
        ]
        plan = parse_subtasks(raw)
        assert plan.valid
        assert plan.n_steps == 3
        assert plan.primitives() == ["move_to", "pick", "place"]
        assert plan.targets() == ["red", "red", "blue"]
        assert len(plan.warnings) == 0

    def test_to_dict(self):
        raw = [{"primitive": "move_to", "target": "red"}]
        plan = parse_subtasks(raw)
        d = plan.to_dict()
        assert "sub_tasks" in d
        assert "valid" in d


# ── Prompt Templates ──────────────────────────────────────────────

class TestPromptTemplates:
    def test_decomposition_prompt_format(self):
        prompt = build_decomposition_prompt("pick up the red block")
        assert "pick up the red block" in prompt
        assert "move_to" in prompt
        assert "pick" in prompt
        assert "place" in prompt

    def test_valid_primitives(self):
        assert VALID_PRIMITIVES == {"move_to", "pick", "place"}

    def test_eval_scenarios_count(self):
        assert len(EVAL_SCENARIOS) >= 20

    def test_eval_scenarios_have_instruction(self):
        for s in EVAL_SCENARIOS:
            assert "instruction" in s
            assert len(s["instruction"]) > 0


# ── Planner ───────────────────────────────────────────────────────

class TestPlanner:
    def setup_method(self):
        self.planner = Planner(MockVLM())

    def test_plan_returns_task_plan(self):
        plan = self.planner.plan("pick up the red block")
        assert isinstance(plan, TaskPlan)
        assert plan.valid

    def test_plan_pick_place(self):
        plan = self.planner.plan(
            "pick up the red block and place it on the blue block")
        assert plan.valid
        prims = plan.primitives()
        assert "move_to" in prims
        assert "pick" in prims
        assert "place" in prims

    def test_plan_move_to(self):
        plan = self.planner.plan("move to the green object")
        assert plan.valid
        assert plan.sub_tasks[0].primitive == "move_to"

    def test_history_tracking(self):
        self.planner.plan("pick up the red block")
        self.planner.plan("move to the blue one")
        assert len(self.planner.history) == 2
        assert self.planner.history[0]["instruction"] == "pick up the red block"

    def test_clear_history(self):
        self.planner.plan("pick up the red block")
        self.planner.clear_history()
        assert len(self.planner.history) == 0


# ── Decomposition Accuracy on EVAL_SCENARIOS ──────────────────────

class TestDecompositionAccuracy:
    """Test MockVLM accuracy on the 20+ eval scenarios."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.planner = Planner(MockVLM())

    @pytest.mark.parametrize(
        "scenario",
        EVAL_SCENARIOS,
        ids=[s["instruction"][:40] for s in EVAL_SCENARIOS],
    )
    def test_scenario_produces_valid_plan(self, scenario: dict):
        """Every scenario should produce a valid, non-empty plan."""
        plan = self.planner.plan(scenario["instruction"])
        assert plan.valid, (
            f"Invalid plan for '{scenario['instruction']}': {plan.errors}"
        )
        assert plan.n_steps > 0

    @pytest.mark.parametrize(
        "scenario",
        [s for s in EVAL_SCENARIOS if s.get("expected_primitives")],
        ids=[s["instruction"][:40]
             for s in EVAL_SCENARIOS if s.get("expected_primitives")],
    )
    def test_scenario_primitive_match(self, scenario: dict):
        """Scenarios with expected primitives should match exactly."""
        plan = self.planner.plan(scenario["instruction"])
        expected = scenario["expected_primitives"]
        actual = plan.primitives()
        assert actual == expected, (
            f"Primitives mismatch for '{scenario['instruction']}':\n"
            f"  expected: {expected}\n  actual:   {actual}"
        )

    def test_overall_accuracy(self):
        """Measure overall decomposition accuracy across all scenarios."""
        total = 0
        correct = 0
        for scenario in EVAL_SCENARIOS:
            plan = self.planner.plan(scenario["instruction"])
            if not plan.valid:
                total += 1
                continue

            expected_prims = scenario.get("expected_primitives")
            if expected_prims is not None:
                total += 1
                if plan.primitives() == expected_prims:
                    correct += 1

        accuracy = correct / total if total > 0 else 0
        print(f"\nDecomposition accuracy: {correct}/{total} = {accuracy:.0%}")
        assert accuracy >= 0.80, (
            f"Accuracy {accuracy:.0%} < 80% threshold"
        )
