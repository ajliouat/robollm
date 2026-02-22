"""Tests for v1.0.7 — Full hierarchical pipeline, Sort env, Complex Language env."""

from __future__ import annotations

import numpy as np
import pytest

from envs.sort import SortEnv
from envs.complex_language import ComplexLanguageEnv
from planner import MockVLM, Planner, SimGrounder
from evaluation.pipeline import HierarchicalExecutor, ExecutionResult, StepResult


# ── SortEnv (L4) ──────────────────────────────────────────────────

class TestSortEnv:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.env = SortEnv(n_objects=3)
        yield
        self.env.close()

    def test_obs_shape(self):
        obs, _ = self.env.reset(seed=0)
        # 22 + 7*3 + 3*3 = 22 + 21 + 9 = 52
        assert obs.shape == (self.env.observation_space.shape[0],)

    def test_obs_dim(self):
        obs, _ = self.env.reset(seed=0)
        assert obs.shape[0] == 52

    def test_action_space(self):
        assert self.env.action_space.shape == (4,)

    def test_zone_positions_in_info(self):
        _, info = self.env.reset(seed=0)
        assert "zone_positions" in info
        assert len(info["zone_positions"]) == 3

    def test_n_sorted_in_info(self):
        _, info = self.env.reset(seed=0)
        assert "n_sorted" in info
        assert isinstance(info["n_sorted"], int)

    def test_reward_finite(self):
        self.env.reset(seed=0)
        for _ in range(5):
            _, r, _, _, _ = self.env.step(self.env.action_space.sample())
            assert np.isfinite(r)

    def test_reset_deterministic(self):
        o1, _ = self.env.reset(seed=42)
        o2, _ = self.env.reset(seed=42)
        np.testing.assert_array_equal(o1, o2)

    def test_multiple_episodes(self):
        for seed in range(3):
            obs, _ = self.env.reset(seed=seed)
            for _ in range(10):
                obs, _, term, trunc, _ = self.env.step(
                    self.env.action_space.sample())
                if term or trunc:
                    break


# ── ComplexLanguageEnv (L5) ───────────────────────────────────────

class TestComplexLanguageEnv:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.env = ComplexLanguageEnv(n_objects=3)
        yield
        self.env.close()

    def test_obs_shape(self):
        obs, _ = self.env.reset(seed=0)
        # 22 + 7*3 + 16 = 22 + 21 + 16 = 59
        assert obs.shape == (self.env.observation_space.shape[0],)

    def test_obs_dim(self):
        obs, _ = self.env.reset(seed=0)
        assert obs.shape[0] == 59

    def test_instruction_in_info(self):
        _, info = self.env.reset(seed=0)
        assert "instruction" in info
        assert len(info["instruction"]) > 0

    def test_conditions_in_info(self):
        _, info = self.env.reset(seed=0)
        assert "conditions" in info
        assert "conditions_met" in info
        assert "conditions_total" in info

    def test_instruction_varies_by_seed(self):
        """Different seeds should sometimes produce different instructions."""
        instructions = set()
        for seed in range(20):
            _, info = self.env.reset(seed=seed)
            instructions.add(info["instruction"])
        assert len(instructions) > 1

    def test_reward_finite(self):
        self.env.reset(seed=0)
        for _ in range(5):
            _, r, _, _, _ = self.env.step(self.env.action_space.sample())
            assert np.isfinite(r)

    def test_reset_deterministic(self):
        o1, _ = self.env.reset(seed=42)
        o2, _ = self.env.reset(seed=42)
        np.testing.assert_array_equal(o1, o2)

    def test_multiple_episodes(self):
        for seed in range(3):
            obs, _ = self.env.reset(seed=seed)
            for _ in range(10):
                obs, _, term, trunc, _ = self.env.step(
                    self.env.action_space.sample())
                if term or trunc:
                    break


# ── HierarchicalExecutor ─────────────────────────────────────────

class TestHierarchicalExecutor:
    @pytest.fixture(autouse=True)
    def setup(self):
        from envs.multi_object_env import MultiObjectEnv
        self.env = MultiObjectEnv(n_objects=3)
        self.env.reset(seed=0)
        self.planner = Planner(MockVLM())
        self.executor = HierarchicalExecutor(
            grounder=SimGrounder(),
            max_steps_per_subtask=50,
        )
        yield
        self.env.close()

    def test_execute_move_to(self):
        # Use the actual first object's color so grounding succeeds
        color = self.env._obj_specs[0].color_name
        instruction = f"move to the {color} object"
        plan = self.planner.plan(instruction)
        result = self.executor.execute(self.env, plan, instruction)
        assert isinstance(result, ExecutionResult)
        assert result.n_sub_tasks >= 1
        assert result.total_steps > 0

    def test_execute_pick_and_place(self):
        color = self.env._obj_specs[0].color_name
        plan = self.planner.plan(
            f"pick up the {color} block and place it on the goal")
        result = self.executor.execute(self.env, plan)
        assert result.n_sub_tasks >= 2  # at least move_to + pick
        assert all(isinstance(s, StepResult) for s in result.step_results)

    def test_execute_empty_plan(self):
        from planner.task_parser import TaskPlan
        plan = TaskPlan(sub_tasks=[], valid=False, errors=["empty"])
        result = self.executor.execute(self.env, plan)
        assert not result.overall_success

    def test_execution_result_fields(self):
        color = self.env._obj_specs[0].color_name
        instruction = f"move to the {color} object"
        plan = self.planner.plan(instruction)
        result = self.executor.execute(self.env, plan, "test instruction")
        assert result.instruction == "test instruction"
        assert isinstance(result.total_reward, float)
        assert isinstance(result.total_steps, int)

    def test_sub_task_success_rate(self):
        plan = self.planner.plan("move to the red object")
        result = self.executor.execute(self.env, plan)
        rate = result.sub_task_success_rate
        assert 0.0 <= rate <= 1.0

    def test_grounding_failure_handled(self):
        """Grounding a nonexistent object should not crash."""
        from planner.task_parser import SubTask, TaskPlan
        plan = TaskPlan(
            sub_tasks=[SubTask(
                primitive="move_to",
                target="purple_unicorn",
                description="impossible target",
            )],
            valid=True,
        )
        result = self.executor.execute(self.env, plan)
        assert result.n_sub_tasks == 1
        assert not result.step_results[0].success


# ── End-to-end pipeline test ─────────────────────────────────────

class TestEndToEndPipeline:
    """Integration test: instruction → plan → ground → execute."""

    def test_full_pipeline(self):
        from envs.multi_object_env import MultiObjectEnv

        env = MultiObjectEnv(n_objects=3)
        obs, info = env.reset(seed=42)

        planner = Planner(MockVLM())
        executor = HierarchicalExecutor(
            grounder=SimGrounder(),
            max_steps_per_subtask=50,
        )

        instruction = "pick up the red block"
        plan = planner.plan(instruction)
        assert plan.valid

        result = executor.execute(env, plan, instruction)
        assert result.n_sub_tasks > 0
        assert result.total_steps > 0

        env.close()
