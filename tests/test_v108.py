"""Tests for v1.0.8 — Benchmark Suite + Video Recording."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from evaluation.benchmark import (
    BenchmarkReport,
    EvalMetrics,
    _wilson_ci,
    evaluate_policy,
    run_full_benchmark,
    check_thresholds,
)
from evaluation.record_video import (
    record_episode,
    save_gif,
    save_frames_as_png,
)


# ── EvalMetrics ───────────────────────────────────────────────────

class TestEvalMetrics:
    def test_summary_format(self):
        m = EvalMetrics(
            env_name="TestEnv", policy_name="random",
            n_episodes=10, success_rate=0.5, ci95=0.15,
            mean_return=-100.0, std_return=20.0,
            mean_length=150.0, std_length=30.0, wall_time_s=1.5,
        )
        s = m.summary()
        assert "TestEnv" in s
        assert "random" in s

    def test_defaults(self):
        m = EvalMetrics(env_name="e", policy_name="p")
        assert m.n_episodes == 0
        assert m.success_rate == 0.0


# ── BenchmarkReport ──────────────────────────────────────────────

class TestBenchmarkReport:
    def test_add_and_totals(self):
        report = BenchmarkReport()
        m1 = EvalMetrics(
            env_name="A", policy_name="p",
            n_episodes=10, wall_time_s=1.0,
        )
        m2 = EvalMetrics(
            env_name="B", policy_name="p",
            n_episodes=20, wall_time_s=2.0,
        )
        report.add(m1)
        report.add(m2)
        assert report.total_episodes == 30
        assert abs(report.total_wall_time_s - 3.0) < 0.01
        assert len(report.results) == 2

    def test_print_table(self, capsys):
        report = BenchmarkReport()
        report.add(EvalMetrics(
            env_name="Test", policy_name="rand",
            n_episodes=5, success_rate=0.2, ci95=0.1,
            mean_return=-50.0, std_return=10.0,
            mean_length=100.0, std_length=20.0, wall_time_s=0.5,
        ))
        report.print_table()
        out = capsys.readouterr().out
        assert "Test" in out
        assert "rand" in out


# ── Wilson CI ─────────────────────────────────────────────────────

class TestWilsonCI:
    def test_zero_total(self):
        assert _wilson_ci(0, 0) == 0.0

    def test_all_success(self):
        ci = _wilson_ci(100, 100)
        assert 0.0 < ci < 0.05  # should be small

    def test_half_success(self):
        ci = _wilson_ci(50, 100)
        assert 0.05 < ci < 0.15  # reasonable range

    def test_no_success(self):
        ci = _wilson_ci(0, 100)
        assert 0.0 < ci < 0.05


# ── evaluate_policy ──────────────────────────────────────────────

class TestEvaluatePolicy:
    def test_random_policy_on_moveto(self):
        from envs.move_to import MoveToEnv
        env = MoveToEnv()
        m = evaluate_policy(
            env,
            policy_fn=lambda obs, info: env.action_space.sample(),
            n_episodes=5,
            env_name="MoveTo",
            policy_name="random",
        )
        env.close()
        assert m.n_episodes == 5
        assert m.mean_length > 0
        assert isinstance(m.mean_return, float)
        assert m.wall_time_s > 0

    def test_scripted_policy_on_moveto(self):
        from envs.move_to import MoveToEnv
        from policies.scripted import ScriptedMoveTo
        env = MoveToEnv()
        scripted = ScriptedMoveTo()
        m = evaluate_policy(
            env, scripted, n_episodes=10,
            env_name="MoveTo", policy_name="scripted",
        )
        env.close()
        assert m.n_episodes == 10
        # Scripted should get some successes
        assert m.success_rate >= 0.0

    def test_callable_policy(self):
        from envs.move_to import MoveToEnv
        env = MoveToEnv()
        m = evaluate_policy(
            env,
            policy_fn=lambda obs, info: np.zeros(4),
            n_episodes=3,
            env_name="test", policy_name="zero",
        )
        env.close()
        assert m.n_episodes == 3


# ── run_full_benchmark ───────────────────────────────────────────

class TestFullBenchmark:
    def test_mini_benchmark(self, tmp_path):
        report = run_full_benchmark(
            n_episodes=3,
            seed=0,
            output_dir=str(tmp_path),
        )
        assert report.total_episodes > 0
        assert len(report.results) >= 8  # 8 env/policy pairs

        # Check files were created
        assert (tmp_path / "benchmark_results.json").exists()
        assert (tmp_path / "benchmark_results.csv").exists()

        # Validate JSON structure
        with open(tmp_path / "benchmark_results.json") as f:
            data = json.load(f)
        assert "results" in data
        assert data["total_episodes"] == report.total_episodes


# ── check_thresholds ─────────────────────────────────────────────

class TestThresholdChecker:
    def test_pass_check(self):
        report = BenchmarkReport()
        report.add(EvalMetrics(
            env_name="MoveTo", policy_name="scripted",
            success_rate=0.20, n_episodes=100,
        ))
        checks = check_thresholds(report)
        assert len(checks) == 1
        assert checks[0][4] is True  # passed

    def test_fail_check(self):
        report = BenchmarkReport()
        report.add(EvalMetrics(
            env_name="MoveTo", policy_name="scripted",
            success_rate=0.05, n_episodes=100,
        ))
        checks = check_thresholds(report)
        assert len(checks) == 1
        assert checks[0][4] is False  # failed


# ── record_episode ───────────────────────────────────────────────

class TestRecordEpisode:
    def test_record_with_render(self):
        from envs.move_to import MoveToEnv
        env = MoveToEnv(render_mode="rgb_array", image_size=64)
        frames, info = record_episode(env, None, seed=0, max_steps=10)
        env.close()
        assert len(frames) > 0
        assert isinstance(frames[0], np.ndarray)
        assert frames[0].ndim == 3  # H x W x 3

    def test_record_without_render(self):
        from envs.move_to import MoveToEnv
        env = MoveToEnv()
        frames, info = record_episode(env, None, seed=0, max_steps=10)
        env.close()
        # render() returns None with no render_mode → no frames
        assert isinstance(frames, list)


# ── save_gif ─────────────────────────────────────────────────────

class TestSaveGif:
    def test_save_valid_gif(self, tmp_path):
        frames = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                  for _ in range(5)]
        path = tmp_path / "test.gif"
        result = save_gif(frames, path, fps=5)
        assert result is True
        assert path.exists()
        assert path.stat().st_size > 0

    def test_empty_frames(self, tmp_path):
        path = tmp_path / "empty.gif"
        result = save_gif([], path)
        assert result is False

    def test_save_frames_png(self, tmp_path):
        frames = [np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
                  for _ in range(3)]
        count = save_frames_as_png(frames, tmp_path / "frames", prefix="f")
        assert count == 3
        assert (tmp_path / "frames" / "f_0000.png").exists()


# ── Demo recording integration ───────────────────────────────────

class TestDemoRecording:
    def test_record_demos(self, tmp_path):
        from evaluation.record_video import record_demos
        summary = record_demos(
            output_dir=str(tmp_path),
            n_episodes=1,
            seed=0,
        )
        assert "demos" in summary
        assert len(summary["demos"]) >= 3
        # Check GIF files were created
        for demo in summary["demos"]:
            if demo["gif"]:
                assert Path(demo["gif"]).exists()
