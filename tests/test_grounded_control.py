"""Protocol/paired-control regressions; real simulation uses development seeds only."""

from __future__ import annotations

import gzip
import json
from types import SimpleNamespace

import numpy as np
import pytest

from evaluation import grounded_control as study


class _ActionSpace:
    def seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def sample(self):
        return self.rng.uniform(-1, 1, 4)


class _KinematicEnv:
    """Transparent test fixture, not a MuJoCo substitute for measured outcomes."""

    def __init__(self, *, initially_satisfied=False, passive_motion=False):
        self.action_space = _ActionSpace()
        self._max_episode_steps = 200
        self.initially_satisfied = initially_satisfied
        self.passive_motion = passive_motion
        self.closed = False
        self._obj_specs = [
            SimpleNamespace(name=f"object{i}", color_name=color, shape="box")
            for i, color in enumerate(("red", "blue", "green"))
        ]

    def reset(self, *, seed):
        self.ee_pos = np.zeros(3)
        self.positions = np.array([[-0.4, 0, 0], [0.2, 0, 0], [0, 0.2, 0]], dtype=float)
        if self.initially_satisfied:
            self.positions[1:] *= 0.05
        self._elapsed_steps = 0
        return self.ee_pos.copy(), {}

    def object_pos(self, index):
        return self.positions[index].copy()

    def step(self, action):
        self.ee_pos += np.asarray(action)[:3] * 0.05
        self._elapsed_steps += 1
        if self.passive_motion:
            self.positions[1, 0] = max(0.0, self.positions[1, 0] - 0.05)
        return (
            self.ee_pos.copy(),
            0.0,
            False,
            self._elapsed_steps >= self._max_episode_steps,
            {},
        )

    def close(self):
        self.closed = True


def test_frozen_protocol_and_subset_keep_original_target_assignment():
    protocol = study.load_protocol()
    assert protocol["controller"] == {
        "gain": 8.0,
        "threshold_m": 0.03,
        "horizon_steps": 200,
        "gripper_action": 1.0,
    }
    assert study.select_seeds(protocol, "smoke", [18]) == [(1, 18)]
    held_out = study.select_seeds(protocol, "held-out")
    assert held_out == list(enumerate(range(260926000, 260926100)))
    report = study.run_study(seeds=[18], env_factory=_KinematicEnv)
    assert all(episode["target_index"] == 2 for episode in report["episodes"])
    assert report["test_fixture"] is True


@pytest.mark.parametrize(
    "split,seeds",
    [
        ("smoke", []),
        ("smoke", [17, 17]),
        ("smoke", [True]),
        ("smoke", [17.0]),
        ("smoke", [19]),
        ("smoke", [260926000]),
        ("held-out", [260926000]),
        ("other", [17]),
    ],
)
def test_invalid_or_unbounded_seed_requests_rejected(split, seeds):
    with pytest.raises(ValueError):
        study.select_seeds(study.load_protocol(), split, seeds)


def test_paired_actions_share_named_live_target_oracle_and_initial_scene():
    report = study.run_study(env_factory=_KinematicEnv)
    for seed in (17, 18):
        episodes = [
            episode for episode in report["episodes"] if episode["env_seed"] == seed
        ]
        assert len({episode["initial_scene_sha256"] for episode in episodes}) == 1
        assert len({episode["target_name"] for episode in episodes}) == 1
        arms = {episode["arm"]: episode for episode in episodes}
        assert arms["grounded_executor"]["success"]
        # The ablation reaches object0, which cannot earn named-target credit.
        assert arms["first_object_action_ablation"]["final_object0_distance_m"] < 0.03
        assert not arms["first_object_action_ablation"]["success"]
        assert not arms["zero_xyz_open"]["success"]
        for episode in episodes:
            assert episode["grounded_name"] == episode["target_name"]
            assert episode["steps"] <= 200
            assert episode["steps"] == len(episode["trace"])
            assert all(row[4] == 1.0 for row in episode["trace"])
            if episode["success"]:
                assert episode["trace"][-1][14] < 0.03
                assert all(row[14] >= 0.03 for row in episode["trace"][:-1])
    paired = report["summary"]["grounded_vs_first_object_action_ablation"]
    assert paired == {
        "both_success": 0,
        "both_failure": 0,
        "grounded_only": 2,
        "ablation_only": 0,
        "n_pairs": 2,
        "success_rate_difference": 1.0,
    }


def test_initial_predicate_is_reported_but_never_control_success():
    report = study.run_study(
        env_factory=lambda: _KinematicEnv(initially_satisfied=True)
    )
    assert all(episode["initially_satisfied"] for episode in report["episodes"])
    assert all(
        not episode["success"] and episode["steps"] == 0
        for episode in report["episodes"]
    )
    for arm in report["summary"]["arms"].values():
        assert arm["n_episodes"] == 2
        assert arm["n_initially_satisfied"] == 2
        assert arm["success_rate"] == 0
        assert arm["wilson95_high"] > 0


def test_noop_detects_passive_target_motion_without_relabeling_it_as_fixed_reaching():
    report = study.run_study(
        seeds=[17], env_factory=lambda: _KinematicEnv(passive_motion=True)
    )
    noop = next(
        episode for episode in report["episodes"] if episode["arm"] == "zero_xyz_open"
    )
    assert noop["success"] and not noop["initially_satisfied"]
    assert noop["max_target_displacement_m"] > 0.15
    assert noop["final_distance_m"] < 0.03
    assert all(
        row[1:4] == [0.0, 0.0, 0.0] and row[5:8] == [0.0, 0.0, 0.0]
        for row in noop["trace"]
    )


def test_exact_replay_ignores_only_runtime_metadata():
    first = study.run_study(seeds=[17], env_factory=_KinematicEnv)
    second = study.run_study(seeds=[17], env_factory=_KinematicEnv)
    assert study.semantic_payload(first) == study.semantic_payload(second)
    assert study.semantic_digest(first) == study.semantic_digest(second)
    second["provenance"] = {"platform": "different", "git_dirty": True}
    second["elapsed_seconds"] = -1
    assert study.semantic_digest(first) == study.semantic_digest(second)
    second["episodes"][0]["trace"][0][1] += 1e-10
    assert study.semantic_digest(first) != study.semantic_digest(second)


def test_pairing_mismatch_fails_and_closes_environment():
    instances = []

    def factory():
        env = _KinematicEnv()
        original_reset = env.reset
        offset = len(instances)
        instances.append(env)

        def reset(*, seed):
            obs, info = original_reset(seed=seed)
            env.positions[0, 0] += offset * 0.01
            return obs, info

        env.reset = reset
        return env

    with pytest.raises(RuntimeError, match="Paired initial scene mismatch"):
        study.run_study(seeds=[17], env_factory=factory)
    assert len(instances) == 2 and all(env.closed for env in instances)


@pytest.mark.parametrize(
    "revision,dirty", [("abc", True), (None, False), ("abc", None)]
)
def test_real_heldout_requires_clean_revision_before_creating_environment(
    monkeypatch, revision, dirty
):
    from envs import multi_object_env

    monkeypatch.setattr(
        study,
        "_runtime_provenance",
        lambda: {"git_revision": revision, "git_dirty": dirty},
    )
    monkeypatch.setattr(
        multi_object_env,
        "MultiObjectEnv",
        lambda **kwargs: pytest.fail("Held-out seed was executed"),
    )
    with pytest.raises(RuntimeError, match="clean recorded Git revision"):
        study.run_study(split="held-out")


def test_protocol_and_executor_threshold_mismatch_is_not_silently_measured(monkeypatch):
    monkeypatch.setattr(study.HierarchicalExecutor, "REACH_DISTANCE", 0.04)
    with pytest.raises(RuntimeError, match="success threshold differs"):
        study.run_study(seeds=[17], env_factory=_KinematicEnv)


def test_json_and_deterministic_gzip_preserve_full_trace_without_overwrite(tmp_path):
    report = study.run_study(seeds=[17], env_factory=_KinematicEnv)
    raw = tmp_path / "one.json"
    compressed = tmp_path / "one.json.gz"
    second_compressed = tmp_path / "two.json.gz"
    study.save_report(report, raw)
    study.save_report(report, compressed)
    study.save_report(report, second_compressed)
    assert gzip.decompress(compressed.read_bytes()) == raw.read_bytes()
    assert compressed.read_bytes() == second_compressed.read_bytes()
    assert json.loads(raw.read_bytes()) == report
    with pytest.raises(FileExistsError):
        study.save_report({"changed": True}, raw)
    assert json.loads(raw.read_bytes()) == report
    with pytest.raises(ValueError):
        study.save_report({"invalid": float("nan")}, raw)
    assert json.loads(raw.read_bytes()) == report
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "one.json",
        "one.json.gz",
        "two.json.gz",
    ]


def test_cli_rejects_existing_output_before_running(tmp_path, monkeypatch):
    output = tmp_path / "existing.json"
    output.write_text("original")
    monkeypatch.setattr(
        study, "run_study", lambda *args: pytest.fail("Unexpected simulation")
    )
    with pytest.raises(SystemExit):
        study.main(["--seeds", "17", "18", "--output", str(output)])
    assert output.read_text() == "original"


def test_mujoco_development_seed_replays_complete_semantic_trace():
    first = study.run_study(seeds=[17])
    second = study.run_study(seeds=[17])
    assert not first["test_fixture"]
    assert study.semantic_payload(first) == study.semantic_payload(second)
    assert len(first["episodes"]) == 4
    assert all(episode["target_index"] == 1 for episode in first["episodes"])
    assert len({episode["initial_scene_sha256"] for episode in first["episodes"]}) == 1
