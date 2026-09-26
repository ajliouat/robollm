"""V2 comparison integrity; no reserved evaluation seeds are simulated."""
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from evaluation import control_repair as study, frozen_v1


class Scene:
    def __init__(self, *, home=0.0, shifted_object=False, initially_satisfied=False):
        from gymnasium.spaces import Box
        self.action_space = Box(-1, 1, (4,), dtype=np.float64)
        self.home, self.shifted_object = home, shifted_object
        self.initially_satisfied = initially_satisfied
        self._max_episode_steps = 200
        self._obj_specs = [SimpleNamespace(name=f"object{i}", color_name=color, shape="box")
                           for i, color in enumerate(("red", "blue", "green"))]
        self.closed = False

    def reset(self, *, seed):
        self._elapsed_steps = 0
        self.ee_pos = np.array([self.home, 0., 0.])
        self.positions = np.array([[-.4, 0, 0], [.2, 0, 0], [0, .2, 0]])
        if self.shifted_object:
            self.positions[0, 0] += .01
        if self.initially_satisfied:
            self.ee_pos = self.positions[1].copy()
        return self.ee_pos.copy(), {}

    def object_pos(self, index):
        return self.positions[index].copy()

    def step(self, action):
        self._elapsed_steps += 1
        self.ee_pos += np.asarray(action)[:3] * .05
        return self.ee_pos.copy(), 0., False, self._elapsed_steps >= self._max_episode_steps, {}

    def close(self):
        self.closed = True


def factories(**kwargs):
    return {variant: lambda: Scene(**kwargs) for variant in study.VARIANTS}


def test_protocol_reserves_new_scenes_and_preserves_subset_assignment():
    p = study.load_protocol()
    assert set(p["held_out_seeds"]).isdisjoint(p["development_seeds"])
    assert set(p["held_out_seeds"]).isdisjoint(p["previously_observed_evaluation_seeds"])
    assert p["held_out_seeds"] == list(range(260927000, 260927100))
    assert study.select_seeds(p, "development", [18, 101]) == [(1, 18), (2, 101)]
    report = study.run_study(seeds=[18], fixture_factories=factories())
    assert {e["target_index"] for e in report["episodes"]} == {2}
    assert report["test_fixture"]


@pytest.mark.parametrize("split,seeds", [
    ("development", []), ("development", [True]), ("development", [17.]),
    ("development", [17, 17]), ("development", [19]),
    ("development", [260927000]), ("held-out", [260927000]),
    ("held-out", list(reversed(study.HELD_OUT_SEEDS))), ("other", [17]),
])
def test_invalid_seed_requests_are_rejected(split, seeds):
    with pytest.raises(ValueError):
        study.select_seeds(study.load_protocol(), split, seeds)


def test_changed_arm_initial_state_is_recorded_but_same_object_scene_is_paired():
    report = study.run_study(seeds=[17], fixture_factories={
        "frozen_v1": Scene, "repaired_system": lambda: Scene(home=.05),
    })
    a, b = report["scenes"]
    assert a["shared_scene_sha256"] == b["shared_scene_sha256"]
    assert a["full_initial_state_sha256"] != b["full_initial_state_sha256"]
    assert report["summary"]["paired"]["both_success"] == 1
    for e in report["episodes"]:
        assert e["trace"][-1][14] < .03
        assert all(row[14] >= .03 for row in e["trace"][:-1])
        assert all(row[4] == 1 for row in e["trace"])


def test_changed_object_scene_fails_and_closes_both_environments():
    made = []
    def make(shift):
        env = Scene(shifted_object=shift)
        made.append(env)
        return env
    with pytest.raises(RuntimeError, match="Shared object scene differs"):
        study.run_study(seeds=[17], fixture_factories={
            "frozen_v1": lambda: make(False), "repaired_system": lambda: make(True),
        })
    assert len(made) == 2 and all(e.closed for e in made)


def test_initially_satisfied_variant_has_no_credit_and_stays_in_denominator():
    report = study.run_study(seeds=[17], fixture_factories={
        "frozen_v1": Scene, "repaired_system": lambda: Scene(initially_satisfied=True),
    })
    repaired = report["episodes"][1]
    assert repaired["initially_satisfied"] and repaired["steps"] == 0
    assert not repaired["success"]
    assert report["summary"]["paired"]["frozen_only"] == 1
    assert report["summary"]["variants"]["repaired_system"]["n_episodes"] == 1


@pytest.mark.parametrize("revision,dirty", [("abc", True), (None, False), ("abc", None)])
def test_real_heldout_requires_clean_revision_before_loading_variants(monkeypatch, revision, dirty):
    monkeypatch.setattr(study, "_runtime_provenance", lambda: {"git_revision": revision, "git_dirty": dirty})
    monkeypatch.setattr(study, "_real_variants", lambda: pytest.fail("Reserved scenes must not be loaded"))
    with pytest.raises(RuntimeError, match="clean recorded Git revision"):
        study.run_study(split="held-out")


def test_frozen_module_and_dependency_are_isolated_from_production(monkeypatch):
    from envs import object_spawner
    production_module = sys.modules["envs.object_spawner"]
    monkeypatch.setattr(object_spawner, "generate_object_specs", lambda *a, **k: pytest.fail("Production spawner leaked"))
    monkeypatch.setattr(object_spawner, "inject_objects_into_xml", lambda *a, **k: pytest.fail("Production XML leaked"))
    cls = frozen_v1.environment_class()
    env = cls(n_objects=3, render_mode=None, max_episode_steps=200)
    try:
        env.reset(seed=17)
        assert cls.__module__ == "evaluation.frozen_v1._environment"
        assert len(env._obj_specs) == 3
        assert sys.modules["envs.object_spawner"] is production_module
        assert cls.reset.__globals__["_SCENE_XML"].is_relative_to(frozen_v1.ROOT)
        np.testing.assert_array_equal(env.data.qpos[:9], [0., -.785, 0., -2.356, 0., 1.571, .785, .02, .02])
        np.testing.assert_array_equal(env.data.ctrl, np.zeros(9))
    finally:
        env.close()


def test_frozen_hash_validation_rejects_modified_baseline(tmp_path, monkeypatch):
    import shutil
    shutil.copytree(frozen_v1.ROOT, tmp_path / "frozen")
    monkeypatch.setattr(frozen_v1, "ROOT", tmp_path / "frozen")
    (frozen_v1.ROOT / "assets/tabletop_scene.xml").write_text("modified")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        frozen_v1.manifest()


def test_success_threshold_cannot_drift(monkeypatch):
    monkeypatch.setattr(study.HierarchicalExecutor, "REACH_DISTANCE", .04)
    with pytest.raises(RuntimeError, match="threshold differs"):
        study.run_study(seeds=[17], fixture_factories=factories())


def test_summary_refuses_missing_duplicate_or_mismatched_pairs():
    r = study.run_study(seeds=[17], fixture_factories=factories())
    with pytest.raises(RuntimeError, match="Missing or duplicate"):
        study.summarize(r["episodes"][:1], [17])
    with pytest.raises(RuntimeError, match="Missing or duplicate"):
        study.summarize(r["episodes"] + r["episodes"][:1], [17])
    r["episodes"][1]["shared_scene_sha256"] = "wrong"
    with pytest.raises(RuntimeError, match="unpaired"):
        study.summarize(r["episodes"], [17])


def test_protected_output_is_rejected_before_any_execution(tmp_path, monkeypatch):
    path = tmp_path / "existing.json"
    path.write_text("original")
    monkeypatch.setattr(study, "run_study", lambda *a, **k: pytest.fail("Unexpected run"))
    with pytest.raises(SystemExit):
        study.main(["--seeds", "17", "--output", str(path)])
    assert path.read_text() == "original"


def test_real_development_replay_retains_diagnostics_and_distinct_variants():
    first = study.run_study(seeds=[17])
    second = study.run_study(seeds=[17])
    assert not first["test_fixture"]
    assert study.semantic_payload(first) == study.semantic_payload(second)
    assert first["provenance"] == second["provenance"]
    assert len(first["episodes"]) == 2
    assert {e["variant"] for e in first["episodes"]} == set(study.VARIANTS)
    for e in first["episodes"]:
        assert len(e["diagnostics"]) == len(e["trace"]) == e["steps"]
        for d in e["diagnostics"]:
            assert len(d) == len(study.DIAGNOSTIC_COLUMNS)
            assert all(len(values) == 7 for values in d[1:9])
            assert len(d[9]) == 7
            assert d[10] is None or len(d[10]) == 7
            assert d[12] == pytest.approx(d[0] * .05)
        json.dumps(e, allow_nan=False)
    assert all("model_parameters_sha256" in s["full_initial_state"] for s in first["scenes"])


def test_numerical_failure_retains_partial_episode_without_summary():
    class Broken(Scene):
        def step(self, action):
            out = super().step(action)
            if self._elapsed_steps == 2:
                return out[0], float("nan"), *out[2:]
            return out
    with pytest.raises(study.StudyExecutionError, match="Nonfinite reward") as caught:
        study.run_study(seeds=[17], fixture_factories={"frozen_v1": Broken, "repaired_system": Scene})
    report = caught.value.report
    assert report["status"] == "aborted" and "summary" not in report
    episode = report["episodes"][0]
    assert episode["status"] == "aborted" and episode["steps"] == 1
    assert episode["attempted_steps"] == 2 and episode["pending_action"] is not None
    assert "success" not in episode
    assert len(episode["trace"]) == 1
    json.dumps(report, allow_nan=False)


def test_cli_preserves_aborted_evidence_and_fails(tmp_path, monkeypatch):
    partial = {"status": "aborted", "episodes": [], "failure": {"error": "simulation failed"}}
    def fail(*args, **kwargs):
        raise study.StudyExecutionError("simulation failed", partial)
    monkeypatch.setattr(study, "run_study", fail)
    output = tmp_path / "aborted.json"
    with pytest.raises(SystemExit) as caught:
        study.main(["--seeds", "17", "--output", str(output)])
    assert caught.value.code == 2
    assert json.loads(output.read_text()) == partial
