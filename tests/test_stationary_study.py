"""Execution/provenance boundaries of the stationary task, not outcome targets."""
import copy
import numpy as np
import pytest
import mujoco

from envs.multi_object_env import MultiObjectEnv
from evaluation import stationary_reach as study
from evaluation.stationary_contract import MujocoObserver
from policies.scripted import ScriptedMoveTo


def integration_state(env):
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.zeros(mujoco.mj_stateSize(env.model, spec))
    mujoco.mj_getState(env.model, env.data, state, spec)
    return state


def test_monitored_direct_actuation_preserves_existing_physics_exactly():
    first, observed = MultiObjectEnv(), MultiObjectEnv()
    try:
        first.reset(seed=17); observed.reset(seed=17)
        goal = observed.object_pos(1) + [0, 0, .12]
        controller = study.DirectDLS(observed, goal, 17)
        observer = MujocoObserver(observed)
        policy = ScriptedMoveTo(gain=8)
        for _ in range(2):
            action = policy.act({'ee_pos': first.ee_pos, 'obj_pos': goal})
            first.step(action)
            study.apply_command(observed, controller.command())
            for _ in range(25):
                study.checked_physics_step(observed)
                before = integration_state(observed)
                observer.sample()
                np.testing.assert_array_equal(before, integration_state(observed))
            np.testing.assert_array_equal(integration_state(first), integration_state(observed))
    finally:
        first.close(); observed.close()


@pytest.mark.parametrize('seeds', [[], [17, 17], [True], [260927000], [260928000]])
def test_development_selection_refuses_empty_duplicates_and_reserved_or_old_seeds(seeds):
    with pytest.raises(ValueError):
        study.select_seeds('development', seeds)


def test_subset_keeps_target_parity_and_reserved_denominator_cannot_shrink():
    assert study.select_seeds('development', [18, 102]) == [(1, 18), (3, 102)]
    for subset in ([260928000], study.HELD_OUT_SEEDS[::-1]):
        with pytest.raises(ValueError):
            study.select_seeds('held-out', subset)


def test_dirty_reserved_evaluation_refused_before_creating_environment(monkeypatch):
    monkeypatch.setattr(study, '_runtime_provenance', lambda: {'git_revision': 'test', 'git_dirty': True})
    monkeypatch.setattr(study, 'MultiObjectEnv', lambda **kw: pytest.fail('Must not instantiate a reserved scene'))
    with pytest.raises(ValueError, match='clean recorded'):
        study.run_study('held-out')


@pytest.mark.parametrize('fault', ['time', 'warning', 'nonfinite'])
def test_simulator_faults_fail_closed(monkeypatch, fault):
    env = MultiObjectEnv()
    try:
        env.reset(seed=17)
        step = mujoco.mj_step
        def broken(model, data):
            step(model, data)
            if fault == 'time': data.time = 0
            elif fault == 'warning': data.warning.number[0] += 1
            else: data.qvel[0] = np.nan
        monkeypatch.setattr(mujoco, 'mj_step', broken)
        with pytest.raises(RuntimeError): study.checked_physics_step(env)
    finally: env.close()


def test_runtime_failure_preserves_partial_report_and_has_no_summary(monkeypatch):
    closed = []
    class FailedReset:
        def __init__(self, **kw): pass
        def reset(self, **kw): raise RuntimeError('injected reset failure')
        def close(self): closed.append(True)
    monkeypatch.setattr(study, 'MultiObjectEnv', FailedReset)
    with pytest.raises(study.StudyExecutionError) as exc:
        study.run_study(seeds=[17])
    report = exc.value.report
    assert report['status'] == 'aborted' and 'summary' not in report
    assert report['failure']['env_seed'] == 17 and report['failure']['variant'] == 'direct_dls'
    assert 'injected reset failure' in report['failure']['message'] and closed == [True]


def test_complete_denominator_and_pairing_are_required():
    def row(variant, success):
        return {'status':'complete', 'env_seed':17, 'variant':variant,
                'physics_intervals':700, 'goal_position':[0,0,.5], 'initial_state_sha256':'same',
                'result':{'done':True, 'success':success, 'stop_reason':'held_goal' if success else 'horizon',
                          'current_distance_m':.01 if success else .05, 'max_motion_bound_m':.001}}
    rows = [row('direct_dls', False), row('collision_aware', True)]
    result = study.summarize(rows, [17])
    assert result['paired']['candidate_only'] == 1 and result['paired']['success_rate_difference'] == 1
    with pytest.raises(ValueError): study.summarize(rows, [17,18])
    with pytest.raises(ValueError): study.summarize(rows+rows[:1], [17])
    broken=copy.deepcopy(rows);broken[1]['initial_state_sha256']='different'
    with pytest.raises(ValueError): study.summarize(broken,[17])
    broken=copy.deepcopy(rows);broken[1]['status']='aborted'
    with pytest.raises(ValueError): study.summarize(broken,[17])


def test_replay_excludes_only_timing_and_provenance_not_planner_work():
    a={'provenance':{'git_dirty':False}, 'timestamp_utc':'a', 'elapsed_seconds':1,
       'timings':[{'planning_seconds':1}], 'episodes':[{'planner':{'ik_iterations':7}}]}
    b=copy.deepcopy(a);b['timings'][0]['planning_seconds']=999;b['timestamp_utc']='b'
    assert study.semantic_digest(a)==study.semantic_digest(b)
    b['episodes'][0]['planner']['ik_iterations']+=1
    assert study.semantic_digest(a)!=study.semantic_digest(b)


def test_protocol_rejects_monitor_implementation_drift(monkeypatch):
    from evaluation import stationary_contract as contract
    monkeypatch.setattr(contract, 'GOAL_RADIUS', .021)
    with pytest.raises(ValueError, match='Monitor implementation'):
        study.load_protocol()


@pytest.mark.parametrize('kind,expected_intervals,reason', [('no_plan',0,'planning_failed'),('late_guard',25,'controller_guard')])
def test_controller_failure_keeps_failed_episode_and_stops_physics(monkeypatch, kind, expected_intervals, reason):
    from evaluation.stationary_contract import StationaryMonitor
    env = MultiObjectEnv()
    class Unavailable:
        def __init__(self, env, goal, seed):
            self.metadata={'planning_status':'failed' if kind=='no_plan' else 'ready', 'failure':None}
            self.calls=0
        def command(self):
            self.calls+=1
            if self.calls==2: self.metadata.update(planning_status='failed',failure='tracking_clearance')
            return env.data.qpos[:7].copy()
    try:
        env.reset(seed=17);observer=MujocoObserver(env);initial=observer.sample()
        monitor=StationaryMonitor(initial,1)
        monkeypatch.setattr(study,'controller_factory',lambda variant:Unavailable)
        episode,_=study.run_episode(env,'collision_aware',17,initial,monitor,observer)
        assert episode['status']=='complete' and not episode['result']['success']
        assert episode['result']['stop_reason']==reason
        assert episode['physics_intervals']==expected_intervals
        assert env.data.time==pytest.approx(expected_intervals*.002)
    finally: env.close()


def test_cli_retains_mid_episode_failure_and_preceding_traces(monkeypatch,tmp_path):
    import json
    step=study.checked_physics_step
    calls=[]
    def failing(env):
        calls.append(True)
        if len(calls)==2: raise RuntimeError('injected step failure')
        step(env)
    monkeypatch.setattr(study,'checked_physics_step',failing)
    path=tmp_path/'partial.json'
    with pytest.raises(SystemExit) as exc:
        study.main(['--seeds','17','--output',str(path)])
    assert exc.value.code==2
    report=json.loads(path.read_text())
    assert report['status']=='aborted' and 'summary' not in report
    episode=report['episodes'][0]
    assert episode['status']=='aborted' and episode['attempted_sample_index']==2
    assert len(episode['physics_trace'])==2 and len(episode['control_trace'])==1
    assert 'injected step failure' in episode['error']
