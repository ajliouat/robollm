"""Same-simulator controller study with a stationary standoff/hold contract.

Development is the default. Reserved scenes require a clean frozen revision.
Neither legacy proximity study nor its success definition is modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from envs.multi_object_env import MultiObjectEnv
from evaluation.benchmark import _runtime_provenance, wilson_interval
from evaluation.control_repair import _initial_state
from evaluation.grounded_control import _canonical, save_report
from planner.grounder import SimGrounder
from policies.scripted import ScriptedMoveTo

REPO = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = Path(__file__).with_name('stationary_protocol.json')
VARIANTS = ('direct_dls', 'collision_aware')
DEVELOPMENT_SEEDS = [17, 18, *range(101, 111)]
HELD_OUT_SEEDS = list(range(260928000, 260928100))
PHYSICS_COLUMNS = ['sample_index', 'time_seconds', 'ee_position', 'object_poses',
                   'contacts_geom_names_distance_m_moving_robot', 'goal_distance_m', 'dwell_intervals', 'max_motion_bound_m']
CONTROL_COLUMNS = ['sample_index_before_action', 'arm_qpos', 'arm_qvel', 'joint_targets',
                   'finger_targets']


class StudyExecutionError(RuntimeError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


class EpisodeExecutionError(RuntimeError):
    def __init__(self, message, episode):
        super().__init__(message)
        self.episode = episode


def sha(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def semantic_payload(report):
    # CPU planning/execution duration differs; all geometry, outcomes and
    # deterministic planner work counters remain in the exact comparison.
    return {k: v for k, v in report.items()
            if k not in {'timestamp_utc', 'elapsed_seconds', 'provenance', 'timings'}}


def semantic_digest(report):
    return sha(semantic_payload(report))


def load_protocol():
    p = json.loads(PROTOCOL_PATH.read_text())
    if (p['development_seeds'] != DEVELOPMENT_SEEDS or p['held_out_seeds'] != HELD_OUT_SEEDS
            or p['variants'] != list(VARIANTS) or p['schema_version'] != 3):
        raise ValueError('Protocol differs from the declared study')
    expected = ((p['environment']['physics_dt_seconds'], .002),
                (p['environment']['control_dt_seconds'], .05),
                (p['environment']['horizon_physics_intervals'], 5000),
                (p['goal']['clearance_above_initial_object_top_m'], .1),
                (p['success']['goal_distance_strictly_below_m'], .02),
                (p['success']['hold_elapsed_physics_intervals'], 500),
                (p['object_motion']['max_surface_displacement_bound_m'], .005))
    if any(actual != fixed for actual, fixed in expected):
        raise ValueError('Task constants differ from the frozen contract')
    from evaluation import stationary_contract as contract
    declared = {'GOAL_CLEARANCE': .1, 'GOAL_RADIUS': .02, 'PHYSICS_DT': .002,
                'HOLD_INTERVALS': 500, 'HORIZON_INTERVALS': 5000,
                'MAX_OBJECT_POSE_DISPLACEMENT': .005}
    if any(getattr(contract, name) != value for name, value in declared.items()) or (
            set(p['contacts']['moving_robot_bodies']) != contract.MOVING_ROBOT_BODIES):
        raise ValueError('Monitor implementation differs from the frozen protocol')
    if (p['environment']['n_objects'] != 3 or p['environment']['settling_steps'] != 0
            or p['environment']['fingers_command_m'] != .04):
        raise ValueError('Environment constants differ from the frozen protocol')
    from policies.stationary_reach import StationaryReachController
    actual = {name: getattr(StationaryReachController, name)
              for name in vars(StationaryReachController) if name.isupper()}
    if p['controllers']['collision_aware']['parameters'] != actual or p['controllers']['planner_seed'] != 1729:
        raise ValueError('Planner parameters differ from the frozen protocol')
    return p


def select_seeds(split, seeds=None):
    if split not in ('development', 'held-out'):
        raise ValueError('Unknown split')
    declared = DEVELOPMENT_SEEDS if split == 'development' else HELD_OUT_SEEDS
    selected = list(declared if seeds is None else seeds)
    if (not selected or any(type(s) is not int or s not in declared for s in selected)
            or len(set(selected)) != len(selected)):
        raise ValueError('Seeds must be unique integers in the declared split')
    if split == 'held-out' and selected != declared:
        raise ValueError('Reserved evaluation retains all 100 seeds in declared order')
    return [(declared.index(seed), seed) for seed in selected]


class DirectDLS:
    """Original gain-8 Cartesian law and environment's existing IK mapping."""
    def __init__(self, env, goal, seed):
        self.env, self.goal = env, np.array(goal, dtype=float)
        self.policy = ScriptedMoveTo(gain=8)
        self.metadata = {'planning_status': 'ready', 'kind': 'gain8_cartesian_dls'}

    def command(self):
        action = self.policy.act({'ee_pos': self.env.ee_pos, 'obj_pos': self.goal})
        return self.env._delta_ee_to_joints(action[:3] * .05)


def controller_factory(variant):
    if variant == 'direct_dls':
        return DirectDLS
    from policies.stationary_reach import StationaryReachController
    return StationaryReachController


def apply_command(env, targets):
    targets = np.asarray(targets, dtype=float)
    if targets.shape != (7,) or not np.all(np.isfinite(targets)):
        raise ValueError('Controller must return seven finite joint targets')
    if np.any(targets < env._jnt_lo) or np.any(targets > env._jnt_hi):
        raise ValueError('Joint target outside the unchanged actuator/joint limits')
    env.data.ctrl[env._arm_act_ids] = targets
    env.data.ctrl[env._finger_l_act] = .04
    env.data.ctrl[env._finger_r_act] = .04
    return targets


def checked_physics_step(env):
    before = float(env.data.time)
    warnings = env.data.warning.number.copy()
    mujoco.mj_step(env.model, env.data)
    if not np.isclose(env.data.time - before, .002, rtol=0, atol=1e-10):
        raise RuntimeError('Unexpected simulator time increment/reset')
    if np.any(env.data.warning.number != warnings):
        raise RuntimeError('MuJoCo warning counter changed')
    if not all(np.all(np.isfinite(getattr(env.data, field)))
               for field in ('qpos', 'qvel', 'qacc', 'ctrl')):
        raise RuntimeError('Nonfinite simulator state')


def compact_sample(index, sample, result):
    return [index, sample['time_seconds'], sample['ee_position'],
            [[*obj['position'], *obj['quaternion']] for obj in sample['objects']],
            [[c['geom1'], c['geom2'], c['signed_distance_m'], c['moving_robot']]
             for c in sample['contacts']], result['current_distance_m'], result['dwell_intervals'],
            result['max_motion_bound_m']]


def run_episode(env, variant, seed, initial, monitor, observer):
    result = dict(monitor.result)
    episode = {'status': 'running', 'variant': variant, 'env_seed': seed,
               'goal_position': list(monitor.goal), 'initial_sample': initial,
               'physics_trace': [compact_sample(0, initial, result)], 'control_trace': [],
               'planner': None}
    controller = None
    timing = {'planning_seconds': 0.0, 'execution_seconds': 0.0}
    try:
        if not result['done']:
            start = time.perf_counter()
            controller = controller_factory(variant)(env, monitor.goal, seed=1729)
            timing['planning_seconds'] = time.perf_counter() - start
            episode['planner'] = json.loads(_canonical(controller.metadata))
            if controller.metadata.get('planning_status') == 'failed':
                result = {**result, 'done': True, 'success': False, 'stop_reason': 'planning_failed'}
        start = time.perf_counter()
        for index in range(1, 5001):
            if result['done']:
                break
            episode['attempted_sample_index'] = index
            if (index - 1) % 25 == 0:
                targets = controller.command()
                if controller.metadata.get('planning_status') == 'failed':
                    result = {**result, 'done': True, 'success': False, 'stop_reason': 'controller_guard'}
                    break
                targets = apply_command(env, targets)
                episode['control_trace'].append([index - 1, env.data.qpos[:7].tolist(),
                    env.data.qvel[:7].tolist(), targets.tolist(), [.04, .04]])
            checked_physics_step(env)
            sample = observer.sample()
            result = dict(monitor.consume(index, sample))
            episode['physics_trace'].append(compact_sample(index, sample, result))
        timing['execution_seconds'] = time.perf_counter() - start
        if not result['done']:
            raise RuntimeError('Monitor did not terminate at the fixed horizon')
        if controller is not None:
            episode['planner'] = json.loads(_canonical(controller.metadata))
        episode.update({'status': 'complete', 'result': result,
                        'physics_intervals': len(episode['physics_trace']) - 1})
        _canonical(episode)
        return episode, timing
    except Exception as exc:
        episode.update({'status': 'aborted', 'partial_result': result,
                        'error_type': type(exc).__name__, 'error': str(exc)})
        raise EpisodeExecutionError(str(exc), episode) from exc


def summarize(episodes, seeds):
    by_variant, summary = {}, {}
    for variant in VARIANTS:
        rows = [e for e in episodes if e['variant'] == variant]
        indexed = {e['env_seed']: e for e in rows}
        if len(rows) != len(seeds) or set(indexed) != set(seeds):
            raise ValueError('Incomplete or duplicate episode denominator')
        if any(e['status'] != 'complete' or not e['result']['done'] for e in rows):
            raise ValueError('Cannot aggregate incomplete episodes')
        by_variant[variant] = indexed
        count = sum(e['result']['success'] for e in rows)
        low, high = wilson_interval(count, len(seeds))
        summary[variant] = {'n_episodes': len(seeds), 'n_successes': count,
            'success_rate': count/len(seeds), 'wilson95_low': low, 'wilson95_high': high,
            'stop_reasons': dict(Counter(e['result']['stop_reason'] for e in rows)),
            'mean_physics_intervals': float(np.mean([e['physics_intervals'] for e in rows])),
            'mean_final_distance_m': float(np.mean([e['result']['current_distance_m'] for e in rows])),
            'max_motion_bound_m': max(e['result']['max_motion_bound_m'] for e in rows)}
    paired = dict.fromkeys(('both_success', 'both_failure', 'candidate_only', 'direct_only'), 0)
    for seed in seeds:
        a, b = (by_variant[v][seed] for v in VARIANTS)
        if a['initial_state_sha256'] != b['initial_state_sha256'] or a['goal_position'] != b['goal_position']:
            raise ValueError('Episode pair has different full initial state or goal')
        x, y = a['result']['success'], b['result']['success']
        paired['both_success' if x and y else 'direct_only' if x else 'candidate_only' if y else 'both_failure'] += 1
    paired.update(n_pairs=len(seeds), success_rate_difference=(paired['candidate_only']-paired['direct_only'])/len(seeds))
    return {'variants': summary, 'paired': paired}


def implementation_manifest():
    paths = [str(p.relative_to(REPO)) for folder in ('envs', 'planner', 'policies', 'evaluation')
             for p in (REPO/folder).rglob('*')
             if p.is_file() and p.suffix in ('.py', '.xml') and 'results' not in p.parts]
    paths.append(str(PROTOCOL_PATH.relative_to(REPO)))
    return {p: hashlib.sha256((REPO/p).read_bytes()).hexdigest() for p in sorted(paths)}


def run_study(split='development', seeds=None):
    from evaluation.stationary_contract import MujocoObserver, StationaryMonitor
    protocol = load_protocol()
    selection = select_seeds(split, seeds)
    provenance = _runtime_provenance()
    if split == 'held-out' and (not provenance.get('git_revision') or provenance.get('git_dirty') is not False):
        raise ValueError('Reserved evaluation requires a clean recorded revision')
    report = {'schema_version': 3, 'study_id': protocol['study_id'], 'split': split,
              'status': 'running', 'protocol': protocol, 'protocol_sha256': sha(protocol),
              'selected_seeds': [seed for _, seed in selection], 'provenance': provenance,
              'implementation_manifest': implementation_manifest(),
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'physics_columns': PHYSICS_COLUMNS, 'control_columns': CONTROL_COLUMNS,
              'scenes': [], 'episodes': [], 'timings': []}
    started = time.perf_counter()
    current = {}
    try:
        for index, seed in selection:
            expected = None
            for variant in VARIANTS:
                current = {'variant': variant, 'env_seed': seed, 'episode_index': index}
                env = MultiObjectEnv(n_objects=3, render_mode=None, max_episode_steps=200)
                try:
                    observation, info = env.reset(seed=seed)
                    if env.model.opt.timestep != .002:
                        raise ValueError('Unexpected physics timestep')
                    target_index = 1 + index % 2
                    target_name = env.object_name(target_index)
                    grounded = SimGrounder().ground(target_name, info)
                    if not grounded.success or grounded.matched is None or grounded.matched.name != target_name:
                        raise ValueError('Target grounding is not exact and unique')
                    if sum(obj.name == target_name for obj in env._obj_specs) != 1:
                        raise ValueError('Ambiguous target identity')
                    full, _ = _initial_state(env, observation, target_index, False)
                    full['additional_model_parameters'] = {name: np.asarray(getattr(env.model, name)).tolist()
                        for name in ('body_ipos', 'body_iquat', 'geom_bodyid', 'geom_margin', 'geom_gap',
                                     'geom_solmix', 'pair_geom1', 'pair_geom2', 'pair_margin', 'pair_gap')}
                    full['additional_model_options'] = {name: int(getattr(env.model.opt, name))
                        for name in ('enableflags', 'disableflags', 'cone', 'jacobian')}
                    observer = MujocoObserver(env)
                    initial = observer.sample()
                    monitor = StationaryMonitor(initial, target_index)
                    snapshot = {'simulator': full, 'initial_sample': initial,
                                'fixed_goal': list(monitor.goal), 'target_name': target_name}
                    state_hash = sha(snapshot)
                    report['scenes'].append({**current, 'initial_state_sha256': state_hash,
                                             'initial_state': snapshot})
                    if expected is not None and state_hash != expected:
                        raise ValueError('Paired full initial states differ')
                    expected = state_hash
                    episode, timing = run_episode(env, variant, seed, initial, monitor, observer)
                    episode.update(episode_index=index, target_name=target_name, target_index=target_index,
                                   initial_state_sha256=state_hash)
                    report['episodes'].append(episode)
                    report['timings'].append({**current, **timing})
                except EpisodeExecutionError as exc:
                    exc.episode.update(current)
                    report['episodes'].append(exc.episode)
                    raise
                finally:
                    env.close()
        if split == 'held-out' and _runtime_provenance() != provenance:
            raise RuntimeError('Code/runtime changed during reserved evaluation')
        report['summary'] = summarize(report['episodes'], report['selected_seeds'])
        report['status'] = 'complete'
        report['elapsed_seconds'] = time.perf_counter() - started
        _canonical(report)
        return report
    except Exception as exc:
        report.update(status='aborted', failure={**current, 'type': type(exc).__name__, 'message': str(exc)},
                      elapsed_seconds=time.perf_counter() - started)
        report.pop('summary', None)
        raise StudyExecutionError(str(exc), report) from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('development', 'held-out'), default='development')
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        select_seeds(args.split, args.seeds)
        if args.output.exists() or not (args.output.name.endswith('.json') or args.output.name.endswith('.json.gz')):
            raise ValueError('Use a new .json or .json.gz output path')
    except ValueError as exc:
        parser.error(str(exc))
    try:
        report = run_study(args.split, args.seeds)
    except StudyExecutionError as exc:
        save_report(exc.report, args.output)
        parser.exit(2, f'{exc}; incomplete evidence retained without aggregate\n')
    save_report(report, args.output)
    print(json.dumps({'semantic_sha256': semantic_digest(report), 'summary': report['summary']}, indent=2))


if __name__ == '__main__':
    main()
