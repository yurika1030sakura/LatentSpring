#!/usr/bin/env python3
"""Join every prespecified empirical-CFM arm and retain the stop decision."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-root', type=Path, required=True)
    p.add_argument('--assessment', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    assessment_path = args.assessment/'assessment.json'
    assessment = json.loads(assessment_path.read_text())
    if not assessment['complete'] or len(assessment['xtb_rows']) != 256:
        raise ValueError('Require all256 prespecified xTB attempts')
    xtb = {row['arm']: row for row in assessment['xtb_summaries']}
    geometry = {row['arm']: row for row in assessment['geometry']}
    rows, sources = [], []
    common = None
    for arm in ['source', 'uniform', 'linear', 'power']:
        version = 'v1' if arm in ['source', 'uniform'] else 'v2'
        directory = args.runs_root/f'forward_work_student_{arm}_{version}'
        path = directory/'results.json'
        result = json.loads(path.read_text())
        if not result['complete'] or not result['forward_state_unchanged_during_reverse_refit']:
            raise ValueError('Incomplete or changed forward model')
        config = {k: v for k, v in result['configuration'].items() if k not in ['arm', 'out']}
        checks = {key: result[key] for key in ['condition', 'teacher_sha256', 'source_checkpoint_sha256',
                  'backward_initialization_sha256', 'source_recipe', 'research_protocol', 'seeds']}
        checks['configuration'] = config
        if common is None:
            common = checks
        elif common != checks:
            raise ValueError('Arms differ in target, provenance or controlled settings')
        if result['arm'] != arm or result['new_oracle_evaluations'] != 512:
            raise ValueError('Arm identity or budget mismatch')
        if result['forward_updates'] != (0 if arm == 'source' else 1000):
            raise ValueError('Unexpected forward training budget')
        for file, expected in result['artifacts'].items():
            if sha(directory/file) != expected:
                raise ValueError('Source artifact changed')
        for sampler, filename in [('ode', 'ode_samples.pt'), ('stochastic', 'stochastic_samples.pt')]:
            label = arm+'_'+sampler
            source = next(row for row in assessment['sources'] if row['arm'] == label)
            if source['samples_sha256'] != result['artifacts'][filename] or source['results_sha256'] != sha(path):
                raise ValueError('Assessment source does not match final result')
        rows.append({'arm': arm, 'new_oracle_evaluations': result['new_oracle_evaluations'],
            'forward_updates': result['forward_updates'], 'seconds': result['seconds'],
            'ode_step_check': result['ode_step_check'], 'ode_projection': result['ode_projection'],
            'stochastic_projection': result['stochastic_projection'], 'mean_energy_eV': result['mean_energy_eV'],
            'full_target_evaluations': result['full_target_evaluations'],
            'xtb': {s: xtb[arm+'_'+s] for s in ['ode', 'stochastic']},
            'overlap': {s: {'count': geometry[arm+'_'+s]['overlap_count'],
                            'attempted': geometry[arm+'_'+s]['particles']} for s in ['ode', 'stochastic']}})
        sources.append({'arm': arm, 'result': str(path.resolve()), 'sha256': sha(path)})
    stopped = all(row['full_target_evaluations']['after_reverse_refit']['weights']['ess'] < 16 for row in rows[1:])
    report = {'complete': True, 'scope': __doc__, 'arms': rows, 'sources': sources,
        'assessment_sha256': sha(assessment_path), 'controls': common,
        'predeclared_stop': {'threshold': 'every student ESS<16 of256 after reverse refit',
            'triggered': stopped, 'action': 'do not extend this frozen-pool recipe' if stopped else 'only qualifies for further independent evaluation'},
        'costs': {'new_student_and_source_oracle_queries': sum(row['new_oracle_evaluations'] for row in rows),
            'teacher_oracle_queries': 4096, 'inherited_forward_run_queries': 25024,
            'inherited_backward_refit_queries': 256, 'student_smoke_queries': 128,
            'xtb_attempts': 256, 'accounting_note': 'Historical method development and earlier smokes are additional; shared costs are not charged four times.'},
        'limitations': ['One development condition and one training seed; no broad or replicated advantage.',
            'Stochastic work ESS does not certify unweighted ODE statistics or global mode coverage.',
            'Intermediate teacher ESS is not full-target ESS.',
            'No independent eight-atom target normalizer; no calibrated logZ claim.',
            'No validated AI novelty or submission readiness follows from this screen.']}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['predeclared_stop']))


if __name__ == '__main__':
    main()
