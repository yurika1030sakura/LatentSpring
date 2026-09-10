#!/usr/bin/env python3
"""Exact-volume linear refinement of a frozen implicit molecular generator."""
import argparse
import json
import math
from pathlib import Path
import time

import torch

from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.linear_entropy_adapter import LinearEntropyAdapter, endpoint_kl_change
from cfm_mol.nonequilibrium import WeightedPaths
from cfm_mol.path_work import external_energy
from molecular_tempered_pilot import sha, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--kind', choices=['typed', 'scalar'], required=True)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--eval-count', type=int, default=256)
    p.add_argument('--device', default='cuda')
    p.add_argument('--selection-seed', type=int, default=9141)
    args = p.parse_args()
    if min(args.steps, args.batch) < 1 or not 2 <= args.eval_count <= 256:
        raise ValueError('Invalid experiment counts')
    output = args.out/'results.json'
    if output.exists():raise FileExistsError(output)
    args.out.mkdir(parents=True, exist_ok=True); start = time.perf_counter()
    source_dir = args.runs_root/'work300_annealed_joint_5846_1500_v1'
    teacher_dir = args.runs_root/'forward_work_teacher_5846_v1'
    evaluation_dir = args.runs_root/'forward_work_student_source_v1'
    source = json.loads((source_dir/'results.json').read_text())
    teacher_report = json.loads((teacher_dir/'results.json').read_text())
    evaluation_report = json.loads((evaluation_dir/'results.json').read_text())
    checkpoint_sha = sha(source_dir/'last.ckpt')
    if not all(r['complete'] for r in [source, teacher_report, evaluation_report]):raise ValueError('Require complete source runs')
    if any(r['source_checkpoint_sha256'] != checkpoint_sha for r in [teacher_report, evaluation_report]):raise ValueError('Base generator differs')
    if evaluation_report['forward_updates'] != 0 or not evaluation_report['forward_state_unchanged_during_reverse_refit']:
        raise ValueError('Assessment source is not the frozen base generator')
    train_path = teacher_dir/'teacher.pt'; eval_path = evaluation_dir/'stochastic_samples.pt'
    if sha(train_path) != teacher_report['teacher_sha256'] or sha(eval_path) != evaluation_report['artifacts']['stochastic_samples.pt']:
        raise ValueError('Sample files changed')
    train = torch.load(str(train_path), map_location='cpu', weights_only=False)
    evaluation = torch.load(str(eval_path), map_location='cpu', weights_only=False)
    condition = source['condition']; recipe = source['configuration']
    if train['condition'] != condition or evaluation['condition'] != condition:raise ValueError('Physical conditions differ')
    if teacher_report['configuration']['seed'] == evaluation_report['seeds']['stochastic_evaluation']:raise ValueError('Training and evaluation streams overlap')
    x_train = train['positions'].double(); x_eval = evaluation['positions'][:args.eval_count].double()
    if max(float(x_train.mean(1).abs().max()), float(x_eval.mean(1).abs().max())) > 1e-8:raise ValueError('Require COM-free source samples')
    adapter = LinearEntropyAdapter(condition['numbers'], kind=args.kind).to(args.device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=.01, weight_decay=0.)
    selection = torch.Generator().manual_seed(args.selection_seed)
    report = {'complete': False, 'scope': __doc__, 'kind': args.kind,
        'configuration': {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'condition': condition, 'source_checkpoint_sha256': checkpoint_sha, 'training_sha256': sha(train_path),
        'evaluation_sha256': sha(eval_path), 'oracle_sha256': source['oracle_sha256'],
        'kT_eV': recipe['kT'], 'restraint_eV_A2': recipe['restraint'], 'steps': args.steps, 'batch': args.batch,
        'seed': args.selection_seed, 'lr_start': .01, 'lr_end': .0001, 'lr_schedule': 'cosine', 'parameters': adapter.raw_weights.numel(), 'maximum_pair_weight': .25,
        'history': [], 'limitations': ['A known exact-entropy normalizing-flow refinement baseline, not a novelty claim.',
            'The pretrained base generator is frozen; no unqualified score critic is used.',
            'Independent paired energy-minus-log-volume changes estimate marginal reverse-KL change, not absolute KL.',
            'One condition and one adapter-training seed; no broad or replicated sampling claim.',
            'Inherited base-generator and sample-production costs remain additional.',
            'Importance weights may still degenerate even when the mean endpoint KL improves.']}
    write_json(output, report)
    root = Path(__file__).resolve().parents[2]; oracle_path = Path(recipe['oracle'])
    if sha(oracle_path) != source['oracle_sha256']:raise ValueError('Target potential changed')
    with EnergyOracle(Path(recipe['oracle_python']), root/'scripts/research/oracle_worker.py', oracle_path,
        numbers=condition['numbers'], charge=condition['charge'], spin_multiplicity=condition['spin_multiplicity'], batch_size=16) as oracle:
        initial_energy, _ = oracle.evaluate(x_eval)
        initial_error = float((initial_energy-evaluation['energy_eV'][:args.eval_count]).abs().max())
        if initial_error > .001:raise ValueError('Cached and current base energies disagree')
        report['base_energy_replay_max_error_eV'] = initial_error
        for step in range(args.steps):
            learning_rate = .0001+.5*(.01-.0001)*(1+math.cos(math.pi*step/max(1,args.steps-1)))
            for group in optimizer.param_groups:group['lr'] = learning_rate
            indices = torch.randint(len(x_train), (args.batch,), generator=selection)
            x = x_train[indices].to(args.device)
            optimizer.zero_grad(set_to_none=True)
            y, logdet = adapter(x)
            energy, force = oracle.evaluate(y.detach())
            if step == 0:
                error = float((energy-train['energy_eV'][indices]).abs().max())
                report['identity_training_energy_replay_max_error_eV'] = error
                if error > .001:raise ValueError('Identity adapter does not reproduce source energies')
            potential = external_energy(y, energy, force)+recipe['restraint']/2*y.square().sum((1, 2))
            loss = ((potential-source['energy_zero_eV'])/recipe['kT']).mean()-logdet
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite adapter objective')
            loss.backward(); gradient = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 10., error_if_nonfinite=True)
            optimizer.step()
            if step == 0 or (step+1) % 20 == 0:
                row = {'step': step+1, 'training_objective': float(loss.detach()), 'log_volume': float(logdet.detach()),
                    'gradient_norm': float(gradient), 'lr': learning_rate, 'seconds': time.perf_counter()-start}
                report['history'].append(row); write_json(output, report); print(json.dumps(row), flush=True)
        with torch.no_grad():
            final_positions, logdet = adapter(x_eval.to(args.device))
            final_positions = final_positions.cpu(); volume = float(logdet)
            transform, _, generator = adapter.matrices()
        final_energy, _ = oracle.evaluate(final_positions)
        change = endpoint_kl_change(initial_energy, final_energy, x_eval, final_positions,
            kT=recipe['kT'], restraint=recipe['restraint'], log_volume=volume)
        base_work = evaluation['work'][:args.eval_count]+(initial_energy-evaluation['energy_eV'][:args.eval_count])/recipe['kT']
        final_work = base_work+change
        report['oracle_evaluations'] = oracle.evaluated
    report['paired_endpoint_kl_change'] = {'mean': float(change.mean()), 'sem': float(change.std()/math.sqrt(len(change))),
        'samples': len(change), 'scope': 'Expected energy-minus-exact-log-volume difference equals marginal KL change under stated fixed-target assumptions.'}
    report['weights'] = {'base': WeightedPaths(x_eval, -base_work, {}).summary(),
        'adapted': WeightedPaths(final_positions, -final_work, {}).summary()}
    report['mean_energy_eV'] = {'base': float(initial_energy.mean()), 'adapted': float(final_energy.mean())}
    report['log_volume'] = volume; report['transform_eigenvalues'] = torch.linalg.eigvalsh(transform).cpu().tolist()
    report['maximum_COM_error_A'] = float(final_positions.mean(1).abs().max())
    torch.save({'positions': x_eval, 'energy_eV': initial_energy, 'work': base_work, 'condition': condition}, args.out/'base_samples.pt')
    torch.save({'positions': final_positions, 'energy_eV': final_energy, 'work': final_work,
        'paired_endpoint_kl_change': change, 'condition': condition}, args.out/'adapted_samples.pt')
    torch.save({'state_dict': adapter.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'kind': args.kind,
        'condition': condition, 'source_checkpoint_sha256': checkpoint_sha}, args.out/'adapter.ckpt')
    report.update(complete=True, seconds=time.perf_counter()-start,
        artifacts={name: sha(args.out/name) for name in ['base_samples.pt', 'adapted_samples.pt', 'adapter.ckpt']})
    write_json(output, report); print(json.dumps(report['paired_endpoint_kl_change']), flush=True)


if __name__ == '__main__':main()
