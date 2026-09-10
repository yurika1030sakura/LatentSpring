#!/usr/bin/env python3
"""Test a fixed residual score calibration on new samples and unfitted probes."""
import argparse
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from cfm_mol.nonequilibrium import centered_orthonormal_basis
from cfm_mol.proposal_score import ProposalEnergyCritic
from cfm_mol.stein_calibration import (radial_score_features, size_score_features,
    angular_score_features, score_moments, fit_stein_correction)
from molecular_tempered_pilot import sha, write_json


def summarize(rows):
    rows = rows.detach().double().cpu()
    mean = float(rows.mean()); sem = float(rows.std()/math.sqrt(len(rows)))
    return {'count': len(rows), 'mean': mean, 'sem': sem,
        'within_three_sem': abs(mean) <= 3*sem, 'negative_by_two_sem': mean+2*sem < 0}


def load_run(directory):
    path = directory/'results.json'; report = json.loads(path.read_text())
    if not report['complete'] or report['new_oracle_evaluations'] or report['forward_model_updates']:
        raise ValueError('Require complete, frozen, oracle-free source')
    for file, expected in report['artifacts'].items():
        if sha(directory/file) != expected:
            raise ValueError('Source artifact changed')
    return report, torch.load(str(directory/'heldout.pt'), map_location='cpu', weights_only=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-root', type=Path, required=True)
    p.add_argument('--fresh-panel', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--limit', type=int, help='Engineering smoke only; never qualifies the method')
    args = p.parse_args()
    output = args.out/'results.json'
    if output.exists():
        raise FileExistsError(output)
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    fresh_report, fresh = load_run(args.fresh_panel)
    calibrations = {}
    for label, name in [('seed9101', 'proposal_score_validation_v1'), ('seed9103', 'proposal_score_replica_v1')]:
        directory = args.runs_root/name
        r, data = load_run(directory)
        if r['source_checkpoint_sha256'] != fresh_report['source_checkpoint_sha256'] or r['condition'] != fresh_report['condition']:
            raise ValueError('Proposal or physical condition differs')
        if r['heldout_parent_seed'] == fresh_report['heldout_parent_seed']:
            raise ValueError('Calibration and assessment parents overlap')
        calibrations[label] = (directory, r, data)
    if fresh_report['heldout_parent_seed'] != 9117 or fresh_report['evaluation_seed'] != 9118 or len(fresh['positions']) != 16384:
        raise ValueError('Require prespecified fresh16384-parent panel')
    first_data = calibrations['seed9101'][2]
    second_data = calibrations['seed9103'][2]
    for key in ['positions', 'intrinsic', 'epsilon']:
        torch.testing.assert_close(first_data[key], second_data[key], rtol=0, atol=0)
    count = len(fresh['positions']) if args.limit is None else args.limit
    if not 1 <= count <= len(fresh['positions']):
        raise ValueError('Invalid engineering subset')
    test_x = fresh['positions'][:count].double()
    condition = fresh_report['condition']; n = len(condition['numbers']); dimension = 3*(n-1)
    sigma = fresh_report['terminal_noise_std']
    basis = centered_orthonormal_basis(n, device=args.device)
    numbers = torch.tensor(condition['numbers'], dtype=torch.long, device=args.device)
    source = json.loads((Path(fresh_report['configuration']['source_run'])/'results.json').read_text())
    if sha(Path(fresh_report['configuration']['source_run'])/'last.ckpt') != fresh_report['source_checkpoint_sha256']:
        raise ValueError('Underlying forward checkpoint changed')
    electronic = torch.tensor([condition['charge']/5., (condition['spin_multiplicity']-1)/5., math.log(source['configuration']['kT'])],
        dtype=torch.float64, device=args.device)
    calibration_x = first_data['positions'].double()
    cal_features, cal_div = radial_score_features(calibration_x)
    test_features, test_div = radial_score_features(test_x)
    unseen_radial, unseen_radial_div = radial_score_features(test_x, (.75, 1.25, 1.75, 2.25, 2.75, 3.25), .35, include_scale=False)
    size, size_div = size_score_features(test_x)
    angle, angle_div = [], []
    report = {'complete': False, 'scope': __doc__, 'engineering_only': args.limit is not None,
        'condition': condition, 'assessment_count': count, 'calibration_count': len(calibration_x),
        'source_checkpoint_sha256': fresh_report['source_checkpoint_sha256'],
        'fresh_panel_sha256': sha(args.fresh_panel/'heldout.pt'),
        'calibration_sources': {label: {'results_sha256': sha(directory/'results.json'), 'samples_sha256': sha(directory/'heldout.pt'),
            'critic_sha256': sha(directory/'critic.ckpt')} for label, (directory, _, _) in calibrations.items()},
        'ridge': .01, 'new_oracle_evaluations': 0, 'forward_model_updates': 0,
        'limitations': ['The population projection identity does not guarantee finite-sample improvement.',
            'Calibration moments alone are not independent validation; unseen radial, size and angular probes are included.',
            'Finite moment checks and relative risk reduction do not certify the whole score or Boltzmann sampling.',
            'This is one condition with two frozen critics; no molecular actor update or ICLR novelty is established.']}
    write_json(output, report)
    for begin in range(0, count, 64):
        vector, divergence = angular_score_features(test_x[begin:begin+64].to(args.device))
        angle.append(vector.cpu()); angle_div.append(divergence.cpu())
        if (begin//64+1) % 32 == 0:
            print(json.dumps({'angular_rows': min(begin+64, count), 'seconds': time.perf_counter()-start}), flush=True)
    angle = torch.cat(angle); angle_div = torch.cat(angle_div)
    all_features = torch.cat([test_features, unseen_radial, size, angle], 1)
    all_div = torch.cat([test_div, unseen_radial_div, size_div, angle_div], 1)
    feature_names = ['fit_scale', 'fit_radial_1', 'fit_radial_2', 'fit_radial_3']
    feature_names += ['new_radial_'+str(v) for v in [.75, 1.25, 1.75, 2.25, 2.75, 3.25]]
    feature_names += ['new_size_3', 'new_size_5', 'new_angle_2', 'new_angle_3']
    base_scores, corrected_scores, corrections = {}, {}, {}
    for label, (directory, r, data) in calibrations.items():
        checkpoint = torch.load(str(directory/'critic.ckpt'), map_location='cpu', weights_only=False)
        if checkpoint['source_checkpoint_sha256'] != fresh_report['source_checkpoint_sha256'] or checkpoint['terminal_noise_std'] != sigma:
            raise ValueError('Critic represents another proposal')
        critic = ProposalEnergyCritic().to(args.device).double()
        critic.load_state_dict(checkpoint['critic_state_dict'], strict=True)
        for parameter in critic.parameters():
            parameter.requires_grad_(False)
        check = critic.score(data['intrinsic'][:8].to(args.device), basis, numbers, electronic).detach().cpu()
        torch.testing.assert_close(check, data['score_arms']['learned'][:8], rtol=1e-8, atol=1e-7)
        tail_precision = float(F.softplus(critic.log_confinement).detach())+1e-4
        calibration_score = torch.einsum('nk,bkd->bnd', basis.cpu(), data['score_arms']['learned'].reshape(len(calibration_x), n-1, 3))
        correction = fit_stein_correction(calibration_score, cal_features, cal_div, base_tail_precision=tail_precision)
        scores = []
        for begin in range(0, count, 64):
            scores.append(critic.score(fresh['intrinsic'][begin:min(begin+64, count)].to(args.device), basis, numbers, electronic).detach().cpu())
        base_scores[label] = torch.einsum('nk,bkd->bnd', basis.cpu(), torch.cat(scores).reshape(count, n-1, 3))
        corrected_scores[label] = correction.apply(base_scores[label], test_features)
        corrections[label] = correction
        del critic
    gaussian_precision = calibrations['seed9101'][1]['gaussian_baseline_precision']
    correction = fit_stein_correction(-gaussian_precision*calibration_x, cal_features, cal_div, base_tail_precision=gaussian_precision)
    base_scores['gaussian'] = -gaussian_precision*test_x
    corrected_scores['gaussian'] = correction.apply(base_scores['gaussian'], test_features)
    corrections['gaussian'] = correction
    scores = {label+'_base': value for label, value in base_scores.items()}
    scores.update({label+'_calibrated': value for label, value in corrected_scores.items()})
    noise = torch.einsum('nk,bkd->bnd', basis.cpu(), fresh['epsilon'][:count].reshape(count, n-1, 3))
    losses = {label: (sigma*value+noise).square().sum((1, 2))/dimension for label, value in scores.items()}
    moments = {label: score_moments(value, all_features, all_div) for label, value in scores.items()}
    report['stein'] = {label: {name: summarize(values[:, j]) for j, name in enumerate(feature_names)} for label, values in moments.items()}
    report['dsm'] = {label: summarize(value) for label, value in losses.items()}
    gram = torch.einsum('bpad,bqad->pq', cal_features, cal_features)/len(cal_features)
    feature_scales = gram.diagonal().clamp_min(1e-20).sqrt()
    normalized_gram = gram/feature_scales[:, None]/feature_scales[None, :]
    report['calibration_design_condition_number'] = float(torch.linalg.cond(normalized_gram+.01*torch.eye(len(feature_scales), dtype=gram.dtype)))
    report['corrections'] = {}
    report['risk_changes'] = {}
    for label, correction in corrections.items():
        report['corrections'][label] = {'coefficients': correction.coefficients.tolist(), 'feature_scales': correction.feature_scales.tolist(),
            'tail_constraint_active': correction.tail_constraint_active, 'corrected_tail_precision': correction.corrected_tail_precision}
        report['risk_changes'][label] = {
            'stein_estimate_per_dimension': summarize(correction.risk_difference_rows(base_scores[label], test_features, test_div)/dimension),
            'denoising_estimate_per_dimension': summarize((losses[label+'_calibrated']-losses[label+'_base'])/sigma**2)}
    report['neural_vs_calibrated_gaussian'] = {label: summarize(losses[label+'_calibrated']-losses['gaussian_calibrated']) for label in calibrations}
    gates = {}
    for label in calibrations:
        checks = report['stein'][label+'_calibrated']
        gates[label] = {'risk_improves': report['risk_changes'][label]['stein_estimate_per_dimension']['negative_by_two_sem'],
            'beats_calibrated_gaussian': report['neural_vs_calibrated_gaussian'][label]['negative_by_two_sem'],
            'all_fitted_moments_pass': all(checks[name]['within_three_sem'] for name in feature_names[:4]),
            'all_unseen_moments_pass': all(checks[name]['within_three_sem'] for name in feature_names[4:])}
        gates[label]['passed'] = all(gates[label].values()) and args.limit is None
    report['qualification'] = {'seeds': gates, 'both_pass': all(row['passed'] for row in gates.values()),
        'scope': 'Necessary calibration screen only; no actor or general sampling qualification.'}
    torch.save({'scores': scores, 'dsm_losses': losses, 'stein_moments': moments, 'feature_names': feature_names,
        'condition': condition, 'corrections': report['corrections']}, args.out/'comparison.pt')
    report.update(complete=True, comparison_sha256=sha(args.out/'comparison.pt'), seconds=time.perf_counter()-start)
    write_json(output, report); print(json.dumps(report['qualification']), flush=True)


if __name__ == '__main__':
    main()
