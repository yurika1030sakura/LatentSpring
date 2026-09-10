#!/usr/bin/env python3
"""Frozen learned-feature score calibration against random and typed radial bases."""
import argparse
import json
import math
from pathlib import Path
import time

import torch

from cfm_mol.neural_stein_features import neural_head_features, antithetic_divergence, typed_radial_features
from cfm_mol.nonequilibrium import centered_orthonormal_basis
from cfm_mol.proposal_score import ProposalEnergyCritic
from cfm_mol.stein_calibration import (fit_stein_moments, score_moments, radial_score_features,
    size_score_features, angular_score_features)
from evaluate_score_calibration import load_run, summarize
from molecular_tempered_pilot import sha, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--limit', type=int, help='Engineering subset of both panels; no qualification')
    args = p.parse_args()
    output = args.out/'results.json'
    if output.exists():
        raise FileExistsError(output)
    args.out.mkdir(parents=True, exist_ok=True); start = time.perf_counter()
    cal_dir = args.runs_root/'proposal_score_validation_v1'
    cal_report, calibration = load_run(cal_dir)
    dev_dir = args.runs_root/'score_calibration_fresh_v1'
    dev_report, development = load_run(dev_dir)
    replica_dir = args.runs_root/'proposal_score_replica_v1'
    replica_report, _ = load_run(replica_dir)
    condition = cal_report['condition']; source_hash = cal_report['source_checkpoint_sha256']
    for r in [dev_report, replica_report]:
        if r['condition'] != condition or r['source_checkpoint_sha256'] != source_hash:
            raise ValueError('Different source proposal or condition')
    if cal_report['heldout_parent_seed'] != 9107 or dev_report['heldout_parent_seed'] != 9117:
        raise ValueError('Unexpected calibration/development parent streams')
    if args.limit is not None and not 2 <= args.limit <= min(len(calibration['positions']), len(development['positions'])):
        raise ValueError('Invalid engineering limit')
    if args.limit is not None:
        for panel in [calibration, development]:
            for key in ['positions', 'intrinsic', 'epsilon']:
                panel[key] = panel[key][:args.limit]
    sigma = cal_report['terminal_noise_std']
    if dev_report['terminal_noise_std'] != sigma:
        raise ValueError('Terminal noise changed')
    n = len(condition['numbers']); dimension = 3*(n-1)
    basis = centered_orthonormal_basis(n, device=args.device)
    numbers = torch.tensor(condition['numbers'], dtype=torch.long, device=args.device)
    source_dir = Path(cal_report['configuration']['source_run'])
    source = json.loads((source_dir/'results.json').read_text())
    if sha(source_dir/'last.ckpt') != source_hash:
        raise ValueError('Forward checkpoint changed')
    electronic = torch.tensor([condition['charge']/5., (condition['spin_multiplicity']-1)/5.,
        math.log(source['configuration']['kT'])], dtype=torch.float64, device=args.device)
    gaussian_precision = cal_report['gaussian_baseline_precision']
    report = {'complete': False, 'scope': __doc__, 'engineering_only': args.limit is not None,
        'source_checkpoint_sha256': source_hash, 'condition': condition, 'terminal_noise_std': sigma,
        'calibration_parents': len(calibration['positions']), 'development_parents': len(development['positions']),
        'calibration_sha256': sha(cal_dir/'heldout.pt'), 'development_sha256': sha(dev_dir/'heldout.pt'),
        'ridge': .01, 'new_oracle_evaluations': 0, 'forward_model_updates': 0, 'arms': {},
        'limitations': ['This16384-parent panel is development data for this new variant, not a blind confirmation.',
            'Deep-feature score matching and linear solves have direct prior art; no identity is claimed novel.',
            'Paired Gaussian noise gives unbiased finite-sigma moments; parent pairs are the uncertainty units.',
            'Classical typed features may outnumber learned features; compare total computation, not just parameter counts.',
            'Finite moment checks are necessary only; no molecular actor or target-distribution result follows.']}
    write_json(output, report)
    # Scale is fitted; other probes are external to the learned feature basis.
    # Typed radial features can correlate with the radial assessment probes.
    x = development['positions'].double()
    radial, rd = radial_score_features(x)
    extra, ed = radial_score_features(x, (.75, 1.25, 1.75, 2.25, 2.75, 3.25), .35, include_scale=False)
    size, sd = size_score_features(x)
    angles, ad = [], []
    for begin in range(0, len(x), 64):
        v, d = angular_score_features(x[begin:begin+64].to(args.device))
        angles.append(v.cpu()); ad.append(d.cpu())
    probes = torch.cat([radial, extra, size, torch.cat(angles)], 1)
    probe_div = torch.cat([rd, ed, sd, torch.cat(ad)], 1)
    names = ['scale', 'radial1', 'radial2', 'radial3']+[f'radial{v}' for v in [.75, 1.25, 1.75, 2.25, 2.75, 3.25]]+['size3', 'size5', 'angle2', 'angle3']
    saved = {}; losses = {}; checkpoint_hashes = {}
    for arm in ['learned9101', 'learned9103', 'random9121', 'typed_rbf']:
        arm_start = time.perf_counter(); model = None
        if arm != 'typed_rbf':
            torch.manual_seed(9121)
            model = ProposalEnergyCritic().to(args.device).double()
            if arm.startswith('learned'):
                directory = cal_dir if arm == 'learned9101' else replica_dir
                checkpoint = directory/'critic.ckpt'; state = torch.load(str(checkpoint), map_location='cpu', weights_only=False)
                if state['source_checkpoint_sha256'] != source_hash or state['terminal_noise_std'] != sigma:
                    raise ValueError('Critic provenance mismatch')
                model.load_state_dict(state['critic_state_dict'], strict=True)
                checkpoint_hashes[arm] = sha(checkpoint)
            else:
                with torch.no_grad():
                    model.log_confinement.copy_(torch.tensor(math.log(math.expm1(gaussian_precision-1e-4)), dtype=torch.float64, device=args.device))
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        max_reconstruction_error = 0.

        def features(z):
            nonlocal max_reconstruction_error
            positions = torch.einsum('nk,bkd->bnd', basis, z.reshape(len(z), n-1, 3))
            if model is None:
                v, d, _ = typed_radial_features(positions, numbers)
                return v, -gaussian_precision*positions, gaussian_precision, d
            v, score, tail, error = neural_head_features(model, z, basis, numbers, electronic)
            max_reconstruction_error = max(max_reconstruction_error, error)
            return v, score, tail, None

        gram = None; residual_sum = None
        for begin in range(0, len(calibration['intrinsic']), 64):
            plus = calibration['intrinsic'][begin:begin+64].to(args.device)
            epsilon = calibration['epsilon'][begin:begin+64].to(args.device)
            minus = plus-2*sigma*epsilon
            fp, sp, tail, dp = features(plus); fm, sm, _, dm = features(minus)
            eps_cart = torch.einsum('nk,bkd->bnd', basis, epsilon.reshape(len(epsilon), n-1, 3))
            div = .5*(dp+dm) if dp is not None else antithetic_divergence(fp, fm, eps_cart, sigma)
            flatp, flatm = fp.flatten(2), fm.flatten(2)
            g = .5*(torch.einsum('bpd,bqd->pq', flatp, flatp)+torch.einsum('bpd,bqd->pq', flatm, flatm))
            r = (.5*((sp[:, None]*fp).sum((-1, -2))+(sm[:, None]*fm).sum((-1, -2)))+div).sum(0)
            gram = g if gram is None else gram+g
            residual_sum = r if residual_sum is None else residual_sum+r
        gram /= len(calibration['intrinsic']); residual_sum /= len(calibration['intrinsic'])
        correction = fit_stein_moments(gram, residual_sum, base_tail_precision=tail, ridge=.01)
        point_scores = []; base_point_scores = []; risk_rows = []; dsm_rows = []; base_dsm_rows = []
        for begin in range(0, len(development['intrinsic']), 64):
            plus = development['intrinsic'][begin:begin+64].to(args.device)
            epsilon = development['epsilon'][begin:begin+64].to(args.device)
            fp, sp, _, dp = features(plus); fm, sm, _, dm = features(plus-2*sigma*epsilon)
            eps_cart = torch.einsum('nk,bkd->bnd', basis, epsilon.reshape(len(epsilon), n-1, 3))
            div = .5*(dp+dm) if dp is not None else antithetic_divergence(fp, fm, eps_cart, sigma)
            cp = correction.apply(sp, fp); cm = correction.apply(sm, fm)
            residual = .5*((sp[:, None]*fp).sum((-1, -2))+(sm[:, None]*fm).sum((-1, -2)))+div
            change = .5*((cp-sp).square().sum((1, 2))+(cm-sm).square().sum((1, 2)))-2*(residual*correction.coefficients).sum(-1)
            risk_rows.append((change/dimension).cpu())
            dsm_rows.append((.5*((sigma*cp+eps_cart).square().sum((1, 2))+(sigma*cm-eps_cart).square().sum((1, 2)))/dimension).cpu())
            base_dsm_rows.append((.5*((sigma*sp+eps_cart).square().sum((1, 2))+(sigma*sm-eps_cart).square().sum((1, 2)))/dimension).cpu())
            point_scores.append(cp.cpu()); base_point_scores.append(sp.cpu())
        points = torch.cat(point_scores); base_points = torch.cat(base_point_scores)
        loss = torch.cat(dsm_rows); base_loss = torch.cat(base_dsm_rows); risks = torch.cat(risk_rows)
        moments = score_moments(points, probes, probe_div)
        base_moments = score_moments(base_points, probes, probe_div)
        scales = correction.feature_scales
        condition_number = float(torch.linalg.cond(gram/scales[:, None]/scales[None, :]+.01*torch.eye(len(scales), device=gram.device, dtype=gram.dtype)))
        report['arms'][arm] = {'feature_count': len(correction.coefficients), 'coefficients': correction.coefficients.tolist(),
            'feature_scales': scales.tolist(), 'gram_condition_number': condition_number,
            'tail_constraint_active': correction.tail_constraint_active, 'corrected_tail_precision': correction.corrected_tail_precision,
            'score_reconstruction_max_error': max_reconstruction_error, 'seconds': time.perf_counter()-arm_start,
            'stein_risk_change_per_dimension': summarize(risks), 'dsm_risk_change_per_dimension': summarize((loss-base_loss)/sigma**2),
            'paired_dsm': summarize(loss), 'base_paired_dsm': summarize(base_loss),
            'stein': {name: summarize(moments[:, j]) for j, name in enumerate(names)},
            'base_stein': {name: summarize(base_moments[:, j]) for j, name in enumerate(names)}}
        saved[arm] = {'scores': points, 'base_scores': base_points, 'risk_rows': risks,
            'dsm_rows': loss, 'base_dsm_rows': base_loss, 'stein_rows': moments,
            'gram': gram.cpu(), 'residual': residual_sum.cpu(), 'coefficients': correction.coefficients.cpu()}
        losses[arm] = loss
        write_json(output, report)
        print(json.dumps({'arm': arm, 'seconds': time.perf_counter()-arm_start,
            'passed_moments': sum(row['within_three_sem'] for row in report['arms'][arm]['stein'].values())}), flush=True)
        del model
    report['qualification'] = {}
    for arm in ['learned9101', 'learned9103']:
        differences = {other: summarize(losses[arm]-losses[other]) for other in ['random9121', 'typed_rbf']}
        gate = {'improves_risk': report['arms'][arm]['stein_risk_change_per_dimension']['negative_by_two_sem'],
            'all_moments_pass': all(row['within_three_sem'] for row in report['arms'][arm]['stein'].values()),
            'beats_both_baselines': all(row['negative_by_two_sem'] for row in differences.values()), 'paired_baseline_differences': differences}
        gate['passes_development_screen'] = gate['improves_risk'] and gate['all_moments_pass'] and gate['beats_both_baselines'] and args.limit is None
        report['qualification'][arm] = gate
    report['both_pass_development'] = all(row['passes_development_screen'] for row in report['qualification'].values())
    torch.save({'arms': saved, 'probe_names': names, 'condition': condition}, args.out/'comparison.pt')
    report.update(complete=True, comparison_sha256=sha(args.out/'comparison.pt'), critic_sha256=checkpoint_hashes, seconds=time.perf_counter()-start)
    write_json(output, report); print(json.dumps(report['qualification']), flush=True)


if __name__ == '__main__':
    main()
