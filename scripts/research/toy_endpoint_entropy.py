#!/usr/bin/env python3
"""Known-answer scalar mechanism check; no molecular or novelty claim."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from cfm_mol.endpoint_entropy import endpoint_kl_surrogate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    sigma, target_variance, initial_a, steps, lr = .2, .25, 1.2, 200, .02
    rows = []
    for seed in range(9095, 9100):
        for arm in ['independent_reverse_joint_kl', 'exact_endpoint_score', 'fitted_endpoint_score', 'stale_endpoint_score']:
            a = torch.tensor(initial_a, dtype=torch.float64, requires_grad=True)
            generator = torch.Generator().manual_seed(seed)
            actor_generator = torch.Generator().manual_seed(seed+10000019)
            history = []
            fitted = 1/(initial_a**2+sigma**2)
            for step in range(steps):
                # A fitted linear DSM critic, using fresh generator samples and
                # actual terminal innovations; independent actor samples follow.
                z_train = torch.randn(8192, 1, dtype=torch.float64, generator=generator)
                eps_train = torch.randn(8192, 1, dtype=torch.float64, generator=generator)
                y_train = a.detach()*z_train+sigma*eps_train
                if arm == 'fitted_endpoint_score':
                    fitted = float((eps_train*y_train).mean()/(sigma*y_train.square().mean()))
                elif arm == 'exact_endpoint_score':
                    fitted = 1/(float(a.detach())**2+sigma**2)
                z = torch.randn(512, 1, dtype=torch.float64, generator=actor_generator)
                epsilon = torch.randn(512, 1, dtype=torch.float64, generator=actor_generator)
                y = a*z+sigma*epsilon
                if arm == 'independent_reverse_joint_kl':
                    # Q(z,y)=phi(z)N(y;az,sigma^2), L(z|y)=phi(z).
                    # Other joint-KL terms have zero a-gradient at fixed sigma.
                    objective = .5*y.square().mean()/target_variance
                else:
                    objective = endpoint_kl_surrogate(y, -fitted*y, -y/target_variance)
                gradient, = torch.autograd.grad(objective, a)
                with torch.no_grad():
                    a -= lr*gradient
                if (step+1) % 20 == 0:
                    history.append({'step': step+1, 'a': float(a.detach()), 'critic_coefficient': fitted})
            variance = float(a.detach())**2+sigma**2
            ratio = variance/target_variance
            rows.append({'seed': seed, 'arm': arm, 'variance': variance,
                'exact_endpoint_kl': .5*(ratio-1-float(torch.log(torch.tensor(ratio, dtype=torch.float64)))),
                'history': history})
    report = {'complete': True, 'scope': __doc__, 'rows': rows,
        'code_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).resolve(), Path(__file__).resolve().parents[2]/'cfm_mol/endpoint_entropy.py']},
        'configuration': {'sigma': sigma, 'target_variance': target_variance, 'initial_a': initial_a,
            'steps': steps, 'learning_rate': lr, 'actor_batch': 512, 'critic_batch': 8192, 'seeds': list(range(9095, 9100))},
        'analytic_optima': {'independent_reverse_joint_kl_variance': sigma**2,
            'endpoint_kl_variance': target_variance},
        'limitations': ['A deliberately misspecified fixed reverse model is not the trained molecular reverse family.',
            'The critic is fitted scalar Gaussian regression, not a validated neural molecular score.',
            'Score-difference updates and denoising regression have direct VSD/DMD prior art.',
            'No molecular inference or ICLR-level method contribution follows from this check.']}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    for arm in ['independent_reverse_joint_kl', 'exact_endpoint_score', 'fitted_endpoint_score', 'stale_endpoint_score']:
        selected = [r for r in rows if r['arm'] == arm]
        print(arm, 'variance_range', [min(r['variance'] for r in selected), max(r['variance'] for r in selected)],
              'max_KL', max(r['exact_endpoint_kl'] for r in selected))


if __name__ == '__main__':
    main()
