#!/usr/bin/env python3
"""Controlled Gaussian CNF fitting: trace-noise bias with exact ground truth.

An analytic volume-preserving flow has A=a*[[0,1],[1,0]], or its rotated
diagonal equivalent. Independent Rademacher probes give a noisy trace in
the first frame and an exact trace in the second. This isolates stochastic
loss bias; it is not a molecular result or a claim of a new Gaussian model.
"""
import argparse
import json
import math
from pathlib import Path
import time
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=1500)
    p.add_argument('--seeds',type=int,default=5)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    target=0.6
    results=[]
    start=time.monotonic()
    for seed in range(args.seeds):
        generator=torch.Generator().manual_seed(100+seed)
        # Same data and minibatches across objectives; narrow off-policy data
        # deliberately expose noise that does not shrink with geometry spread.
        x=0.35*torch.randn((128,8,2),generator=generator,dtype=torch.float64)
        squares=x.square()
        energy=0.5*(squares[...,0]*math.exp(-2*target)+squares[...,1]*math.exp(2*target))
        batches=torch.randint(0,128,(args.steps,16),generator=generator)
        noise=(2*torch.randint(0,2,(args.steps,2,16,8),generator=generator)-1).double()
        for arm in ['exact','squared_2','product_2','common_2','rotated_squared_2']:
            a=torch.nn.Parameter(torch.tensor(0.,dtype=torch.float64))
            optimizer=torch.optim.Adam([a],lr=0.006)
            negative=0
            for step in range(args.steps):
                idx=batches[step]
                logq=-0.5*(squares[idx,...,0]*torch.exp(-2*a)+squares[idx,...,1]*torch.exp(2*a))
                r=logq+energy[idx]
                r=r-r.mean(-1,keepdim=True)
                if arm in ['exact','rotated_squared_2']:
                    loss=r.square().mean()
                else:
                    eps=noise[step]
                    if arm=='common_2':eps=eps[...,:1].expand_as(eps)
                    replicas=r[None,:,:]-2*a*eps
                    replicas=replicas-replicas.mean(-1,keepdim=True)
                    if arm=='product_2':loss=(replicas[0]*replicas[1]).mean()
                    else:loss=replicas.mean(0).square().mean()
                if not torch.isfinite(loss):raise FloatingPointError('Non-finite toy loss')
                negative += int(float(loss.detach())<0)
                optimizer.zero_grad();loss.backward();optimizer.step()
            value=float(a.detach());delta=value-target
            # Exact KL(q_a || q_target) for covariances with eigenvalues exp(+-2a).
            kl=math.cosh(2*delta)-1
            row={'seed':seed,'arm':arm,'a':value,'target_a':target,'exact_kl':kl,
                 'negative_loss_fraction':negative/args.steps}
            results.append(row);print(json.dumps(row),flush=True)
    report={'definition':__doc__,'target_a':target,'steps':args.steps,
        'data_std':0.35,'rows':results,'seconds':time.monotonic()-start,
        'limitations':['Analytic Gaussian mechanism example, not molecular evidence',
            'Common probes exactly cancel noise for this affine field; nonlinear molecular fields require their own comparison',
            'Trace-products and common random numbers are standard identities, not novelty claims']}
    (args.out/'toy.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
