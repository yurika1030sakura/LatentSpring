#!/usr/bin/env python3
"""Nonlinear shear CNF: exact density, state-dependent trace noise, strong controls.

v_a(x)=(a sin(3 x2),0) has inverse (y1-a sin(3 y2),y2), zero
true divergence and exact normalized density. Its Hutchinson trace is
3 a cos(3 x2) xi1 xi2: common probes do not cancel across geometries.
Gaussian probes remove Rademacher orientation dependence, not squared-loss
bias. This is a mechanism test and is not a molecular sampling result.
"""
import argparse
import json
import math
from pathlib import Path
import torch


def one_run(seed,arm,steps):
    generator=torch.Generator().manual_seed(seed)
    a=torch.tensor(.3,dtype=torch.float64,requires_grad=True)
    optimizer=torch.optim.Adam([a],lr=.02)
    negatives=0
    for step in range(steps):
        x=torch.randn((64,2),generator=generator,dtype=torch.float64)*.35
        f=torch.sin(3*x[:,1]);fp=3*torch.cos(3*x[:,1])
        exact=-.5*((x[:,0]-a*f).square()+x[:,1].square())-math.log(2*math.pi)
        energy=.5*((x[:,0]-.6*f).square()+x[:,1].square())
        common='common' in arm
        shape=(2,1 if common else len(x),2)
        # Separate per-step geometry randomness from probe distribution choices.
        pg=torch.Generator().manual_seed(4000003*seed+step+9011)
        if 'gaussian' in arm:
            probes=torch.randn(shape,generator=pg,dtype=x.dtype)
        else:
            probes=(2*torch.randint(0,2,shape,generator=pg)-1).to(x)
        trace=a*fp[None,:]*(probes[:,:,0]*probes[:,:,1])
        if arm=='exact':trace=trace*0
        residual=exact[None,:]-trace+energy[None,:]
        residual=residual-residual.mean(-1,keepdim=True)
        if 'product' in arm:
            loss=(residual[0]*residual[1]).mean()
        else:
            loss=residual.mean(0).square().mean()
        negatives+=int(float(loss.detach())<0)
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if not torch.isfinite(a):raise FloatingPointError('Non-finite toy parameter')
    value=float(a.detach())
    return {'seed':seed,'arm':arm,'a':value,'target_a':.6,
            'exact_kl_q_model_to_target':.25*(value-.6)**2*(1-math.exp(-18)),
            'negative_loss_fraction':negatives/steps}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=3000)
    args=p.parse_args()
    arms=['exact','squared_rademacher','product_rademacher','squared_common_rademacher',
          'product_common_rademacher','squared_gaussian','product_gaussian','squared_common_gaussian']
    report={'definition':__doc__,'steps':args.steps,'target_a':.6,'off_policy_std':.35,
        'probe_count':2,'group_size':64,'rows':[],'complete':False}
    args.out.mkdir(parents=True,exist_ok=True)
    for seed in range(5):
        for arm in arms:
            row=one_run(seed,arm,args.steps);report['rows'].append(row)
            print(json.dumps(row),flush=True)
            (args.out/'toy.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    report['complete']=True
    (args.out/'toy.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
