#!/usr/bin/env python3
"""Known-target AIS, basin-mass controls and matched CFM distillation.

This reproduces established work reweighting, not a new sampling algorithm or
evidence of molecular benefit. Every seed and raw-particle control is retained.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from cfm_mol.nonequilibrium import normalized_weights, random_walk_ais


def log_initial(x):
    return -.5*(x/3).square().sum(-1)-2*math.log(3*math.sqrt(2*math.pi))


def log_target(x):
    means=x.new_tensor([[-3.,0.],[3.,0.]])
    components=-.5*((x[:,None,:]-means)/.5).square().sum(-1)-2*math.log(.5*math.sqrt(2*math.pi))
    return torch.logsumexp(components+x.new_tensor([.2,.8]).log(),dim=1)


def metrics(x, log_weights):
    w=normalized_weights(log_weights)
    return {'right_basin_probability':float(w@(x[:,0]>0).double()),
            'mean_x':float(w@x[:,0].double()),
            'second_moment_about_target_mean_x':float(w@(x[:,0].double()-1.8).square()),
            'mean_nearest_mode_squared_distance':float(w@((x[:,0].abs()-3).square()+x[:,1].square()).double())}


def distill(endpoints, weights, seed, steps, eval_count):
    torch.manual_seed(30000+seed)
    model=nn.Sequential(nn.Linear(3,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,2))
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    generator=torch.Generator().manual_seed(50000+seed)
    endpoints=endpoints.float()
    start=time.perf_counter()
    for _ in range(steps):
        # Resampling endpoints realizes the fixed weighted empirical teacher.
        idx=torch.multinomial(weights,256,replacement=True,generator=generator)
        x1=endpoints[idx]
        x0=torch.randn((256,2),generator=generator)*3
        t=torch.rand((256,1),generator=generator)
        prediction=model(torch.cat([(1-t)*x0+t*x1,t],dim=1))
        loss=(prediction-(x1-x0)).square().mean()
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if not torch.isfinite(loss):raise FloatingPointError('Non-finite distillation loss')
    with torch.no_grad():
        g=torch.Generator().manual_seed(70000+seed)
        x=torch.randn((eval_count,2),generator=g)*3
        dt=1/128
        for i in range(128):
            t=x.new_full((len(x),1),i*dt)
            v=model(torch.cat([x,t],dim=1))
            vm=model(torch.cat([x+.5*dt*v,t+.5*dt],dim=1))
            x=x+dt*vm
    return {'metrics':metrics(x,torch.zeros(len(x))), 'seconds':time.perf_counter()-start,
            'steps':steps,'training_batch_size':256,'sampler':'midpoint128',
            'last_training_loss':float(loss.detach())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--particles',type=int,default=8192)
    p.add_argument('--seeds',type=int,default=5)
    p.add_argument('--ais-steps',type=int,default=16)
    p.add_argument('--fm-steps',type=int,default=1000)
    args=p.parse_args()
    if min(args.particles,args.seeds,args.ais_steps)<1 or args.fm_steps<0:raise ValueError('Invalid experiment size')
    args.out.mkdir(parents=True,exist_ok=True)
    output=args.out/'results.json'
    if output.exists():raise FileExistsError(output)
    root=Path(__file__).resolve().parents[2]
    paths=[Path(__file__).resolve(),root/'cfm_mol/nonequilibrium.py']
    report={'complete':False,'scope':__doc__,'target':{'dimension':2,'right_basin_probability':.8,
        'mean_x':1.8,'variance_x':6.01,'mean_nearest_mode_squared_distance':.5,'log_Z':0},
        'protocol':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'source_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        'torch':torch.__version__,'rows':[],
        'exponential_noise_counterexample':{'true_right_mass':.8,'left_log_noise_variance':0,
            'right_log_noise_variance':2,'limiting_mass_after_naive_noisy_reweighting':.8*math.e/(.2+.8*math.e),
            'reason':'E[exp(-epsilon)|x]=exp(variance(x)/2) for zero-mean Gaussian log noise'}}
    for seed in range(args.seeds):
        g=torch.Generator().manual_seed(9100+seed)
        x0=torch.randn((args.particles,2),generator=g,dtype=torch.float64)*3
        start=time.perf_counter()
        result=random_walk_ais(x0,log_initial,log_target,torch.linspace(0,1,args.ais_steps+1,dtype=torch.float64),.8,g)
        row={'seed':seed,'ais':result.summary(),'ais_seconds':time.perf_counter()-start,
             'weighted_teacher':metrics(result.positions,result.log_weights),
             'unweighted_same_paths':metrics(result.positions,torch.zeros(args.particles)),
             # Deliberately incorrect: missing the path/proposal factor.
             'terminal_energy_only':metrics(result.positions,log_target(result.positions)),
             'distilled':{}}
        if args.fm_steps:
            for name,w in [('weighted',normalized_weights(result.log_weights)),
                           ('unweighted',torch.ones(args.particles,dtype=torch.float64)/args.particles)]:
                row['distilled'][name]=distill(result.positions,w,seed,args.fm_steps,args.particles)
        report['rows'].append(row)
        output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps(row),flush=True)
    arms=['weighted_teacher','unweighted_same_paths','terminal_energy_only']
    report['aggregate']={name:{'mean_absolute_right_basin_error':float(np.mean([
        abs(row[name]['right_basin_probability']-.8) for row in report['rows']])),
        'mean_right_basin_probability':float(np.mean([row[name]['right_basin_probability'] for row in report['rows']]))} for name in arms}
    if args.fm_steps:
        for name in ['weighted','unweighted']:
            report['aggregate']['distilled_'+name]={'mean_absolute_right_basin_error':float(np.mean([
                abs(row['distilled'][name]['metrics']['right_basin_probability']-.8) for row in report['rows']])),
                'mean_right_basin_probability':float(np.mean([row['distilled'][name]['metrics']['right_basin_probability'] for row in report['rows']]))}
    for name in arms:
        report['aggregate'][name]['mean_nearest_mode_squared_distance']=float(np.mean([
            row[name]['mean_nearest_mode_squared_distance'] for row in report['rows']]))
    if args.fm_steps:
        for name in ['weighted','unweighted']:
            report['aggregate']['distilled_'+name]['mean_nearest_mode_squared_distance']=float(np.mean([
                row['distilled'][name]['metrics']['mean_nearest_mode_squared_distance'] for row in report['rows']]))
    report['complete']=True
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report['aggregate']),flush=True)


if __name__=='__main__':main()
