#!/usr/bin/env python3
"""Bounded, exact-reference test of a conditional molecular proposal candidate.

The target is a known radial Gaussian mixture on H, not a molecular potential.
Equal total analytic-target budgets include neural training. ESJD training is
standard methodology; no novelty or molecular benefit is inferred from this toy.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from cfm_mol.conditional_molecular_proposal import (ConditionalMolecularProposal,center,
    invariant_jump_squared,metropolis_transition)
from cfm_mol.mcmc_diagnostics import diagnose_chains


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,data):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--kind',choices=['nonlinear','affine','mala','rwm'],required=True)
    parser.add_argument('--replica',type=int,choices=[0,1],required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[2]
    protocol_path=root/'research/evidence/conditional_proposal_toy_protocol_v1.json'
    protocol=json.loads(protocol_path.read_text())
    if not protocol['frozen']:raise ValueError('Require a frozen toy protocol')
    args.out.mkdir(parents=True,exist_ok=True);output=args.out/'results.json'
    if output.exists():raise FileExistsError(output)
    n=protocol['atoms'];d=3*(n-1);chains=protocol['chains'];seed=args.replica
    sigma=torch.tensor(protocol['target_sigmas'],dtype=torch.float64)
    numbers=torch.full((n,),6,dtype=torch.long);electronic=torch.zeros(3,dtype=torch.float64)
    queried=0
    def components(x):
        return -.5*x.square().sum((1,2))[:,None]/sigma.square()-d*sigma.log()-.5*d*math.log(2*math.pi)-math.log(2)
    def target(x,with_score=False):
        nonlocal queried
        queried+=len(x);terms=components(x);value=torch.logsumexp(terms,-1)
        if with_score:
            precision=(terms.softmax(-1)/sigma.square()).sum(-1)
            return value,-x*precision[:,None,None]
        return value
    def draw(count,generator):
        labels=torch.randint(2,(count,),generator=generator)
        return center(torch.randn(count,n,3,dtype=torch.float64,generator=generator))*sigma[labels,None,None]
    torch.manual_seed(protocol['init_seed']+seed)
    model=None;training_generator=torch.Generator().manual_seed(protocol['training_seed']+seed)
    report=dict(complete=False,scope=__doc__,kind=args.kind,replica=seed,protocol_sha256=sha(protocol_path),
        history=[],scientific_submission_ready=False,molecular_oracle_queries=0,
        source_sha256={str(p.relative_to(root)):sha(p) for p in [Path(__file__).resolve(),
            root/'cfm_mol/conditional_molecular_proposal.py',root/'cfm_mol/mcmc_diagnostics.py']})
    write(output,report);start=time.perf_counter()
    try:
        if args.kind in ['nonlinear','affine']:
            model=ConditionalMolecularProposal(**protocol['model'],nonlinear=args.kind=='nonlinear').double()
            report['parameters']=sum(p.numel() for p in model.parameters())
            optimizer=torch.optim.Adam(model.parameters(),lr=protocol['learning_rate'])
            for step in range(protocol['training_steps']):
                x=draw(protocol['training_batch'],training_generator)
                noise=center(torch.randn(x.shape,dtype=x.dtype,generator=training_generator))
                y,forward=model.transform(x,noise,numbers,electronic)
                ratio=target(y)-target(x)+model.log_prob(x,y,numbers,electronic)-forward
                objective=-(ratio.clamp_max(0).exp()*invariant_jump_squared(x,y,numbers)).mean()
                if not torch.isfinite(objective):raise ValueError('Nonfinite training objective')
                optimizer.zero_grad();objective.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
                optimizer.step()
                if step==0 or (step+1)%20==0:
                    row=dict(step=step+1,objective=float(objective.detach()),gradient_norm=float(norm),target_queries=queried)
                    report['history'].append(row);write(output,report);print(json.dumps(row),flush=True)
            torch.save(dict(configuration=model.configuration,state_dict=model.state_dict()),args.out/'model.pt')
            checkpoint=torch.load(args.out/'model.pt',map_location='cpu',weights_only=False)
            restored=ConditionalMolecularProposal(**checkpoint['configuration']).double()
            restored.load_state_dict(checkpoint['state_dict'])
            verification_generator=torch.Generator().manual_seed(9959+seed)
            vx=draw(16,verification_generator);vn=center(torch.randn(vx.shape,dtype=vx.dtype,generator=verification_generator))
            with torch.no_grad():
                vy,vq=model.transform(vx,vn,numbers,electronic)
                ry,rq=restored.transform(vx,vn,numbers,electronic)
                torch.testing.assert_close(ry,vy,atol=1e-10,rtol=1e-10)
                torch.testing.assert_close(rq,vq,atol=1e-10,rtol=1e-10)
                torch.testing.assert_close(restored.log_prob(vy,vx,numbers,electronic),vq,atol=1e-8,rtol=1e-9)
                backward=model.log_prob(vx,vy,numbers,electronic)
                restored.inverse_iterations=64
                torch.testing.assert_close(restored.log_prob(vx,vy,numbers,electronic),backward,atol=1e-8,rtol=1e-9)
            report['trained_checkpoint_density_and_inverse_ladder_passed']=True
        training_queries=queried
        evaluation_steps=protocol['evaluation_steps'] if model is not None else protocol['baseline_steps']
        initial_generator=torch.Generator().manual_seed(protocol['evaluation_initial_seed']+seed)
        generator=torch.Generator().manual_seed(protocol['evaluation_transition_seed']+seed)
        x=draw(chains,initial_generator)
        if args.kind=='mala':value,cached_score=target(x,with_score=True)
        else:value=target(x)
        states=[x.clone()];accepted=[];jumps=[]
        evaluation_start=time.perf_counter()
        with torch.no_grad():
            for step in range(evaluation_steps):
                if model is not None:
                    x,value,info=metropolis_transition(model,x,value,target,numbers,electronic,generator=generator)
                    take=info['accepted'];jump=info['accepted_invariant_jump']
                else:
                    std=protocol[args.kind+'_std']
                    def clip(gradient):
                        norm=gradient.square().sum((1,2)).sqrt()
                        return gradient*(100/norm.clamp_min(100))[:,None,None]
                    mean=x+.5*std**2*clip(cached_score) if args.kind=='mala' else x
                    y=mean+std*center(torch.randn(x.shape,dtype=x.dtype,generator=generator))
                    if args.kind=='mala':new_value,new_score=target(y,with_score=True)
                    else:new_value=target(y)
                    ratio=new_value-value
                    if args.kind=='mala':
                        reverse=y+.5*std**2*clip(new_score)
                        ratio=ratio+((y-mean).square().sum((1,2))-(x-reverse).square().sum((1,2)))/(2*std**2)
                    take=torch.rand(chains,dtype=x.dtype,generator=generator).log()<ratio.clamp_max(0)
                    jump=invariant_jump_squared(x,y,numbers)*take
                    x=torch.where(take[:,None,None],y,x);value=torch.where(take,new_value,value)
                    if args.kind=='mala':cached_score=torch.where(take[:,None,None],new_score,cached_score)
                states.append(x.clone());accepted.append(take);jumps.append(jump)
        if queried!=protocol['total_analytic_target_queries_per_arm']:
            raise RuntimeError('Total training/evaluation target-query budget differs')
        states=torch.stack(states)
        flat=states[1:].reshape(-1,n,3)
        mode=components(flat).softmax(-1)[:,1].reshape(evaluation_steps,chains).T.numpy()
        radius=flat.square().sum((1,2)).reshape(evaluation_steps,chains).T.numpy()
        torch.save(dict(positions=states,accepted=torch.stack(accepted),invariant_jumps=torch.stack(jumps)),args.out/'chains.pt')
        report.update(complete=True,training_target_queries=training_queries,total_target_queries=queried,
            evaluation_steps=evaluation_steps,seconds=time.perf_counter()-start,
            evaluation_seconds=time.perf_counter()-evaluation_start,
            acceptance=float(torch.stack(accepted).double().mean()),
            accepted_invariant_jump=float(torch.stack(jumps).mean()),
            large_component_probability=diagnose_chains(mode),squared_radius=diagnose_chains(radius),
            exact_large_component_probability=.5,exact_mean_squared_radius=d*float(sigma.square().mean()),
            artifacts={p.name:sha(p) for p in args.out.glob('*.pt')},
            limitations=['Exact target iid states initialize training and evaluation; this is a controlled toy only.',
                'Equal analytic-target counts include neural training; runtime and retained-chain lengths differ.',
                'Accepted invariant jump is not a mixing or global spectral-gap certificate.'])
        write(output,report);print(json.dumps({k:report[k] for k in ['complete','kind','replica','total_target_queries','large_component_probability','squared_radius']}),flush=True)
    except Exception as exc:
        if model is not None:torch.save(dict(configuration=model.configuration,state_dict=model.state_dict()),args.out/'failed_model.pt')
        report.update(complete=False,failure=f'{type(exc).__name__}: {exc}',target_queries=queried)
        write(output,report);raise


if __name__=='__main__':main()
