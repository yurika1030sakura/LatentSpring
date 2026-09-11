#!/usr/bin/env python3
"""Longer frozen-kernel check, including a prespecified local/global mixture.

No parameters are updated. All methods receive the same total analytic-target
budget including the inherited neural-training cost. This remains a toy study.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from cfm_mol.conditional_molecular_proposal import ConditionalMolecularProposal,center
from cfm_mol.mcmc_diagnostics import diagnose_chains


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--case',choices=['nonlinear_global','affine_global','nonlinear_hybrid','affine_hybrid','mala','rwm'],required=True)
    parser.add_argument('--replica',type=int,choices=[0,1],required=True);args=parser.parse_args()
    root=Path(__file__).resolve().parents[2];protocol_path=root/'research/evidence/frozen_proposal_long_protocol_v1.json'
    protocol=json.loads(protocol_path.read_text())
    if not protocol['frozen']:raise ValueError('Require a frozen evaluation protocol')
    args.out.mkdir(parents=True,exist_ok=True);output=args.out/'results.json'
    if output.exists():raise FileExistsError(output)
    sigma=torch.tensor(protocol['target_sigmas'],dtype=torch.float64);n=4;d=9;m=protocol['chains']
    numbers=torch.full((n,),6,dtype=torch.long);electronic=torch.zeros(3,dtype=torch.float64)
    queries=0
    def terms(x):return -.5*x.square().sum((1,2))[:,None]/sigma.square()-d*sigma.log()-.5*d*math.log(2*math.pi)-math.log(2)
    def target(x):
        nonlocal queries
        queries+=len(x);t=terms(x)
        return t.logsumexp(-1),-x*(t.softmax(-1)/sigma.square()).sum(-1)[:,None,None]
    def clipped(score):
        norm=score.square().sum((1,2)).sqrt()
        return score*(100/norm.clamp_min(100))[:,None,None]
    learned=args.case.startswith(('nonlinear','affine'));model=None;training_cost=0
    report=dict(complete=False,scope=__doc__,case=args.case,replica=args.replica,
        protocol_sha256=sha(protocol_path),scientific_submission_ready=False,molecular_oracle_queries=0)
    if learned:
        kind=args.case.split('_')[0]
        path=args.runs_root/f'conditional_proposal_spectral_v1/{kind}_s{args.replica}'
        trained=json.loads((path/'results.json').read_text())
        if not trained['complete'] or trained['privileged_mode_feature'] or not trained['trained_checkpoint_density_and_inverse_ladder_passed']:
            raise ValueError('Require qualified label-free trained proposal')
        if sha(path/'model.pt')!=trained['artifacts']['model.pt']:raise ValueError('Changed trained checkpoint')
        saved=torch.load(path/'model.pt',map_location='cpu',weights_only=False)
        model=ConditionalMolecularProposal(**saved['configuration']).double().eval();model.load_state_dict(saved['state_dict'])
        training_cost=trained['training_target_queries']
        report.update(training_results_sha256=sha(path/'results.json'),model_sha256=sha(path/'model.pt'),parameters=trained['parameters'])
    steps=protocol['learned_steps'] if learned else protocol['baseline_steps']
    if training_cost+m*(1+steps)!=protocol['total_target_budget']:raise ValueError('Budget including training differs')
    initial_generator=torch.Generator().manual_seed(protocol['initial_seed']+args.replica)
    generator=torch.Generator().manual_seed(protocol['transition_seed']+args.replica)
    choices=torch.Generator().manual_seed(protocol['mixture_seed']+args.replica)
    label=torch.randint(2,(m,),generator=initial_generator)
    x=center(torch.randn(m,n,3,dtype=torch.float64,generator=initial_generator))*sigma[label,None,None]
    value,score=target(x);states=[x.clone()];acceptance=[];neural_counts=[]
    output.write_text(json.dumps(report,indent=2)+'\n');start=time.perf_counter()
    try:
        with torch.no_grad():
            for step in range(steps):
                use_neural=torch.zeros(m,dtype=torch.bool)
                if learned:
                    use_neural=(torch.rand(m,generator=choices)<protocol['hybrid_neural_probability']
                                if args.case.endswith('hybrid') else torch.ones(m,dtype=torch.bool))
                all_take=torch.zeros(m,dtype=torch.bool)
                for neural,mask in [(True,use_neural),(False,~use_neural)]:
                    if not mask.any():continue
                    current=x[mask];old=value[mask];old_score=score[mask]
                    noise=center(torch.randn(current.shape,dtype=current.dtype,generator=generator))
                    if neural:
                        y,forward=model.transform(current,noise,numbers,electronic)
                        proposed,new_score=target(y)
                        reverse=model.log_prob(current,y,numbers,electronic)
                        ratio=proposed-old+reverse-forward
                    else:
                        std=protocol['rwm_std'] if args.case=='rwm' else protocol['mala_std']
                        mean=current if args.case=='rwm' else current+.5*std**2*clipped(old_score)
                        y=mean+std*noise;proposed,new_score=target(y);ratio=proposed-old
                        if args.case!='rwm':
                            reverse=y+.5*std**2*clipped(new_score)
                            ratio=ratio+((y-mean).square().sum((1,2))-(current-reverse).square().sum((1,2)))/(2*std**2)
                    if not torch.isfinite(ratio).all():raise ValueError('Nonfinite evaluation acceptance ratio')
                    take=torch.rand(len(y),dtype=y.dtype,generator=generator).log()<ratio.clamp_max(0)
                    x[mask]=torch.where(take[:,None,None],y,current)
                    value[mask]=torch.where(take,proposed,old);score[mask]=torch.where(take[:,None,None],new_score,old_score)
                    all_take[mask]=take
                states.append(x.clone());acceptance.append(all_take);neural_counts.append(int(use_neural.sum()))
                if (step+1)%512==0:print(json.dumps(dict(step=step+1,target_queries=queries)),flush=True)
        if queries+training_cost!=protocol['total_target_budget']:raise RuntimeError('Measured target budget differs')
        states=torch.stack(states);flat=states[1:].reshape(-1,n,3)
        mode=terms(flat).softmax(-1)[:,1].reshape(steps,m).T
        radius=flat.square().sum((1,2)).reshape(steps,m).T
        torch.save(dict(positions=states,accepted=torch.stack(acceptance),neural_proposals_per_step=neural_counts),args.out/'chains.pt')
        report.update(complete=True,seconds=time.perf_counter()-start,steps=steps,training_target_queries=training_cost,
            new_evaluation_target_queries=queries,total_target_budget=queries+training_cost,
            acceptance=float(torch.stack(acceptance).double().mean()),neural_proposals=sum(neural_counts),
            large_component_probability=diagnose_chains(mode.numpy()),squared_radius=diagnose_chains(radius.numpy()),
            artifacts={'chains.pt':sha(args.out/'chains.pt')},
            limitations=['Frozen trained models and fresh independent evaluation streams; no retraining.',
                'Local/global mixture probability is fixed and selected independently per chain.',
                'Training is inherited and counted per method, not performed again by this evaluation.',
                'This known Gaussian-mixture target is not molecular sampling validation.'])
        output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report['large_component_probability']),flush=True)
    except Exception as exc:
        report.update(complete=False,failure=f'{type(exc).__name__}: {exc}',new_evaluation_target_queries=queries)
        output.write_text(json.dumps(report,indent=2)+'\n');raise


if __name__=='__main__':main()
