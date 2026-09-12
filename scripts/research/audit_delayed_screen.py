#!/usr/bin/env python3
"""Replay delayed-screen metrics and independently evaluate both orientations."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem

from cfm_mol.delayed_acceptance import DelayedAcceptanceScreen,screen_edge_values
from scripts.research.train_delayed_screen import evaluate,load_data,population_weights
from scripts.research.audit_action_geometry import independent_action_logp
from scripts.research.audit_masked_angular import equal,sha


def independent_factor(model,row,reverse=False):
    x,y,b,new=[row[k].numpy() for k in ['x','y','bonds','new_bonds']]
    action,inverse=row['action'],row['inverse_action']
    proposal=float(row['log_behavior_reverse']-row['log_behavior_forward']+row['action_log_ratio'])
    if reverse:x,y,b,new,action,inverse,proposal=y,x,new,b,inverse,action,-proposal
    z=row['numbers'].numpy();periodic=Chem.GetPeriodicTable()
    radii=np.array([periodic.GetRcovalent(int(v)) for v in z]);n=len(z)
    def features(p,graph):
        strain=repulsion=0.
        for i in range(n):
            for j in range(i+1,n):
                scaled=np.linalg.norm(p[i]-p[j])/(radii[i]+radii[j])
                if graph[i,j]>0:strain+=(scaled-1)**2*graph[i,j]
                else:repulsion+=scaled**-12
        return np.array([strain,repulsion,((p-p.mean(0))**2).sum()])
    f=(features(x,b)-features(y,new))*np.array([10.,1.,model.restraint/2])/float(row['electronic'][2])
    f=np.append(f,proposal);raw=float(f@model.coefficients.detach().numpy())
    if model.encoder is not None:
        values=[]
        for pos,graph,act in [(x,b,action),(y,new,inverse)]:
            actions,scores=independent_action_logp(model.encoder,torch.tensor(pos),torch.tensor(graph),
                row['numbers'],row['electronic'],raw_scores=True)
            values.append(float(scores[actions.index(tuple(act))]))
        raw+=values[0]-values[1]
    return float(model.log_factor_bound*np.tanh(raw/model.log_factor_bound))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','run','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--variant',choices=['linear','neural'],required=True)
    p.add_argument('--replica',choices=[0,1],type=int,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[2]
    pp=root/'research/evidence/delayed_screen_training_protocol_v1.json';protocol=json.loads(pp.read_text())
    fit,validation=load_data(root,a.project,protocol);run=a.run/a.variant/f'replica_{a.replica}'
    report=json.loads((run/'results.json').read_text());assert report['complete'] and report['protocol_sha256']==sha(pp)
    assert report['steps']==protocol['steps'] and report['seed']==protocol['seeds'][a.replica] and report['variant']==a.variant
    assert report['replica']==a.replica and report['data_sha256']==protocol['data_sha256']
    assert sha(run/'model.pt')==report['model_sha256']
    saved=torch.load(run/'model.pt',map_location='cpu',weights_only=False)
    model=DelayedAcceptanceScreen(a.variant,**protocol['model']).double()
    assert saved['configuration']==model.configuration and saved['protocol_sha256']==sha(pp) and saved['data_sha256']==protocol['data_sha256']
    assert saved['seed']==protocol['seeds'][a.replica] and saved['replica']==a.replica and saved['steps']==protocol['steps']
    model.load_state_dict(saved['state_dict'])
    for role,rows in [('fit',fit),('validation',validation)]:equal(evaluate(model,rows),report['final_'+role])
    for name in ['zero','physical']:
        baseline=DelayedAcceptanceScreen(name,**protocol['model']).double()
        for role,rows in [('fit',fit),('validation',validation)]:equal(evaluate(baseline,rows),report[f'{name}_{role}'])
    maximum=0.;count=0
    with torch.no_grad():
        for row in fit+validation:
            if not row['valid']:continue
            factor=independent_factor(model,row);reverse=independent_factor(model,row,True)
            assert abs(factor+reverse)<1e-9 and abs(factor)<=model.log_factor_bound+1e-10
            u,c,g,s,accept=screen_edge_values(model,row)
            ratio=float(row['target_log_ratio']+row['log_behavior_reverse']-row['log_behavior_forward']+row['action_log_ratio'])
            total=min(0.,factor)+min(0.,ratio-factor)
            assert total<=min(0.,ratio)+1e-12
            assert abs(total-(min(0.,reverse)+min(0.,-ratio-reverse))-ratio)<1e-9
            expected_accept=math.exp(total)
            errors=[abs(float(s)-factor),abs(float(u)-float(row['reward_eV'])*expected_accept),
                abs(float(c)-2*math.exp(min(0.,factor))),abs(float(accept)-expected_accept)]
            maximum=max(maximum,max(errors));assert max(errors)<1e-8;count+=1
    assert count==1597
    weights=population_weights(fit);cost=sum(float(w)*2*r['valid'] for w,r in zip(weights,fit))
    rate=sum(float(w)*r.get('baseline_expected_utility_eV',0.) for w,r in zip(weights,fit))/cost
    assert abs(cost-report['fixed_baseline_query_cost'])<1e-12 and abs(rate-report['fixed_baseline_utility_per_query_eV'])<1e-12
    # Pick sensitive finite-difference checks from the first scored edge per FIT
    # parent; this choice is only for numerical validation, never for fitting.
    first={}
    for row in fit:
        if row['valid']:first.setdefault((row['index'],row['parent']),row)
    heads=[('linear',model.coefficients)]
    if model.encoder is not None:heads.append(('neural',model.encoder.action_head[-1].weight))
    def loss(row):
        u,c,_,factor,_=screen_edge_values(model,row)
        return -(u-rate*c)/cost+protocol['factor_penalty_eV']*(factor/model.log_factor_bound).square()
    best={}
    for row in first.values():
        gradients=torch.autograd.grad(loss(row),[w for name,w in heads])
        for (name,w),g in zip(heads,gradients):
            magnitude=float(g.abs().max())
            if name not in best or magnitude>best[name][0]:best[name]=(magnitude,row,g)
    fd={}
    for name,w in heads:
        magnitude,row,g=best[name];flat=int(g.abs().argmax());h=1e-5
        assert magnitude>1e-12,'No sensitive trained-head gradient in declared FIT audit panel'
        with torch.no_grad():w.flatten()[flat]+=h
        high=float(loss(row))
        with torch.no_grad():w.flatten()[flat]-=2*h
        low=float(loss(row))
        with torch.no_grad():w.flatten()[flat]+=h
        error=abs((high-low)/(2*h)-float(g.flatten()[flat]));assert error<1e-6+1e-5*magnitude
        fd[name]=dict(error=error,gradient=float(g.flatten()[flat]),index=row['index'],parent=row['parent'],step=row['step'])
    result=dict(complete=True,variant=a.variant,replica=a.replica,protocol_sha256=sha(pp),data_sha256=protocol['data_sha256'],
        results_sha256=sha(run/'results.json'),model_sha256=sha(run/'model.pt'),all1671_attempts_retained=True,
        split_membership_checked=True,all_metrics_replayed=True,fixed_controls_replayed=True,
        independent_supported_pairs=count,maximum_independent_error=maximum,reverse_factor_and_balance_verified=True,
        trained_objective_gradients=fd,new_physical_queries=0,actual_queries_saved=0,
        optimizer_trajectory_replayed=False,scientific_submission_ready=False)
    if a.out.exists():raise FileExistsError(a.out)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
