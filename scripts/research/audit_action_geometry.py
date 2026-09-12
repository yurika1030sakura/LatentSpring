#!/usr/bin/env python3
"""Replay bounded action/geometry training and independently check complete densities."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch

from cfm_mol.bounded_action_geometry import ActionGeometryGuide, edge_values, full_observed_densities
from cfm_mol.joint_arc_geometry import observed_joint_arc_density
from scripts.research.train_action_geometry import evaluate, population_weights
from scripts.research.audit_joint_arc_support import independent_arc_q
from scripts.research.audit_masked_angular import equal, sha


def independent_action_logp(model, x, bonds, numbers, electronic):
    """NumPy legal enumeration and complete neural action probabilities."""
    s={k:v.detach().cpu().numpy() for k,v in model.state_dict().items()}
    x,b,z,e=[v.detach().cpu().numpy() for v in (x,bonds,numbers,electronic)]
    n=len(x);adj=b>0;degree=adj.sum(1)
    leaves=[i for i in range(n) if z[i] in {1,9,17,35,53} and degree[i]==1]
    actions=[]
    for pos,i in enumerate(leaves):
        k=int(np.flatnonzero(adj[i])[0])
        for j in leaves[pos+1:]:
            l=int(np.flatnonzero(adj[j])[0])
            if z[i]!=z[j] and k!=l and k not in (i,j) and l not in (i,j) and b[i,k]==b[j,l]==1:
                actions.append((i,j,k,l))
    def silu(v):return v*np.exp(-np.logaddexp(0.,-v))
    def mlp(prefix,v,last_silu=False):
        for layer in [0,2]:
            v=v@s[f'{prefix}.{layer}.weight'].T+s[f'{prefix}.{layer}.bias']
            if layer==0 or last_silu:v=silu(v)
        return v
    distance=np.sqrt(np.sum((x[:,None]-x[None,:])**2,axis=-1)+1e-8)
    radial=np.exp(-.5*((distance[...,None]-s['centers'])/.4)**2)
    nodes=mlp('elements',s['atomic_features'][z])+mlp('state',e)
    for layer in range(2):
        a=np.broadcast_to(nodes[:,None],(n,n,nodes.shape[-1]))
        other=np.broadcast_to(nodes[None,:],a.shape)
        pair=mlp(f'messages.{layer}',np.concatenate([a+other,(a-other)**2,radial,b[...,None]/3],axis=-1),True)
        aggregate=np.sum(pair*(1-np.eye(n))[...,None],axis=1)/n
        nodes=nodes+mlp(f'updates.{layer}',np.concatenate([nodes,aggregate],axis=-1))
    if not actions:return actions,np.empty(0)
    raw=[]
    for i,j,k,l in actions:
        d1,d2,dij=distance[i,k],distance[j,l],distance[i,j]
        f=np.concatenate([nodes[i],nodes[j],nodes[k],nodes[l],[d1,d2,dij]])
        r=np.concatenate([nodes[j],nodes[i],nodes[l],nodes[k],[d2,d1,dij]])
        raw.append(float((mlp('action_head',f)+mlp('action_head',r))[0]/2))
    logits=model.logit_bound*np.tanh(np.asarray(raw)/model.logit_bound)
    return actions,logits-np.logaddexp.reduce(logits)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','run','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--variant',choices=['action','geometry','joint'],required=True)
    p.add_argument('--replica',choices=[0,1],type=int,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[2]
    pp=root/'research/evidence/action_geometry_training_protocol_v1.json';protocol=json.loads(pp.read_text())
    dp=a.project/protocol['data_run'];header=json.loads((dp/'results.json').read_text())
    assert header['complete'] and sha(dp/'results.json')==protocol['data_results_sha256']
    assert sha(dp/'data.pt')==header['data_sha256']==protocol['data_sha256']
    data=torch.load(dp/'data.pt',map_location='cpu',weights_only=False)
    dpp=root/'research/evidence/accepted_utility_data_protocol_v1.json';dprot=json.loads(dpp.read_text())
    assert header['protocol_sha256']==sha(dpp)
    source=json.loads((root/dprot['source_protocol']).read_text())
    for index in [1,2,3,5]:
        ids=source['parent_ids_by_condition'][str(index)]
        order=sorted(ids,key=lambda pid:hashlib.sha256(f"{dprot['split_seed']}|{index}|{pid}".encode()).hexdigest())
        assert dprot['splits'][str(index)]==dict(fit_parent_ids=sorted(order[3:]),withheld_parent_ids=sorted(order[:3]))
    for row in data:
        split=dprot['splits'][str(row['index'])]
        assert row['parent'] in split['fit_parent_ids']+split['withheld_parent_ids']
        assert row['role']==('fit' if row['parent'] in split['fit_parent_ids'] else 'withheld_parent')
    assert len(data)==1671 and sum(r['valid'] for r in data)==1597
    fit=[r for r in data if r['role']=='fit'];validation=[r for r in data if r['role']=='withheld_parent']
    run=a.run/a.variant/f'replica_{a.replica}';report=json.loads((run/'results.json').read_text())
    assert report['complete'] and report['steps']==protocol['steps'] and report['protocol_sha256']==sha(pp)
    assert report['variant']==a.variant and report['replica']==a.replica and report['seed']==protocol['seeds'][a.replica]
    assert report['data_sha256']==sha(dp/'data.pt') and sha(run/'model.pt')==report['model_sha256']
    saved=torch.load(run/'model.pt',map_location='cpu',weights_only=False)
    model=ActionGeometryGuide(a.variant,**protocol['models'][a.variant]).double()
    assert saved['configuration']==model.configuration and saved['protocol_sha256']==sha(pp)
    assert saved['steps']==protocol['steps'] and saved['score_semantics']=='bounded_conditional_action_and_geometry'
    model.load_state_dict(saved['state_dict'])
    for role,rows in [('fit',fit),('validation',validation)]:equal(evaluate(model,rows,protocol['arc_options']),report['final_'+role])
    baseline=ActionGeometryGuide(a.variant,**protocol['models'][a.variant]).double()
    for role,rows in [('fit',fit),('validation',validation)]:equal(evaluate(baseline,rows,protocol['arc_options']),report['baseline_'+role])
    weights=population_weights(fit);cost=sum(float(w)*2*r['valid'] for w,r in zip(weights,fit))
    rate=sum(float(w)*r.get('baseline_expected_utility_eV',0.) for w,r in zip(weights,fit))/cost
    assert abs(cost-report['fixed_baseline_query_cost'])<1e-12
    assert abs(rate-report['fixed_baseline_utility_per_query_eV'])<1e-12
    heads=[]
    if a.variant!='geometry':heads.append(('action',model.action.action_head[-1].weight))
    if a.variant!='action':heads.append(('geometry',model.geometry.query_head[-1].weight))
    chosen=max((r for r in fit if r['valid']),key=lambda r:abs(r['baseline_expected_utility_eV']))
    def loss():
        u,c,_,_,penalty,_=edge_values(model,chosen,protocol['arc_options'])
        return -(u-rate*c)/cost+protocol['ratio_penalty_eV']*penalty
    gradients=torch.autograd.grad(loss(),[w for name,w in heads]);fd={}
    for (name,w),g in zip(heads,gradients):
        ix=int(g.abs().argmax());ij=(ix//w.shape[1],ix%w.shape[1]);h=1e-5
        with torch.no_grad():w[ij]+=h
        high=float(loss())
        with torch.no_grad():w[ij]-=2*h
        low=float(loss())
        with torch.no_grad():w[ij]+=h
        error=abs((high-low)/(2*h)-float(g[ij]));assert error<1e-6+1e-5*abs(float(g[ij]))
        fd[name]=dict(error=error,gradient=float(g[ij]))
    first={}
    for row in data:
        if row['valid']:first.setdefault((row['index'],row['parent']),row)
    independent=[]
    with torch.no_grad():
        for row in first.values():
            full=full_observed_densities(model,row,protocol['arc_options'])
            for reverse in [False,True]:
                x,y,b,act=((row['y'],row['x'],row['new_bonds'],row['inverse_action']) if reverse else
                           (row['x'],row['y'],row['bonds'],row['action']))
                actions,lp=independent_action_logp(model.action,x,b,row['numbers'],row['electronic'])
                ta,tlp=model.action.legal_log_probabilities(x,b,row['numbers'],row['electronic'])
                assert ta==actions and np.max(np.abs(tlp.numpy()-lp))<1e-9
                action_lp=-math.log(len(actions)) if a.variant=='geometry' else lp[actions.index(tuple(act))]
                kind='arc_site' if a.variant=='action' else 'arc_bounded'
                q,trace=observed_joint_arc_density(x,y,b,row['numbers'],row['electronic'],row['radii'],act,
                    kind=kind,model=None if a.variant=='action' else model.geometry,**protocol['arc_options'])
                independent_q=independent_arc_q(trace)
                error=abs(independent_q+action_lp-float(full[int(reverse)]));assert error<1e-7
                independent.append(error)
    result=dict(complete=True,variant=a.variant,replica=a.replica,protocol_sha256=sha(pp),data_sha256=sha(dp/'data.pt'),
        results_sha256=sha(run/'results.json'),model_sha256=sha(run/'model.pt'),split_rebuilt=True,
        all1671_attempts_retained=True,all_edge_metrics_replayed=True,baseline_replayed=True,
        trained_objective_gradients=fd,independent_full_densities=len(independent),maximum_independent_density_error=max(independent),
        new_physical_queries=0,optimizer_trajectory_replayed=False,scientific_submission_ready=False)
    if a.out.exists():raise FileExistsError(a.out)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
