#!/usr/bin/env python3
"""Independent source-force gate calculations and whole-prefix metric replay."""
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from cfm_mol.source_force_screen import SourceForceScreen,force_edge_values
from scripts.research.train_source_force_screen import load_inputs,evaluate
from scripts.research.audit_masked_angular import equal,sha


def independent_gate(model,row,reverse=False):
    x,y,b,new=[row[k].numpy() for k in ['x','y','bonds','new_bonds']]
    z=row['numbers'].numpy();e=row['electronic'].numpy();act=row['action']
    force=row['source_force_eV_A'].numpy()
    q=float(row['log_behavior_reverse']-row['log_behavior_forward']+row['action_log_ratio'])
    if reverse:x,y,b,new,act,force,q=y,x,new,b,row['inverse_action'],row['candidate_force_eV_A'].numpy(),-q
    if model.variant=='thinning':return math.log(model.thinning_probability)
    periodic=Chem.GetPeriodicTable();radii=np.array([periodic.GetRcovalent(int(v)) for v in z]);n=len(z)
    def cheap(p,graph):
        strain=repulsion=0.
        for i in range(n):
            for j in range(i+1,n):
                r=np.linalg.norm(p[i]-p[j])/(radii[i]+radii[j])
                if graph[i,j]>0:strain+=(r-1)**2*graph[i,j]
                else:repulsion+=r**-12
        return np.array([strain,repulsion,((p-p.mean(0))**2).sum()])
    x0=x-x.mean(0);y0=y-y.mean(0);delta=y0-x0
    features=np.append((cheap(x,b)-cheap(y,new))*np.array([10,1,model.restraint/2])/e[2],
                       [q,(force*delta).sum()/e[2]])
    raw=float(features@model.coefficients.detach().numpy())
    if model.encoder is not None:
        s={k:v.detach().numpy() for k,v in model.encoder.state_dict().items()}
        def silu(v):return v*np.exp(-np.logaddexp(0.,-v))
        def mlp(prefix,v,final_silu=False):
            for layer in [0,2]:
                v=v@s[f'{prefix}.{layer}.weight'].T+s[f'{prefix}.{layer}.bias']
                if layer==0 or final_silu:v=silu(v)
            return v
        f=force-model.restraint*x0;f-=f.mean(0);sf=f*radii[:,None]/e[2];sd=delta/radii[:,None]
        scalars=np.stack([np.arcsinh((sf*sd).sum(1)),np.log1p(np.linalg.norm(sf,axis=1)),np.log1p(np.linalg.norm(sd,axis=1))],1)
        roles=np.zeros(n,dtype=int);i,j,k,l=act;roles[[i,j]]=1;roles[[k,l]]=2
        nodes=mlp('atom',s['atomic_features'][z])+mlp('state',e)[None]+mlp('force',scalars)+s['roles.weight'][roles]
        dx=np.sqrt(((x0[:,None]-x0[None,:])**2).sum(-1)+1e-8)
        dy=np.sqrt(((y0[:,None]-y0[None,:])**2).sum(-1)+1e-8);rsum=radii[:,None]+radii[None,:]
        radial=np.concatenate([np.exp(-.5*((d[...,None]/rsum[...,None]-s['centers'])/.4)**2) for d in [dx,dy]],-1)
        cut=lambda d:np.where(d<6,.5*(1+np.cos(np.pi*d/6)),0.)
        gate=np.maximum(cut(dx),cut(dy))*(1-np.eye(n))
        for layer in range(2):
            a=np.broadcast_to(nodes[:,None],(n,n,nodes.shape[-1]));other=np.broadcast_to(nodes[None,:],a.shape)
            pair=mlp(f'messages.{layer}',np.concatenate([a+other,(a-other)**2,radial,b[...,None]/3,new[...,None]/3],-1),True)
            aggregate=(pair*gate[...,None]).sum(1)/np.maximum(1,gate.sum(1))[:,None]
            nodes=nodes+mlp(f'updates.{layer}',np.concatenate([nodes,aggregate],-1))
        raw+=float((nodes@s['head.weight'].T+s['head.bias']).sum()/math.sqrt(n))
    return min(0.,float(model.log_factor_bound*np.tanh(raw/model.log_factor_bound)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','run','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--variant',choices=['linear','neural'],required=True);p.add_argument('--replica',choices=[0,1],type=int,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[2]
    pp=root/'research/evidence/source_force_screen_training_protocol_v1.json';protocol=json.loads(pp.read_text())
    data,accounting=load_inputs(root,a.project,protocol);groups={role:[r for r in data if r['role']==role] for role in ['fit','withheld_parent']}
    prefix={role:[r for r in accounting['rows'] if r['role']==role] for role in groups}
    run=a.run/a.variant/f'replica_{a.replica}';report=json.loads((run/'results.json').read_text())
    assert report['complete'] and report['protocol_sha256']==sha(pp) and report['steps']==protocol['steps']
    assert report['variant']==a.variant and report['replica']==a.replica and report['seed']==protocol['seeds'][a.replica]
    assert sha(run/'model.pt')==report['model_sha256'] and report['data_sha256']==protocol['data_sha256']
    saved=torch.load(run/'model.pt',map_location='cpu',weights_only=False);model=SourceForceScreen(a.variant,**protocol['model']).double()
    assert saved['configuration']==model.configuration and saved['protocol_sha256']==sha(pp) and saved['data_sha256']==protocol['data_sha256']
    assert saved['steps']==protocol['steps'] and saved['seed']==protocol['seeds'][a.replica] and saved['replica']==a.replica
    model.load_state_dict(saved['state_dict'])
    for role in groups:equal(evaluate(model,groups[role],prefix[role]),report['final_'+role])
    controls={name:SourceForceScreen(name,**protocol['model']).double() for name in protocol['fixed_controls']}
    base_joint=sum(r['base_calls']-r['nonjoint_calls'] for r in prefix['fit'])/len(prefix['fit'])
    probability=report['final_fit']['metrics']['joint_calls']/base_joint
    assert abs(probability-report['fit_matched_thinning_probability'])<1e-12
    controls['thinning']=SourceForceScreen('thinning',**protocol['model'],thinning_probability=report['fit_matched_thinning_probability']).double()
    for name,control in controls.items():
        for role in groups:equal(evaluate(control,groups[role],prefix[role]),report[name+'_'+role])
    checks=0;maximum=0.
    with torch.no_grad():
        for row in data:
            if not row['valid']:continue
            f=independent_gate(model,row);rev=independent_gate(model,row,True)
            u,c,_,actual_f,actual_r,accept=force_edge_values(model,row)
            ratio=float(row['target_log_ratio']+row['log_behavior_reverse']-row['log_behavior_forward']+row['action_log_ratio'])
            total=min(f,ratio+rev);back=min(rev,-ratio+f)
            assert abs(total-back-ratio)<1e-9 and total<=min(0.,ratio)+1e-12
            err=max(abs(f-float(actual_f)),abs(rev-float(actual_r)),abs(float(u)-float(row['reward_eV'])*math.exp(total)),abs(float(c)-2*math.exp(f)),abs(float(accept)-math.exp(total)))
            assert err<1e-8;maximum=max(maximum,err);checks+=1
    assert checks==1597
    cost=accounting['summary']['fit']['mean_base_calls'];rate=accounting['summary']['fit']['baseline_utility_eV_per_raw_call']
    assert abs(rate-report['fixed_baseline_rate'])<1e-12 and cost==report['fixed_prefix_cost']==128
    mean_joints=len(groups['fit'])/len(prefix['fit'])
    def loss(row):
        u,c,_,f,rev,_=force_edge_values(model,row)
        penalty=(f.square()+rev.square())/(2*model.log_factor_bound**2)
        return -(u-rate*c)/cost+protocol['gate_penalty_eV']*penalty/mean_joints
    first={}
    for row in groups['fit']:
        if row['valid']:first.setdefault((row['index'],row['parent']),row)
    heads=[('linear',model.coefficients)]
    if model.encoder is not None:heads.append(('neural',model.encoder.head.weight))
    best={}
    for row in first.values():
        grads=torch.autograd.grad(loss(row),[w for name,w in heads])
        for (name,w),g in zip(heads,grads):
            mag=float(g.abs().max())
            if name not in best or mag>best[name][0]:best[name]=(mag,row,g)
    fd={}
    for name,w in heads:
        mag,row,g=best[name];ix=int(g.abs().argmax());assert mag>1e-12;h=1e-5
        with torch.no_grad():w.flatten()[ix]+=h
        high=float(loss(row))
        with torch.no_grad():w.flatten()[ix]-=2*h
        low=float(loss(row))
        with torch.no_grad():w.flatten()[ix]+=h
        error=abs((high-low)/(2*h)-float(g.flatten()[ix]));assert error<1e-6+1e-5*mag
        fd[name]=dict(error=error,gradient=float(g.flatten()[ix]),index=row['index'],parent=row['parent'])
    output=dict(complete=True,variant=a.variant,replica=a.replica,protocol_sha256=sha(pp),data_sha256=protocol['data_sha256'],
        prefix_accounting_sha256=protocol['prefix_accounting_sha256'],results_sha256=sha(run/'results.json'),model_sha256=sha(run/'model.pt'),
        all1671_attempts_retained=True,all_metrics_replayed=True,all_controls_replayed=True,independent_supported_pairs=checks,
        maximum_independent_error=maximum,general_gate_balance_verified=True,trained_objective_gradients=fd,
        thinning_uses_fit_only=True,new_physical_queries=0,actual_queries_saved=0,optimizer_trajectory_replayed=False,scientific_submission_ready=False)
    if a.out.exists():raise FileExistsError(a.out)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output))


if __name__=='__main__':main()
