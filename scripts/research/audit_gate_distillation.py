#!/usr/bin/env python3
"""Independent source-force gate calculations and whole-prefix metric replay."""
import argparse,json,math,hashlib
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from cfm_mol.source_force_screen import SourceForceScreen,force_edge_values
from scripts.research.train_source_force_screen import load_inputs,evaluate
from scripts.research.audit_masked_angular import equal,sha


from scripts.research.audit_source_force_screen import independent_gate
from scripts.research.train_gate_distillation import per_edge_loss


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','run','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--recipe',choices=['direct','distill'],required=True)
    p.add_argument('--variant',choices=['linear','neural'],required=True);p.add_argument('--replica',choices=[0,1],type=int,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[2]
    pp=root/'research/evidence/gate_distillation_training_protocol_v1.json';protocol=json.loads(pp.read_text())
    data,accounting=load_inputs(root,a.project,protocol);groups={role:[r for r in data if r['role']==role] for role in ['fit','withheld_parent']}
    prefix={role:[r for r in accounting['rows'] if r['role']==role] for role in groups}
    run=a.run/a.recipe/a.variant/f'replica_{a.replica}';report=json.loads((run/'results.json').read_text())
    assert report['recipe']==a.recipe
    assert report['complete'] and report['protocol_sha256']==sha(pp) and report['steps']==protocol['steps']
    assert report['variant']==a.variant and report['replica']==a.replica and report['seed']==protocol['seeds'][a.replica]
    assert sha(run/'model.pt')==report['model_sha256'] and report['data_sha256']==protocol['data_sha256']
    saved=torch.load(run/'model.pt',map_location='cpu',weights_only=False);model=SourceForceScreen(a.variant,**protocol['model']).double()
    assert saved['recipe']==a.recipe
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
    boundary_path=run/'phase_boundary_model.pt';assert sha(boundary_path)==report['phase_boundary_model_sha256']
    boundary=torch.load(boundary_path,map_location='cpu',weights_only=False)
    assert boundary['configuration']==model.configuration and boundary['seed']==protocol['seeds'][a.replica] and boundary['replica']==a.replica and boundary['data_sha256']==protocol['data_sha256']
    assert boundary['recipe']==a.recipe and boundary['steps']==protocol['phase_boundary'] and boundary['protocol_sha256']==sha(pp)
    stage='teacher' if a.recipe=='distill' else 'utility'
    intermediate=SourceForceScreen(a.variant,**protocol['model']).double();intermediate.load_state_dict(boundary['state_dict'])
    teacher_heads=[('linear',intermediate.coefficients)]
    if intermediate.encoder is not None:teacher_heads.append(('neural',intermediate.encoder.head.weight))
    best_boundary={}
    def boundary_loss(row):return per_edge_loss(intermediate,row,stage,rate,cost,mean_joints,protocol['gate_penalty_eV'])
    for row in first.values():
        gs=torch.autograd.grad(boundary_loss(row),[w for name,w in teacher_heads])
        for (name,w),g in zip(teacher_heads,gs):
            mag=float(g.abs().max())
            if name not in best_boundary or mag>best_boundary[name][0]:best_boundary[name]=(mag,row,g)
    boundary_fd={}
    for name,w in teacher_heads:
        mag,row,g=best_boundary[name];assert mag>1e-12;ix=int(g.abs().argmax());h=1e-6
        with torch.no_grad():w.flatten()[ix]+=h
        high=float(boundary_loss(row))
        with torch.no_grad():w.flatten()[ix]-=2*h
        low=float(boundary_loss(row))
        with torch.no_grad():w.flatten()[ix]+=h
        error=abs((high-low)/(2*h)-float(g.flatten()[ix]));assert error<1e-6+1e-5*mag
        boundary_fd[name]=dict(error=error,gradient=float(g.flatten()[ix]),index=row['index'],parent=row['parent'])
    fit=groups['fit'];base_u=torch.tensor([r.get('baseline_expected_utility_eV',0.) for r in fit],dtype=torch.float64)
    base_c=torch.tensor([2*int(r['valid']) for r in fit],dtype=torch.float64)
    activity=(base_u-rate*base_c).abs();selection=.5/len(fit)+.5*activity/activity.sum()
    generator=torch.Generator().manual_seed(protocol['seeds'][a.replica]+1);digest=hashlib.sha256()
    for step in range(1,protocol['steps']+1):
        chosen=torch.multinomial(selection,protocol['batch_edges'],replacement=True,generator=generator)
        digest.update(chosen.numpy().tobytes())
        if step==protocol['phase_boundary']:assert digest.hexdigest()==report['phase_boundary_index_sha256']
    assert digest.hexdigest()==report['index_stream_sha256']
    output=dict(complete=True,variant=a.variant,recipe=a.recipe,replica=a.replica,protocol_sha256=sha(pp),data_sha256=protocol['data_sha256'],
        prefix_accounting_sha256=protocol['prefix_accounting_sha256'],results_sha256=sha(run/'results.json'),model_sha256=sha(run/'model.pt'),
        all1671_attempts_retained=True,all_metrics_replayed=True,all_controls_replayed=True,independent_supported_pairs=checks,
        maximum_independent_error=maximum,general_gate_balance_verified=True,trained_objective_gradients=fd,phase_boundary_sha256=sha(boundary_path),phase_boundary_stage=stage,phase_boundary_gradients=boundary_fd,
        index_stream_rebuilt=True,index_stream_sha256=digest.hexdigest(),thinning_uses_fit_only=True,new_physical_queries=0,actual_queries_saved=0,optimizer_trajectory_replayed=False,scientific_submission_ready=False)
    if a.out.exists():raise FileExistsError(a.out)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output))


if __name__=='__main__':main()
