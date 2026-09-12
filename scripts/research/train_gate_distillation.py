#!/usr/bin/env python3
"""Matched gate-label pretraining versus direct whole-prefix utility fitting."""
import argparse,json,math,time,hashlib
from pathlib import Path
import torch
from cfm_mol.source_force_screen import SourceForceScreen,force_edge_values
from scripts.research.audit_masked_angular import sha
from scripts.research.evaluate_chemical_policy import write


from scripts.research.train_source_force_screen import load_inputs,evaluate
from cfm_mol.gate_teacher import oracle_gate_targets


def per_edge_loss(model,row,stage,rate,cost,mean_joints,penalty_weight):
    u,c,_,f,rev,_=force_edge_values(model,row)
    if stage=='teacher':
        if not row['valid']:return u
        ratio=row['target_log_ratio']+row['log_behavior_reverse']-row['log_behavior_forward']+row['action_log_ratio']
        tf,tr=oracle_gate_targets(ratio,model.log_factor_bound)
        return ((f-tf).square()+(rev-tr).square())/(2*model.log_factor_bound**2)
    if stage!='utility':raise ValueError('Unknown fitting stage')
    penalty=(f.square()+rev.square())/(2*model.log_factor_bound**2)
    return -(u-rate*c)/cost+penalty_weight*penalty/mean_joints


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--recipe',choices=['direct','distill'],required=True)
    p.add_argument('--variant',choices=['linear','neural'],required=True);p.add_argument('--replica',choices=[0,1],type=int,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[2]
    pp=root/'research/evidence/gate_distillation_training_protocol_v1.json';protocol=json.loads(pp.read_text())
    data,accounting=load_inputs(root,a.project,protocol)
    groups={role:[r for r in data if r['role']==role] for role in ['fit','withheld_parent']}
    prefix={role:[r for r in accounting['rows'] if r['role']==role] for role in groups}
    fit=groups['fit'];n_prefix=len(prefix['fit']);assert n_prefix==72 and len(prefix['withheld_parent'])==24
    cost=accounting['summary']['fit']['mean_base_calls'];rate=accounting['summary']['fit']['baseline_utility_eV_per_raw_call']
    assert cost==128 and abs(rate-protocol['fixed_fit_baseline_rate'])<1e-12
    weights=torch.full((len(fit),),1/n_prefix,dtype=torch.float64)
    base_u=torch.tensor([r.get('baseline_expected_utility_eV',0.) for r in fit],dtype=torch.float64)
    base_c=torch.tensor([2*int(r['valid']) for r in fit],dtype=torch.float64)
    activity=(base_u-rate*base_c).abs();selection=.5/len(fit)+.5*activity/activity.sum()
    seed=protocol['seeds'][a.replica];torch.manual_seed(seed);rng=torch.Generator().manual_seed(seed+1)
    model=SourceForceScreen(a.variant,**protocol['model']).double()
    optimizer=torch.optim.Adam(model.parameters(),lr=protocol['learning_rate'],weight_decay=protocol['weight_decay'])
    a.out.mkdir(parents=True,exist_ok=True);output=a.out/'results.json'
    if output.exists():raise FileExistsError(output)
    report=dict(complete=False,variant=a.variant,recipe=a.recipe,replica=a.replica,seed=seed,protocol_sha256=sha(pp),data_sha256=protocol['data_sha256'],
        prefix_accounting_sha256=protocol['prefix_accounting_sha256'],fixed_baseline_rate=rate,fixed_prefix_cost=cost,
        curves=[],new_physical_queries=0,actual_queries_saved=0,scientific_submission_ready=False)
    write(output,report);start=time.monotonic()
    try:
        for name in protocol['fixed_controls']:
            baseline=SourceForceScreen(name,**protocol['model']).double()
            for role in groups:report[f'{name}_{role}']=evaluate(baseline,groups[role],prefix[role])
        for role in groups:
            for row in report['zero_'+role]['prefixes']:
                assert row['expected_raw_calls']==row['base_calls']==128
                assert abs(row['expected_work_eV']-row['base_work_eV'])<1e-10
        mean_joints=len(fit)/n_prefix;index_stream=hashlib.sha256()
        for step in range(1,protocol['steps']+1):
            stage='teacher' if a.recipe=='distill' and step<=protocol['phase_boundary'] else 'utility'
            chosen=torch.multinomial(selection,protocol['batch_edges'],replacement=True,generator=rng)
            index_stream.update(chosen.numpy().tobytes())
            optimizer.zero_grad(set_to_none=True);loss=sum(p.sum()*0 for p in model.parameters())
            for i in chosen.tolist():
                adjusted=per_edge_loss(model,fit[i],stage,rate,cost,mean_joints,protocol['gate_penalty_eV'])
                weight=1/len(fit) if stage=='teacher' else weights[i]
                loss=loss+(weight/selection[i])*adjusted/protocol['batch_edges']
            if not torch.isfinite(loss):raise ValueError('Nonfinite source-force loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),protocol['gradient_clip'])
            if not torch.isfinite(norm):raise ValueError('Nonfinite source-force gradient')
            optimizer.step()
            if step in protocol['log_steps']:
                report['curves'].append(dict(step=step,stage=stage,batch_loss=float(loss),gradient_norm=float(norm),elapsed_seconds=time.monotonic()-start))
                write(output,report);print(json.dumps(report['curves'][-1]),flush=True)
            if step==protocol['phase_boundary']:
                torch.save(dict(state_dict=model.state_dict(),configuration=model.configuration,steps=step,seed=seed,replica=a.replica,
                    recipe=a.recipe,protocol_sha256=sha(pp),data_sha256=protocol['data_sha256']),a.out/'phase_boundary_model.pt')
                report['phase_boundary_model_sha256']=sha(a.out/'phase_boundary_model.pt')
                report['phase_boundary_index_sha256']=index_stream.hexdigest()
                # Reset identically in direct and distill to isolate the objective.
                optimizer=torch.optim.Adam(model.parameters(),lr=protocol['learning_rate'],weight_decay=protocol['weight_decay'])
        for role in groups:report['final_'+role]=evaluate(model,groups[role],prefix[role])
        base_joint=sum(r['base_calls']-r['nonjoint_calls'] for r in prefix['fit'])/n_prefix
        probability=report['final_fit']['metrics']['joint_calls']/base_joint
        probability=max(math.exp(-model.log_factor_bound),min(1.,probability))
        thin=SourceForceScreen('thinning',**protocol['model'],thinning_probability=probability).double()
        report['fit_matched_thinning_probability']=probability
        for role in groups:report['thinning_'+role]=evaluate(thin,groups[role],prefix[role])
        torch.save(dict(state_dict=model.state_dict(),configuration=model.configuration,steps=protocol['steps'],seed=seed,replica=a.replica,recipe=a.recipe,
            protocol_sha256=sha(pp),data_sha256=protocol['data_sha256']),a.out/'model.pt')
        report.update(index_stream_sha256=index_stream.hexdigest(),complete=True,steps=protocol['steps'],model_sha256=sha(a.out/'model.pt'),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),elapsed_seconds=time.monotonic()-start)
        write(output,report);print(json.dumps(report['final_withheld_parent']['metrics']),flush=True)
    except Exception as exc:
        torch.save(dict(state_dict=model.state_dict(),configuration=model.configuration),a.out/'failed_model.pt')
        report.update(failure=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-start);write(output,report);raise


if __name__=='__main__':main()
