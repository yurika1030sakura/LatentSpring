#!/usr/bin/env python3
"""Offline cost-adjusted accepted-work learning with bounded proposal scores."""
import argparse,json,math
from pathlib import Path
from statistics import mean
import time
import torch
from cfm_mol.accepted_utility import accepted_importance_utility
from cfm_mol.bounded_arc_guide import BoundedArcGuide
from cfm_mol.joint_arc_geometry import observed_joint_arc_density
from scripts.research.audit_masked_angular import sha
from scripts.research.evaluate_chemical_policy import write


def population_weights(records):
    groups={}
    for i,r in enumerate(records):groups.setdefault((r['index'],r['parent'],r['replica']),[]).append(i)
    cases=sorted({r['index'] for r in records});weights=torch.zeros(len(records),dtype=torch.float64)
    for (index,parent,replica),ids in groups.items():
        parents={r['parent'] for r in records if r['index']==index}
        replicas={r['replica'] for r in records if r['index']==index and r['parent']==parent}
        weights[ids]=1/(len(cases)*len(parents)*len(replicas)*len(ids))
    assert abs(float(weights.sum())-1)<1e-12
    return weights


def edge_values(model,row,options):
    zero=model.query_head[-1].weight.sum()*0
    if not row['valid']:
        return zero,zero,zero,zero,zero,zero
    qf,_=observed_joint_arc_density(row['x'],row['y'],row['bonds'],row['numbers'],row['electronic'],row['radii'],row['action'],
        kind='arc_bounded',model=model,**options)
    qr,_=observed_joint_arc_density(row['y'],row['x'],row['new_bonds'],row['numbers'],row['electronic'],row['radii'],row['inverse_action'],
        kind='arc_bounded',model=model,**options)
    if not torch.isfinite(qf) or not torch.isfinite(qr):raise ValueError('Recorded supported edge changed support')
    lf=qf-row['log_behavior_forward'];lr=qr-row['log_behavior_reverse']
    bound=4*model.log_score_bound
    if abs(float(lf))>bound+1e-7 or abs(float(lr))>bound+1e-7:raise ValueError('Joint proposal likelihood-ratio bound violated')
    utility,flow=accepted_importance_utility(qf,qr,row['log_behavior_forward'],row['target_log_ratio'],row['action_log_ratio'],row['reward_eV'])
    importance=lf.exp()
    penalty=(lf.square()+lr.square())/2
    changed=flow*int(row['connectivity_changed'])
    return utility,2*importance,changed,importance,penalty,lf


@torch.no_grad()
def evaluate(model,records,options,edge_evaluator=edge_values):
    weights=population_weights(records);values=[]
    for r in records:
        u,c,g,w,p,l=edge_evaluator(model,r,options)
        values.append(dict(index=r['index'],parent=r['parent'],replica=r['replica'],step=r['step'],valid=r['valid'],
            utility_eV=float(u),expected_raw_cost=float(c),accepted_constitutional_flow=float(g),
            scored_importance=float(w),ratio_penalty=float(p),log_forward_ratio=float(l)))
    metrics={key:sum(float(weights[i])*r[key] for i,r in enumerate(values)) for key in
        ['utility_eV','expected_raw_cost','accepted_constitutional_flow','ratio_penalty']}
    mass=torch.tensor([r['scored_importance'] for r in values],dtype=weights.dtype)*weights
    metrics.update(utility_eV_per_expected_raw_call=metrics['utility_eV']/metrics['expected_raw_cost'],
        scored_probability_estimate=float(mass.sum()),weighted_scored_edge_ESS=float(mass.sum().square()/mass.square().sum()),
        maximum_scored_importance=max(r['scored_importance'] for r in values),
        maximum_absolute_log_forward_ratio=max(abs(r['log_forward_ratio']) for r in values))
    by_case={}
    for index in sorted({r['index'] for r in values}):
        ids=[i for i,r in enumerate(values) if r['index']==index];normalizer=float(weights[ids].sum())
        by_case[str(index)]={key:sum(float(weights[i])*values[i][key] for i in ids)/normalizer for key in
            ['utility_eV','expected_raw_cost','accepted_constitutional_flow']}
        d=by_case[str(index)];d['utility_eV_per_expected_raw_call']=d['utility_eV']/d['expected_raw_cost']
    return dict(metrics=metrics,per_condition=by_case,edges=values,
        scope='Fixed empirical source population and scored-edge importance diagnostics. ESS is not independent molecular samples. No full-chain prediction or equilibrium claim.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--replica',type=int,choices=[0,1],required=True)
    args=p.parse_args();root=Path(__file__).resolve().parents[2]
    pp=root/'research/evidence/accepted_utility_training_protocol_v1.json';protocol=json.loads(pp.read_text())
    dp=args.project/protocol['data_run'];header=json.loads((dp/'results.json').read_text())
    assert header['complete'] and sha(dp/'results.json')==protocol['data_results_sha256']
    assert sha(dp/'data.pt')==header['data_sha256']==protocol['data_sha256']
    records=torch.load(dp/'data.pt',map_location='cpu',weights_only=False)
    fit=[r for r in records if r['role']=='fit'];validation=[r for r in records if r['role']=='withheld_parent']
    assert len({(r['index'],r['parent']) for r in fit})==36 and len({(r['index'],r['parent']) for r in validation})==12
    weights=population_weights(fit)
    baseline_u=torch.tensor([r.get('baseline_expected_utility_eV',0.) for r in fit],dtype=torch.float64)
    baseline_c=torch.tensor([2*int(r['valid']) for r in fit],dtype=torch.float64)
    initial_cost=float((weights*baseline_c).sum());rate=float((weights*baseline_u).sum())/initial_cost
    activity=(baseline_u-rate*baseline_c).abs()
    useful=weights*activity
    if not float(useful.sum())>0:raise ValueError('No accepted-work variation in FIT behavior data')
    useful/=useful.sum()
    selection=.5*weights+.5*useful
    assert (selection>0).all() and abs(float(selection.sum())-1)<1e-12
    seed=protocol['seeds'][args.replica];torch.manual_seed(seed);rng=torch.Generator().manual_seed(seed+1)
    model=BoundedArcGuide(**protocol['model']).double()
    optimizer=torch.optim.Adam(model.parameters(),lr=protocol['learning_rate'],weight_decay=protocol['weight_decay'])
    args.out.mkdir(parents=True,exist_ok=True);output=args.out/'results.json'
    if output.exists():raise FileExistsError(output)
    report=dict(complete=False,replica=args.replica,seed=seed,protocol_sha256=sha(pp),data_sha256=protocol['data_sha256'],
        fixed_baseline_utility_per_query_eV=rate,fixed_baseline_query_cost=initial_cost,curves=[],
        new_physical_queries=0,scientific_submission_ready=False,
        scope='Offline empirical cost-adjusted accepted-work objective. All attempt weights retained; sampling indices are stratified with exact empirical selection correction. No behavior weights are clipped. No learned-chain result follows.')
    write(output,report);start=time.monotonic()
    try:
        report['baseline_fit']=evaluate(model,fit,protocol['arc_options'])
        report['baseline_validation']=evaluate(model,validation,protocol['arc_options']);write(output,report)
        assert abs(report['baseline_fit']['metrics']['utility_eV_per_expected_raw_call']-rate)<1e-9
        for step in range(1,protocol['steps']+1):
            choices=torch.multinomial(selection,protocol['batch_edges'],replacement=True,generator=rng)
            optimizer.zero_grad(set_to_none=True);loss=model.query_head[-1].weight.sum()*0
            for index in choices.tolist():
                u,c,g,w,penalty,l=edge_values(model,fit[index],protocol['arc_options'])
                adjusted=-(u-rate*c)/initial_cost+protocol['ratio_penalty_eV']*penalty
                loss=loss+(weights[index]/selection[index])*adjusted/protocol['batch_edges']
            if not torch.isfinite(loss):raise ValueError('Nonfinite accepted-utility loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),protocol['gradient_clip'])
            if not torch.isfinite(norm):raise ValueError('Nonfinite observed-density gradient')
            optimizer.step()
            if step in protocol['log_steps']:
                report['curves'].append(dict(step=step,batch_loss=float(loss),gradient_norm=float(norm),elapsed_seconds=time.monotonic()-start))
                write(output,report);print(json.dumps(report['curves'][-1]),flush=True)
        report['final_fit']=evaluate(model,fit,protocol['arc_options'])
        report['final_validation']=evaluate(model,validation,protocol['arc_options'])
        torch.save(dict(state_dict=model.state_dict(),configuration=model.configuration,replica=args.replica,seed=seed,
            protocol_sha256=sha(pp),data_sha256=protocol['data_sha256'],steps=protocol['steps'],
            score_semantics='bounded_log_score'),args.out/'model.pt')
        report.update(complete=True,steps=protocol['steps'],model_sha256=sha(args.out/'model.pt'),elapsed_seconds=time.monotonic()-start)
        write(output,report);print(json.dumps(report['final_validation']['metrics']),flush=True)
    except Exception as exc:
        torch.save(dict(state_dict=model.state_dict(),configuration=model.configuration),args.out/'failed_model.pt')
        report.update(failure=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-start);write(output,report);raise


if __name__=='__main__':main()
