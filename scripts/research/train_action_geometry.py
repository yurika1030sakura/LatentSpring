#!/usr/bin/env python3
"""Matched conditional action/geometry accepted-utility ablations."""
import argparse,json,math
from pathlib import Path
from statistics import mean
import time
import torch
from cfm_mol.bounded_action_geometry import ActionGeometryGuide,edge_values
from scripts.research.train_accepted_utility import population_weights,evaluate as evaluate_geometry
from scripts.research.audit_masked_angular import sha
from scripts.research.evaluate_chemical_policy import write


def evaluate(model,records,options):
    return evaluate_geometry(model,records,options,edge_evaluator=edge_values)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--variant',choices=['action','geometry','joint'],required=True)
    p.add_argument('--replica',type=int,choices=[0,1],required=True)
    args=p.parse_args();root=Path(__file__).resolve().parents[2]
    pp=root/'research/evidence/action_geometry_training_protocol_v1.json';protocol=json.loads(pp.read_text())
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
    model=ActionGeometryGuide(variant=args.variant,**protocol['models'][args.variant]).double()
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=protocol['learning_rate'],weight_decay=protocol['weight_decay'])
    args.out.mkdir(parents=True,exist_ok=True);output=args.out/'results.json'
    if output.exists():raise FileExistsError(output)
    report=dict(complete=False,replica=args.replica,variant=args.variant,seed=seed,protocol_sha256=sha(pp),data_sha256=protocol['data_sha256'],
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
            optimizer.zero_grad(set_to_none=True);loss=sum(p.sum()*0 for p in model.parameters() if p.requires_grad)
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
            score_semantics='bounded_conditional_action_and_geometry'),args.out/'model.pt')
        report.update(trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),complete=True,steps=protocol['steps'],model_sha256=sha(args.out/'model.pt'),elapsed_seconds=time.monotonic()-start)
        write(output,report);print(json.dumps(report['final_validation']['metrics']),flush=True)
    except Exception as exc:
        torch.save(dict(state_dict=model.state_dict(),configuration=model.configuration),args.out/'failed_model.pt')
        report.update(failure=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-start);write(output,report);raise


if __name__=='__main__':main()
