#!/usr/bin/env python3
"""Rebuild protected data, reload every frozen learner, and replay all metrics."""
import argparse
import json
from pathlib import Path
from statistics import mean
import torch

from cfm_mol.conditional_arc_energy import ConditionalArcEnergy
from scripts.research.audit_masked_angular import equal,sha
from scripts.research.build_conditional_arc_data import build
from scripts.research.train_conditional_arc_energy import evaluate,objective


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','run','out']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();root=Path(__file__).resolve().parents[2]
    pp=root/'research/evidence/conditional_arc_training_protocol_v1.json';protocol=json.loads(pp.read_text())
    directory=args.project/protocol['data_run']
    report=json.loads((directory/'results.json').read_text())
    assert sha(directory/'results.json')==protocol['data_results_sha256']
    assert sha(directory/'data.pt')==protocol['data_sha256']==report['data_sha256']
    saved=torch.load(directory/'data.pt',map_location='cpu',weights_only=False)
    records,header=build(args.project,root)
    equal(records,saved);equal(header,{k:v for k,v in report.items() if k!='data_sha256'})
    summaries={};audits=[]
    for kind in protocol['objectives']:
        for replica,seed in enumerate(protocol['seeds']):
            run=args.run/f'{kind}_s{replica}';r=json.loads((run/'results.json').read_text())
            assert r['complete'] and r['objective']==kind and r['replica']==replica and r['seed']==seed
            assert r['protocol_sha256']==sha(pp) and r['data_sha256']==protocol['data_sha256']
            assert r['optimizer_steps']==protocol['steps'] and [v['step'] for v in r['curves']]==protocol['checkpoints']
            assert sha(run/'model.pt')==r['model_sha256']
            checkpoint=torch.load(run/'model.pt',map_location='cpu',weights_only=False)
            for key,value in [('objective',kind),('replica',replica),('seed',seed),('step',protocol['steps']),('protocol_sha256',sha(pp)),('data_sha256',protocol['data_sha256'])]:
                assert checkpoint[key]==value
            assert checkpoint['configuration']==protocol['model']
            model=ConditionalArcEnergy(**checkpoint['configuration']).double();model.load_state_dict(checkpoint['state_dict'])
            assert all(torch.isfinite(p).all() for p in model.parameters())
            metric=evaluate(model,records);equal(metric,r['final'])
            equal(metric['summary'],r['curves'][-1]['metrics'])
            for key,restraint in [('initialization',protocol['model']['restraint']),('physical_site',0.)]:
                baseline=ConditionalArcEnergy(**dict(protocol['model'],restraint=restraint)).double()
                equal(evaluate(baseline,records),r[key])
            # Finite differences of the actual fitting objective at a trained
            # nonzero readout coefficient, including force-training derivatives.
            rows=[v for v in records if v['role']=='fit' and v['index']==1][:4]
            loss,_=objective(model,rows,protocol['force_weights'][kind],protocol['energy_scale_eV'])
            weight=model.query_head[-1].weight
            grad=torch.autograd.grad(loss,weight)[0];flat=int(grad.abs().argmax());i,j=flat//weight.shape[1],flat%weight.shape[1]
            h=1e-5
            with torch.no_grad():weight[i,j]+=h
            a=float(objective(model,rows,protocol['force_weights'][kind],protocol['energy_scale_eV'])[0])
            with torch.no_grad():weight[i,j]-=2*h
            b=float(objective(model,rows,protocol['force_weights'][kind],protocol['energy_scale_eV'])[0])
            with torch.no_grad():weight[i,j]+=h
            fd=(a-b)/(2*h);error=abs(fd-float(grad[i,j]))
            assert error<1e-6+1e-5*abs(fd)
            summaries[f'{kind}_s{replica}']={g:{name:r[name]['summary'][g] for name in ['physical_site','initialization','final']}
                for g in ['withheld_parent/arc','withheld_composition/arc','withheld_parent/local_check','withheld_composition/local_check']}
            audits.append(dict(name=f'{kind}_s{replica}',results_sha256=sha(run/'results.json'),model_sha256=sha(run/'model.pt'),
                all_final_context_metrics_replayed=True,baseline_metrics_replayed=True,actual_loss_fd_gradient_error=error))
    result=dict(complete=True,protocol_sha256=sha(pp),data_results_sha256=sha(directory/'results.json'),
        full_data_rebuilt=True,all438_contexts_equal=True,all_split_masks_rebuilt=True,rows=audits,internal_metrics=summaries,
        optimizer_trajectory_replayed=False,new_physical_queries=0,scientific_submission_ready=False,
        scope='All four final models and both physical score baselines re-evaluated on rebuilt labels; actual trained-loss finite differences checked. Internal conditional prediction only, not complete-chain benefit or final test.')
    if args.out.exists():raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(complete=True,models=len(audits),full_data_rebuilt=True,gradient_errors=[r['actual_loss_fd_gradient_error'] for r in audits])))


if __name__=='__main__':main()
