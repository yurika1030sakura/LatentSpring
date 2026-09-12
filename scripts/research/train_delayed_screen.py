#!/usr/bin/env python3
"""FIT-only cost-adjusted delayed-screen learning with fixed cheap controls."""
import argparse
import json
from pathlib import Path
import time
import torch
from cfm_mol.delayed_acceptance import DelayedAcceptanceScreen,screen_edge_values
from scripts.research.train_accepted_utility import population_weights
from scripts.research.audit_masked_angular import sha
from scripts.research.evaluate_chemical_policy import write


@torch.no_grad()
def evaluate(model,rows):
    weights=population_weights(rows);values=[]
    for row in rows:
        u,c,g,s,accept=screen_edge_values(model,row)
        values.append(dict(index=row['index'],parent=row['parent'],replica=row['replica'],step=row['step'],valid=row['valid'],
            utility_eV=float(u),expected_raw_cost=float(c),accepted_constitutional_flow=float(g),
            log_factor=float(s),expected_acceptance=float(accept)))
    keys=['utility_eV','expected_raw_cost','accepted_constitutional_flow','expected_acceptance']
    def metrics(ids):
        normalizer=sum(float(weights[i]) for i in ids)
        m={k:sum(float(weights[i])*values[i][k] for i in ids)/normalizer for k in keys}
        m['utility_eV_per_expected_raw_call']=m['utility_eV']/m['expected_raw_cost']
        m['maximum_absolute_log_factor']=max(abs(values[i]['log_factor']) for i in ids)
        return m
    return dict(metrics=metrics(list(range(len(rows)))),edges=values,
        per_condition={str(index):metrics([i for i,r in enumerate(rows) if r['index']==index]) for index in sorted({r['index'] for r in rows})},
        scope='Fixed empirical physical proposal population; no importance sampling of new coordinates. Full failed-attempt denominators retained. No achieved query savings, wall-time gain, new chain or equilibrium result.')


def load_data(root,project,protocol):
    dp=project/protocol['data_run'];header=json.loads((dp/'results.json').read_text())
    assert header['complete'] and sha(dp/'results.json')==protocol['data_results_sha256']
    assert header['data_sha256']==sha(dp/'data.pt')==protocol['data_sha256']
    rows=torch.load(dp/'data.pt',map_location='cpu',weights_only=False)
    assert len(rows)==1671 and sum(r['valid'] for r in rows)==1597
    data_protocol=root/'research/evidence/accepted_utility_data_protocol_v1.json'
    assert header['protocol_sha256']==sha(data_protocol)
    split=json.loads(data_protocol.read_text())['splits']
    for row in rows:
        group=split[str(row['index'])]
        assert row['parent'] in group['fit_parent_ids']+group['withheld_parent_ids']
        assert row['role']==('fit' if row['parent'] in group['fit_parent_ids'] else 'withheld_parent')
    fit=[r for r in rows if r['role']=='fit'];validation=[r for r in rows if r['role']=='withheld_parent']
    assert len({(r['index'],r['parent']) for r in fit})==36 and len({(r['index'],r['parent']) for r in validation})==12
    return fit,validation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--variant',choices=['linear','neural'],required=True)
    p.add_argument('--replica',choices=[0,1],type=int,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[2]
    pp=root/'research/evidence/delayed_screen_training_protocol_v1.json';protocol=json.loads(pp.read_text())
    fit,validation=load_data(root,a.project,protocol);weights=population_weights(fit)
    base_u=torch.tensor([r.get('baseline_expected_utility_eV',0.) for r in fit],dtype=torch.float64)
    base_c=torch.tensor([2*int(r['valid']) for r in fit],dtype=torch.float64)
    cost=float((weights*base_c).sum());rate=float((weights*base_u).sum())/cost
    activity=weights*(base_u-rate*base_c).abs();selection=.5*weights+.5*activity/activity.sum()
    seed=protocol['seeds'][a.replica];torch.manual_seed(seed);rng=torch.Generator().manual_seed(seed+1)
    model=DelayedAcceptanceScreen(a.variant,**protocol['model']).double()
    optimizer=torch.optim.Adam(model.parameters(),lr=protocol['learning_rate'],weight_decay=protocol['weight_decay'])
    a.out.mkdir(parents=True,exist_ok=True);output=a.out/'results.json'
    if output.exists():raise FileExistsError(output)
    report=dict(complete=False,variant=a.variant,replica=a.replica,seed=seed,protocol_sha256=sha(pp),data_sha256=protocol['data_sha256'],
        fixed_baseline_query_cost=cost,fixed_baseline_utility_per_query_eV=rate,curves=[],
        new_physical_queries=0,actual_queries_saved=0,scientific_submission_ready=False)
    write(output,report);start=time.monotonic()
    try:
        for name in ['zero','physical']:
            baseline=DelayedAcceptanceScreen(name,**protocol['model']).double()
            for role,rows in [('fit',fit),('validation',validation)]:report[f'{name}_{role}']=evaluate(baseline,rows)
        assert abs(report['zero_fit']['metrics']['utility_eV_per_expected_raw_call']-rate)<1e-10
        for step in range(1,protocol['steps']+1):
            choices=torch.multinomial(selection,protocol['batch_edges'],replacement=True,generator=rng)
            optimizer.zero_grad(set_to_none=True);loss=sum(p.sum()*0 for p in model.parameters())
            for idx in choices.tolist():
                u,c,_,factor,_=screen_edge_values(model,fit[idx])
                adjusted=-(u-rate*c)/cost+protocol['factor_penalty_eV']*(factor/model.log_factor_bound).square()
                loss=loss+(weights[idx]/selection[idx])*adjusted/protocol['batch_edges']
            if not torch.isfinite(loss):raise ValueError('Nonfinite delayed-screen loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),protocol['gradient_clip'])
            if not torch.isfinite(norm):raise ValueError('Nonfinite delayed-screen gradient')
            optimizer.step()
            if step in protocol['log_steps']:
                report['curves'].append(dict(step=step,batch_loss=float(loss),gradient_norm=float(norm),elapsed_seconds=time.monotonic()-start))
                write(output,report);print(json.dumps(report['curves'][-1]),flush=True)
        for role,rows in [('fit',fit),('validation',validation)]:report['final_'+role]=evaluate(model,rows)
        torch.save(dict(state_dict=model.state_dict(),configuration=model.configuration,seed=seed,replica=a.replica,
            steps=protocol['steps'],protocol_sha256=sha(pp),data_sha256=protocol['data_sha256']),a.out/'model.pt')
        report.update(complete=True,steps=protocol['steps'],model_sha256=sha(a.out/'model.pt'),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),elapsed_seconds=time.monotonic()-start)
        write(output,report);print(json.dumps(report['final_validation']['metrics']),flush=True)
    except Exception as exc:
        torch.save(dict(state_dict=model.state_dict(),configuration=model.configuration),a.out/'failed_model.pt')
        report.update(failure=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-start);write(output,report);raise


if __name__=='__main__':main()
