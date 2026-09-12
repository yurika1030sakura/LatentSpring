#!/usr/bin/env python3
"""Source-force gates fitted against a fixed whole-prefix cost/utility proxy."""
import argparse,json,math,time
from pathlib import Path
import torch
from cfm_mol.source_force_screen import SourceForceScreen,force_edge_values
from scripts.research.audit_masked_angular import sha
from scripts.research.evaluate_chemical_policy import write


def load_inputs(root,project,protocol):
    directory=project/protocol['data_run'];header=json.loads((directory/'results.json').read_text())
    assert header['complete'] and sha(directory/'results.json')==protocol['data_results_sha256']
    assert sha(directory/'data.pt')==header['data_sha256']==protocol['data_sha256']
    ap=project/protocol['prefix_accounting'];assert sha(ap)==protocol['prefix_accounting_sha256']
    accounting=json.loads(ap.read_text());assert accounting['complete'] and accounting['pair_data_sha256']==header['data_sha256']
    data=torch.load(directory/'data.pt',map_location='cpu',weights_only=False)
    assert len(data)==1671 and sum(r['valid'] for r in data)==1597
    split=json.loads((root/'research/evidence/accepted_utility_data_protocol_v1.json').read_text())['splits']
    for r in data:
        s=split[str(r['index'])];assert r['parent'] in s['fit_parent_ids']+s['withheld_parent_ids']
        assert r['role']==('fit' if r['parent'] in s['fit_parent_ids'] else 'withheld_parent')
    return data,accounting


@torch.no_grad()
def evaluate(model,rows,prefixes):
    totals={(r['index'],r['parent'],r['replica']):dict(r,joint_work_eV=0.,joint_calls=0.,joint_acceptance=0.,joint_constitutional_flow=0.) for r in prefixes}
    edges=[]
    for row in rows:
        u,c,g,f,rev,accept=force_edge_values(model,row);key=(row['index'],row['parent'],row['replica']);target=totals[key]
        target['joint_work_eV']+=float(u);target['joint_calls']+=float(c);target['joint_acceptance']+=float(accept);target['joint_constitutional_flow']+=float(g)
        edges.append(dict(index=row['index'],parent=row['parent'],replica=row['replica'],step=row['step'],valid=row['valid'],
            utility_eV=float(u),expected_raw_cost=float(c),forward_log_gate=float(f),reverse_log_gate=float(rev),expected_acceptance=float(accept)))
    for row in totals.values():
        row['expected_work_eV']=row['nonjoint_work_eV']+row['joint_work_eV']
        row['expected_raw_calls']=row['nonjoint_calls']+row['joint_calls']
    def metric(selected):
        keys=['expected_work_eV','expected_raw_calls','joint_work_eV','joint_calls','joint_acceptance','joint_constitutional_flow']
        m={k:sum(r[k] for r in selected)/len(selected) for k in keys}
        m['utility_eV_per_expected_raw_call']=m['expected_work_eV']/m['expected_raw_calls']
        m['joint_utility_eV_per_expected_raw_call']=m['joint_work_eV']/m['joint_calls']
        return m
    values=list(totals.values())
    return dict(metrics=metric(values),per_condition={str(i):metric([r for r in values if r['index']==i]) for i in sorted({r['index'] for r in values})},
        prefixes=values,edges=edges,scope='Whole recorded-prefix proxy: initial/nonjoint calls and utilities fixed, screened joint expectations substituted at recorded states. Not a changed chain, an achieved query saving or an endpoint-density estimate.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--variant',choices=['linear','neural'],required=True);p.add_argument('--replica',choices=[0,1],type=int,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[2]
    pp=root/'research/evidence/source_force_screen_training_protocol_v1.json';protocol=json.loads(pp.read_text())
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
    report=dict(complete=False,variant=a.variant,replica=a.replica,seed=seed,protocol_sha256=sha(pp),data_sha256=protocol['data_sha256'],
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
        mean_joints=len(fit)/n_prefix
        for step in range(1,protocol['steps']+1):
            chosen=torch.multinomial(selection,protocol['batch_edges'],replacement=True,generator=rng)
            optimizer.zero_grad(set_to_none=True);loss=sum(p.sum()*0 for p in model.parameters())
            for i in chosen.tolist():
                u,c,_,f,rev,_=force_edge_values(model,fit[i])
                penalty=(f.square()+rev.square())/(2*model.log_factor_bound**2)
                adjusted=-(u-rate*c)/cost+protocol['gate_penalty_eV']*penalty/mean_joints
                loss=loss+(weights[i]/selection[i])*adjusted/protocol['batch_edges']
            if not torch.isfinite(loss):raise ValueError('Nonfinite source-force loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),protocol['gradient_clip'])
            if not torch.isfinite(norm):raise ValueError('Nonfinite source-force gradient')
            optimizer.step()
            if step in protocol['log_steps']:
                report['curves'].append(dict(step=step,batch_loss=float(loss),gradient_norm=float(norm),elapsed_seconds=time.monotonic()-start))
                write(output,report);print(json.dumps(report['curves'][-1]),flush=True)
        for role in groups:report['final_'+role]=evaluate(model,groups[role],prefix[role])
        base_joint=sum(r['base_calls']-r['nonjoint_calls'] for r in prefix['fit'])/n_prefix
        probability=report['final_fit']['metrics']['joint_calls']/base_joint
        probability=max(math.exp(-model.log_factor_bound),min(1.,probability))
        thin=SourceForceScreen('thinning',**protocol['model'],thinning_probability=probability).double()
        report['fit_matched_thinning_probability']=probability
        for role in groups:report['thinning_'+role]=evaluate(thin,groups[role],prefix[role])
        torch.save(dict(state_dict=model.state_dict(),configuration=model.configuration,steps=protocol['steps'],seed=seed,replica=a.replica,
            protocol_sha256=sha(pp),data_sha256=protocol['data_sha256']),a.out/'model.pt')
        report.update(complete=True,steps=protocol['steps'],model_sha256=sha(a.out/'model.pt'),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),elapsed_seconds=time.monotonic()-start)
        write(output,report);print(json.dumps(report['final_withheld_parent']['metrics']),flush=True)
    except Exception as exc:
        torch.save(dict(state_dict=model.state_dict(),configuration=model.configuration),a.out/'failed_model.pt')
        report.update(failure=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-start);write(output,report);raise


if __name__=='__main__':main()
