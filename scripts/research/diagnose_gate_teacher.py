#!/usr/bin/env python3
"""FIT-only gate-target/flow diagnostics; no fitting and no new oracle calls."""
import argparse,json,math
from pathlib import Path
import torch
from cfm_mol.source_force_screen import SourceForceScreen,force_edge_values
from cfm_mol.gate_teacher import oracle_gate_targets,log_retention_lower_bound
from scripts.research.audit_masked_angular import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[2];pp=root/'research/evidence/source_force_screen_training_protocol_v1.json'
    protocol=json.loads(pp.read_text());dp=a.project/protocol['data_run'];assert sha(dp/'data.pt')==protocol['data_sha256']
    data=torch.load(dp/'data.pt',map_location='cpu',weights_only=False);rows=[r for r in data if r['role']=='fit']
    assert len(rows)==1242 and sum(r['valid'] for r in rows)==1186
    reports=[]
    models={name:SourceForceScreen(name,**protocol['model']).double() for name in ['zero','physical','work']}
    provenance={}
    for variant in ['linear','neural']:
        for replica in range(2):
            run=a.project/f'runs/source_force_screen_train_v1/{variant}/replica_{replica}';report=json.loads((run/'results.json').read_text())
            ap=a.project/f'runs/source_force_screen_audit_v1/{variant}/replica_{replica}/results.json';audit=json.loads(ap.read_text())
            assert report['complete'] and audit['complete'] and audit['results_sha256']==sha(run/'results.json')
            assert sha(run/'model.pt')==report['model_sha256']==audit['model_sha256']
            saved=torch.load(run/'model.pt',map_location='cpu',weights_only=False)
            model=SourceForceScreen(**saved['configuration']).double();model.load_state_dict(saved['state_dict'])
            name=f'{variant}_{replica}';models[name]=model;provenance[name]=dict(model_sha256=sha(run/'model.pt'),audit_sha256=sha(ap))
    with torch.no_grad():
        for name,model in models.items():
            mse=over=under=accept=base_accept=cost=ideal_cost=0.;maximum_gap=0.
            for row in rows:
                if not row['valid']:continue
                ratio=row['target_log_ratio']+row['log_behavior_reverse']-row['log_behavior_forward']+row['action_log_ratio']
                tf,tr=oracle_gate_targets(ratio,model.log_factor_bound)
                _,c,_,f,rev,alpha=force_edge_values(model,row)
                mse+=float(((f-tf).square()+(rev-tr).square())/2)
                over+=max(0.,math.exp(float(f))-math.exp(float(tf)))
                under+=max(0.,math.exp(float(tf))-math.exp(float(f)))
                accept+=float(alpha);base_accept+=math.exp(min(0.,float(ratio)))
                cost+=float(c);ideal_cost+=2*math.exp(float(tf))
                bound=log_retention_lower_bound(ratio,f,rev,model.log_factor_bound)
                actual=min(float(f),float(ratio+rev))-min(0.,float(ratio))
                assert actual>=float(bound)-1e-9;maximum_gap=max(maximum_gap,-actual)
            reports.append(dict(method=name,scored_pair_log_gate_mse=mse/1186,
                acceptance_retention=accept/base_accept,query_cost_relative_to_oracle_teacher=cost/ideal_cost,
                forward_gate_overprediction_per_scored_pair=over/1186,forward_gate_underprediction_per_scored_pair=under/1186,
                maximum_log_acceptance_loss=maximum_gap,retention_bound_verified=True))
    result=dict(complete=True,fit_attempts=1242,fit_scored=1186,provenance=provenance,rows=reports,new_physical_queries=0,
        model_fitted=False,oracle_teacher_deployable=False,scientific_submission_ready=False,
        scope='FIT-only diagnosis against oracle-derived gate labels. Teacher needs the unknown candidate energy and is not a cheap algorithm. No new evaluation outcomes or optimizer updates used. Empirical label error does not certify a uniform retention bound on unseen states.')
    if a.out.exists():raise FileExistsError(a.out)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(reports,indent=2))


if __name__=='__main__':main()
