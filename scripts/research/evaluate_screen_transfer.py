#!/usr/bin/env python3
"""Evaluation-only reuse of a larger held-out physical cohort; no oracle calls."""
import argparse,json,math
from pathlib import Path
import torch
from cfm_mol.source_force_screen import SourceForceScreen,force_edge_values
from scripts.research.train_source_force_screen import evaluate
from scripts.research.audit_source_force_screen import independent_gate
from scripts.research.audit_masked_angular import equal,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--run',type=Path);p.add_argument('--index',type=int,required=True)
    p.add_argument('--phase',choices=['evaluate','audit'],required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[2];pp=root/'research/evidence/screen_transfer_protocol_v1.json';protocol=json.loads(pp.read_text())
    assert a.index in protocol['condition_indices'] and not protocol['model_fitting_allowed']
    manifest=json.loads((root/'research/evidence/screen_transfer_data_v1.json').read_text());directory=a.project/manifest['run']
    header=json.loads((directory/'results.json').read_text())
    assert header['complete'] and header['protocol_sha256']==sha(pp) and sha(directory/'results.json')==manifest['results_sha256']
    assert header['data_sha256']==manifest['data_sha256']==sha(directory/'data.pt')
    assert header['prefixes_sha256']==manifest['prefixes_sha256']==sha(directory/'prefixes.json')
    data=torch.load(directory/'data.pt',map_location='cpu',weights_only=False);rows=[r for r in data if r['index']==a.index]
    prefix=[r for r in json.loads((directory/'prefixes.json').read_text()) if r['index']==a.index]
    assert all(r['role']=='evaluation_only' for r in rows+prefix)
    models={name:SourceForceScreen(name).double() for name in protocol['fixed_controls']};provenance={}
    for info in protocol['models']:
        path=a.project/info['path'];ap=a.project/info['audit'];assert sha(path)==info['sha256'] and sha(ap)==info['audit_sha256']
        audit=json.loads(ap.read_text());assert audit['complete'] and audit['model_sha256']==sha(path)
        saved=torch.load(path,map_location='cpu',weights_only=False);model=SourceForceScreen(**saved['configuration']).double();model.load_state_dict(saved['state_dict'])
        assert model.variant==info['variant'] and model.restraint==models['zero'].restraint and model.log_factor_bound==models['zero'].log_factor_bound
        models[info['name']]=model
        models[info['name']+'_thinning']=SourceForceScreen('thinning',thinning_probability=info['fit_thinning_probability']).double()
        provenance[info['name']]=dict(model_sha256=sha(path),audit_sha256=sha(ap),fit_thinning_probability=info['fit_thinning_probability'])
    for model in models.values():model.eval();model.requires_grad_(False)
    outcomes={name:evaluate(model,rows,prefix) for name,model in models.items()}
    for r in outcomes['zero']['prefixes']:assert r['expected_raw_calls']==128 and abs(r['expected_work_eV']-r['base_work_eV'])<1e-10
    result=dict(complete=True,phase=a.phase,index=a.index,protocol_sha256=sha(pp),data_sha256=header['data_sha256'],
        prefixes_sha256=header['prefixes_sha256'],provenance=provenance,attempts=len(rows),scored=sum(r['valid'] for r in rows),
        parents=len(prefix)//2,new_physical_queries=0,actual_queries_saved=0,model_fitting_allowed=False,scientific_submission_ready=False,scope=protocol['scope'])
    if a.phase=='evaluate':result['methods']=outcomes
    else:
        report=json.loads((a.run/'results.json').read_text());assert report['complete'] and report['protocol_sha256']==sha(pp)
        assert report['data_sha256']==result['data_sha256'];equal(report['provenance'],provenance);equal(report['methods'],outcomes)
        checked=0;maximum=0.
        with torch.no_grad():
            for name,model in models.items():
                for row in rows:
                    if not row['valid']:continue
                    f=independent_gate(model,row);rev=independent_gate(model,row,True)
                    u,c,g,tf,tr,accept=force_edge_values(model,row)
                    ratio=float(row['target_log_ratio']+row['log_behavior_reverse']-row['log_behavior_forward']+row['action_log_ratio'])
                    total=min(f,ratio+rev);back=min(rev,-ratio+f);assert abs(total-back-ratio)<1e-9
                    error=max(abs(float(tf)-f),abs(float(tr)-rev),abs(float(u)-float(row['reward_eV'])*math.exp(total)),abs(float(c)-2*math.exp(f)),abs(float(accept)-math.exp(total)))
                    assert error<1e-8;maximum=max(maximum,error);checked+=1
        result.update(source_results_sha256=sha(a.run/'results.json'),all_metrics_replayed=True,independent_model_pair_checks=checked,maximum_independent_error=maximum)
    if a.out.exists():raise FileExistsError(a.out)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['methods','provenance','scope']}))


if __name__=='__main__':main()
