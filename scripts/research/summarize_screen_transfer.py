#!/usr/bin/env python3
"""Composition-balanced transfer summaries for frozen gate models."""
import argparse,json,random
from pathlib import Path
import numpy as np
from scripts.research.audit_masked_angular import sha
from scripts.research.summarize_source_force_screen import parents


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','run','audit','out']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[2];pp=root/'research/evidence/screen_transfer_protocol_v1.json';protocol=json.loads(pp.read_text())
    reports={};provenance={};checks=0;max_error=0.
    for index in protocol['condition_indices']:
        rp=a.run/f'condition_{index:02d}/results.json';ap=a.audit/f'condition_{index:02d}/results.json'
        r=json.loads(rp.read_text());audit=json.loads(ap.read_text())
        assert r['complete'] and audit['complete'] and audit['all_metrics_replayed']
        assert r['protocol_sha256']==audit['protocol_sha256']==sha(pp) and audit['source_results_sha256']==sha(rp)
        assert r['provenance']==audit['provenance'] and r['data_sha256']==audit['data_sha256']
        reports[index]=r;provenance[str(index)]=dict(results_sha256=sha(rp),audit_sha256=sha(ap))
        checks+=audit['independent_model_pair_checks'];max_error=max(max_error,audit['maximum_independent_error'])
    names=list(reports[0]['methods']);assert all(set(r['methods'])==set(names) for r in reports.values())
    values={};resampled={}
    for index,r in reports.items():
        groups={name:parents(r['methods'][name]) for name in names};keys=sorted(groups['zero'])
        assert len(keys)==r['parents'] and all(set(v)==set(keys) for v in groups.values())
        rng=random.Random(26051+index);draws=np.array([rng.choices(range(len(keys)),k=len(keys)) for _ in range(5000)])
        for name,rows in groups.items():
            u=np.array([rows[k]['expected_work_eV'] for k in keys]);c=np.array([rows[k]['expected_raw_calls'] for k in keys])
            values[index,name]=(float(u.mean()),float(c.mean()))
            resampled[index,name]=(u[draws].mean(1),c[draws].mean(1))
    sections={}
    for group,indices in {'all':[0,1,2,3,5,7],'seen_compositions':[1,2,3,5],'unseen_compositions':[0,7]}.items():
        rates={};boot={};metrics={}
        for name in names:
            u=np.mean([values[i,name][0] for i in indices]);c=np.mean([values[i,name][1] for i in indices])
            rates[name]=float(u/c);boot[name]=np.mean([resampled[i,name][0] for i in indices],axis=0)/np.mean([resampled[i,name][1] for i in indices],axis=0)
            metrics[name]=dict(expected_work_eV=float(u),expected_raw_calls=float(c),utility_eV_per_expected_raw_call=rates[name])
        pairs=[('physical','zero'),('work','zero')]
        for info in protocol['models']:
            name=info['name'];pairs += [(name,c) for c in ['zero','physical','work',name+'_thinning']]
        for replica in range(2):
            for recipe in ['direct','distill']:pairs.append((f'{recipe}_neural_{replica}',f'{recipe}_linear_{replica}'))
            for variant in ['linear','neural']:pairs.append((f'distill_{variant}_{replica}',f'direct_{variant}_{replica}'))
        comparisons=[]
        for method,control in pairs:
            diff=np.sort(boot[method]-boot[control]);before,after=rates[control],rates[method]
            comparisons.append(dict(method=method,control=control,method_rate=after,control_rate=before,difference_eV_per_expected_raw_call=after-before,
                relative_difference=after/before-1 if before>0 else None,fixed_composition_parent_bootstrap95=[float(diff[125]),float(diff[4875])]))
        sections[group]=dict(condition_indices=indices,parents=sum(reports[i]['parents'] for i in indices),metrics=metrics,comparisons=comparisons)
    result=dict(complete=True,protocol_sha256=sha(pp),provenance=provenance,sections=sections,
        per_condition={str(i):{name:dict(expected_work_eV=values[i,name][0],expected_raw_calls=values[i,name][1],utility_eV_per_expected_raw_call=values[i,name][0]/values[i,name][1]) for name in names} for i in reports},
        attempts=sum(r['attempts'] for r in reports.values()),scored=sum(r['scored'] for r in reports.values()),
        independent_model_pair_checks=checks,maximum_independent_error=max_error,new_physical_queries=0,actual_queries_saved=0,
        model_fitting_allowed=False,scientific_submission_ready=False,
        scope='Reused INTERNAL48-parent cohort excluded from gate fitting and original12-parent selection. Six fixed compositions are weighted equally despite unequal parent counts; bootstrap resamples parents within each composition and retains both trajectories. Two composition identities were absent from fitting. Not an untouched final test, changed chain, actual query saving or population-level unseen-chemistry guarantee.')
    if a.out.exists():raise FileExistsError(a.out)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    for group,section in sections.items():
        print(group)
        for row in section['comparisons']:
            if row['control']=='physical' and 'neural' in row['method']:print(json.dumps(row))


if __name__=='__main__':main()
