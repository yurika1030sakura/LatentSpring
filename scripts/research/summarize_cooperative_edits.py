#!/usr/bin/env python3
"""Complete audited cooperative-policy comparisons; retain every parent."""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.research.audit_masked_angular import sha
from scripts.research.evaluate_chemical_policy import write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run','audit','out','protocol']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();protocol=json.loads(a.protocol.read_text());rows=[];sources={};new=reused=trace_calls=checks=0
    for i in [0,1,2,3,5,7]:
        directory=a.run/f'condition_{i:02d}';ap=a.audit/f'condition_{i:02d}/results.json'
        report=json.loads((directory/'results.json').read_text());audit=json.loads(ap.read_text())
        assert report['complete'] and audit['complete'] and audit['full_replay']
        assert report['protocol_sha256']==audit['protocol_sha256']==sha(a.protocol)
        assert sha(directory/'results.json')==audit['source_results_sha256']
        assert sha(directory/'trace.pt')==report['trace_sha256']==audit['trace_sha256']
        assert report.get('physical_calls_in_trace',report['new_raw_queries'])==audit['producer_raw_queries']==protocol['counts'][str(i)]['raw_queries']
        new+=report['new_raw_queries'];reused+=report.get('reused_raw_queries',0);trace_calls+=audit['producer_raw_queries'];checks+=audit['independent_MH_checks']
        rows.extend(dict(index=i,**row) for row in report['rows'])
        sources[str(i)]=dict(results_sha256=sha(directory/'results.json'),audit_sha256=sha(ap),trace_sha256=sha(directory/'trace.pt'))
    assert len(rows)==18 and new==protocol['maximum_new_raw_queries']
    assert trace_calls==protocol.get('maximum_trace_raw_queries',protocol['maximum_new_raw_queries'])
    methods=list(rows[0]['methods']);metrics=list(rows[0]['methods'][methods[0]])
    means={name:{k:float(np.mean([row['methods'][name][k] for row in rows])) for k in metrics} for name in methods}
    values={name:np.array([row['methods'][name]['utility_eV'] for row in rows]) for name in methods}
    families=['linear','restraint','linear_interaction','blind_interaction','environment_interaction']
    for family in families:values[family]=(values[family+'_s0']+values[family+'_s1'])/2
    rng=np.random.default_rng(28791);boot=rng.integers(0,18,size=(10000,18));comparisons={}
    for left,right in [('linear','uniform'),('restraint','linear'),('linear_interaction','restraint'),('blind_interaction','restraint'),
                       ('environment_interaction','restraint'),('environment_interaction','linear'),
                       ('environment_interaction','linear_interaction'),('environment_interaction','blind_interaction')]:
        delta=values[left]-values[right]
        replicas=[]
        for s in [0,1]:
            aa=values.get(left+f'_s{s}',values[left]);bb=values.get(right+f'_s{s}',values[right]);replicas.append(float((aa-bb).mean()))
        comparisons[left+' minus '+right]=dict(mean_utility_difference_eV=float(delta.mean()),per_seed_difference_eV=replicas,
            descriptive_parent_bootstrap95=np.quantile(delta[boot].mean(1),[.025,.975]).tolist(),parent_differences_eV=delta.tolist())
    if a.out.exists():raise FileExistsError(a.out)
    write(a.out,dict(complete=True,protocol_sha256=sha(a.protocol),sources=sources,parents=rows,mean_one_step=means,comparisons=comparisons,
        new_raw_queries=new,reused_raw_queries=reused,physical_calls_in_trace=trace_calls,independent_MH_checks=checks,scientific_submission_ready=False,
        scope=protocol['interpretation'],uncertainty='Descriptive bootstrap of18 repeatedly evaluated development parents, preserving empty parents and averaging model seeds within a parent.'))


if __name__=='__main__':main()
