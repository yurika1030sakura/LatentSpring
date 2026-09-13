#!/usr/bin/env python3
"""Summarize actual edit proposals, all failures, query costs and paired contrasts."""
import argparse,json,random
from pathlib import Path
import numpy as np
from scripts.research.audit_masked_angular import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','run','audit','out']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();root=a.project;pp=root/'research/evidence/edit_bridge_evaluation_protocol_v1.json';protocol=json.loads(pp.read_text())
    methods=protocol['methods'];parents={};provenance={};calls=checks=inverses=0;timing={m:dict(proposal_seconds=0.,physical_dispatch_seconds=0.) for m in methods}
    counts={m:dict(attempts=0,geometry_valid=0,scored=0,accepted=0,candidate_raw_calls=0) for m in methods}
    for index in protocol['condition_indices']:
        rp=a.run/f'condition_{index:02d}/results.json';ap=a.audit/f'condition_{index:02d}/results.json';r=json.loads(rp.read_text());audit=json.loads(ap.read_text())
        assert r['complete'] and audit['complete'] and audit['full_replay']
        assert r['protocol_sha256']==audit['protocol_sha256']==sha(pp) and sha(rp)==audit['source_results_sha256']
        assert sha(rp.parent/'trace.pt')==r['trace_sha256']==audit['trace_sha256']
        assert r['new_raw_queries']==r['requested_raw_queries']==audit['raw_queries_in_producer']
        assert r['initial_raw_queries']==6
        calls+=r['new_raw_queries'];checks+=audit['independent_MH_checks'];inverses+=audit['trained_and_control_inverse_checks']
        provenance[str(index)]=dict(results_sha256=sha(rp),audit_sha256=sha(ap),trace_sha256=r['trace_sha256'])
        for method in methods:
            selected=[v for v in r['rows'] if v['method']==method];assert len(selected)==3*protocol['trials_per_source']
            c=counts[method];c['attempts']+=len(selected);c['geometry_valid']+=sum(v['valid'] for v in selected);c['scored']+=sum(v['scored'] for v in selected);c['accepted']+=sum(v['accepted'] for v in selected);c['candidate_raw_calls']+=sum(v['raw_cost'] for v in selected)
            for key in timing[method]:timing[method][key]+=r['method_timing'][method][key]
            for parent in sorted({v['parent'] for v in selected}):
                group=[v for v in selected if v['parent']==parent];assert len(group)==protocol['trials_per_source']
                parents[index,parent,method]=dict(expected_utility_eV=sum(v['expected_utility_eV'] for v in group),charged_raw_calls=2+sum(v['raw_cost'] for v in group),
                    expected_constitutional_flow=sum(v['expected_constitutional_flow'] for v in group),expected_invariant_jump_A2=sum(v['expected_invariant_jump_A2'] for v in group))
    assert calls<=protocol['maximum_new_raw_queries'] and calls==24+sum(v['candidate_raw_calls'] for v in counts.values())
    assert checks==sum(v['scored'] for v in counts.values())
    keys=['expected_utility_eV','charged_raw_calls','expected_constitutional_flow','expected_invariant_jump_A2']
    group_indices=[];ordered=[]
    for index in protocol['condition_indices']:
        ids=sorted({pid for i,pid,m in parents if i==index});assert len(ids)==3
        group_indices.append(list(range(len(ordered),len(ordered)+3)));ordered.extend((index,pid) for pid in ids)
    array=np.array([[[parents[i,pid,m][k] for k in keys] for m in methods] for i,pid in ordered]);assert array.shape==(12,9,4)
    def metrics(values):
        result=dict(zip(keys,map(float,values.sum(0))))
        result.update(utility_eV_per_charged_raw_call=result['expected_utility_eV']/result['charged_raw_calls'],
            constitutional_flow_per_charged_raw_call=result['expected_constitutional_flow']/result['charged_raw_calls'],
            invariant_jump_A2_per_charged_raw_call=result['expected_invariant_jump_A2']/result['charged_raw_calls'])
        return result
    points={m:metrics(array[:,j]) for j,m in enumerate(methods)}
    rng=random.Random(28241);draws=np.array([[j for group in group_indices for j in rng.choices(group,k=3)] for _ in range(5000)])
    totals=array[draws].sum(1);rates=totals[:,:,0]/totals[:,:,1]
    comparisons={}
    for method in methods:
        if method in ['physical_arc','zero_bridge','analytic_bridge']:continue
        controls=['physical_arc','zero_bridge','analytic_bridge']
        if method.startswith('edit_collective'):
            suffix=method.rsplit('_',1)[1];controls+=['blind_collective_'+suffix,'edit_roots_'+suffix]
        for control in controls:
            delta=rates[:,methods.index(method)]-rates[:,methods.index(control)];interval=np.sort(delta)[[125,4875]].tolist()
            point=points[method]['utility_eV_per_charged_raw_call']-points[control]['utility_eV_per_charged_raw_call']
            comparisons[method+' minus '+control]=dict(difference_eV_per_charged_raw_call=point,descriptive_fixed_composition_parent_bootstrap95=interval)
    result=dict(complete=True,protocol_sha256=sha(pp),provenance=provenance,new_raw_queries=calls,independent_MH_checks=checks,inverse_checks=inverses,
        counts=counts,methods=points,comparisons=comparisons,method_timing_components=timing,
        per_condition={str(i):{m:metrics(array[[j for j,(ii,pid) in enumerate(ordered) if ii==i],k]) for k,m in enumerate(methods)} for i in protocol['condition_indices']},
        per_parent=[dict(index=i,parent=pid,methods={m:parents[i,pid,m] for m in methods}) for i,pid in ordered],scientific_submission_ready=False,
        scope='Fixed32 attempted proposals on each of12 INTERNAL sources. All failures retained. Each method is charged shared source initialization once; physical research queries share it. Ratios are one-step fixed-source efficiency, not exactly matched-cost trajectories, mixing or amortized generation superiority. Training/preparation costs remain additional. Timings are proposal and oracle dispatch components, not complete end-to-end generator runtime. Intervals fix four compositions and resample parents, without multiplicity correction.')
    if a.out.exists():raise FileExistsError(a.out)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['new_raw_queries','counts','methods','comparisons']},indent=2))


if __name__=='__main__':main()
