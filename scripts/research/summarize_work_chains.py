#!/usr/bin/env python3
"""Complete matched-budget exploration/energy results with strong controls."""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.research.audit_masked_angular import sha
from scripts.research.evaluate_chemical_policy import write


METRICS=['distinct_connectivities','potential_change_eV','connectivity_transitions','connectivity_returns',
         'final_typed_distance_change_A2','max_typed_distance_change_A2','accepted_uphill_moves','accepted_joint_moves','type_fallback_attempts']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run','audit','out','protocol']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();protocol=json.loads(a.protocol.read_text());values={};timing={};loop={};sources={};cost=0;checks={};attempts={}
    for index in protocol['condition_indices']:
        rp=a.run/f'condition_{index:02d}/results.json';ap=a.audit/f'condition_{index:02d}/results.json'
        r=json.loads(rp.read_text());audit=json.loads(ap.read_text())
        assert r['complete'] and audit['complete'] and audit['full_replay'] and r['all_caps_reached']
        assert r['protocol_sha256']==audit['protocol_sha256']==sha(a.protocol)
        assert sha(rp)==audit['source_results_sha256'] and sha(rp.parent/'trace.pt')==r['trace_sha256']==audit['trace_sha256']
        assert r['new_raw_queries']==r['requested_raw_queries']==audit['raw_queries_in_producer']
        cost+=r['new_raw_queries'];sources[str(index)]=dict(results_sha256=sha(rp),audit_sha256=sha(ap),trace_sha256=r['trace_sha256'])
        for key in ['independent_local_MH_checks','independent_rotation_MH_checks','independent_joint_MH_checks','independent_normalized_joint_MH_checks',
                    'independent_catalogue_normalizations','independent_history_steps','inverse_checks']:
            checks[key]=checks.get(key,0)+audit[key]
        for key,seconds in r['method_seconds'].items():timing[key]=timing.get(key,0)+seconds
        for key,seconds in r['loop_seconds'].items():loop[key]=loop.get(key,0)+seconds
        for chain in r['chains']:
            method,replica=chain['method'],chain['replica'];key=f'{method}_s{replica}'
            entry=attempts.setdefault(key,dict(attempted=0,accepted=0,joint_accepted=0,raw_queries=0))
            for field in ['attempted','accepted','joint_accepted']:entry[field]+=chain[field]
            entry['raw_queries']+=sum(chain['queries_per_parent'])
            assert all(chain['cap_reached']) and all(q==protocol['query_cap_per_parent'] for q in chain['queries_per_parent'])
            for cap,rows in chain['readouts'].items():
                for row in rows:
                    assert row['reached'];values[index,row['parent'],method,replica,int(cap)]=row
    methods=protocol['methods'];replicas=protocol['replicas'];caps=protocol['readouts']
    parents=sorted({(i,parent) for i,parent,m,s,c in values});assert len(parents)==18
    assert cost==protocol['maximum_new_raw_queries']==18*len(methods)*len(replicas)*protocol['query_cap_per_parent']
    assert len(values)==18*len(methods)*len(replicas)*len(caps)
    rng=np.random.default_rng(28891);groups=[[j for j,(i,pid) in enumerate(parents) if i==index] for index in protocol['condition_indices']]
    assert all(len(g)==3 for g in groups)
    sample=np.concatenate([rng.choice(g,size=(10000,3),replace=True) for g in groups],axis=1)
    # Publish every pair, avoiding selection of a favorable control after results.
    pairs=[(left,right) for i,left in enumerate(methods) for right in methods[:i]]
    readouts={}
    for cap in caps:
        arrays={metric:np.array([[[values[i,pid,m,s,cap][metric] for s in replicas] for m in methods] for i,pid in parents],dtype=float) for metric in METRICS}
        means={method:{metric:dict(mean=float(x[:,j].mean()),per_replica=x[:,j].mean(0).tolist()) for metric,x in arrays.items()} for j,method in enumerate(methods)}
        comparisons={}
        for left,right in pairs:
            result={}
            for metric,x in arrays.items():
                delta=x[:,methods.index(left)]-x[:,methods.index(right)]
                result[metric]=dict(mean_difference=float(delta.mean()),per_replica_difference=delta.mean(0).tolist(),
                    descriptive_fixed_composition_parent_bootstrap95=np.quantile(delta[sample].mean((1,2)),[.025,.975]).tolist())
            comparisons[left+' minus '+right]=result
        readouts[str(cap)]=dict(methods=means,comparisons=comparisons)
    runtime={m:dict(total_seconds=sum(timing[f'{m}_s{s}'] for s in replicas),seconds_per_trajectory=sum(timing[f'{m}_s{s}'] for s in replicas)/(18*len(replicas)),
        per_replica_seconds=[timing[f'{m}_s{s}'] for s in replicas],new_inference_raw_queries=sum(attempts[f'{m}_s{s}']['raw_queries'] for s in replicas),
        marginal_training_label_calls=protocol['additional_label_queries'][m]) for m in methods}
    if a.out.exists():raise FileExistsError(a.out)
    report=dict(complete=True,protocol_sha256=sha(a.protocol),sources=sources,trajectories=18*len(methods)*len(replicas),all_caps_reached=True,
        new_raw_queries=cost,checks=checks,counts=attempts,runtime=runtime,loop_seconds=loop,readouts=readouts,
        parents=[dict(index=i,parent=pid,data={str(cap):{m:[values[i,pid,m,s,cap] for s in replicas] for m in methods} for cap in caps}) for i,pid in parents],
        scientific_submission_ready=False,scope=protocol['interpretation'],uncertainty='Descriptive parent bootstrap within six fixed compositions, keeping both replicas together, all15 pairwise method comparisons shown, no multiplicity correction or final-test claim.')
    write(a.out,report)
    final=readouts[str(caps[-1])]['methods']
    print(json.dumps(dict(trajectories=report['trajectories'],new_raw_queries=cost,
        final={m:{k:final[m][k]['mean'] for k in ['distinct_connectivities','potential_change_eV']} for m in methods})))


if __name__=='__main__':main()
