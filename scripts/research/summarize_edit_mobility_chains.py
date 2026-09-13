#!/usr/bin/env python3
"""Matched128-query endpoints with parent-paired replicate uncertainty."""
import argparse,json,random
from pathlib import Path
import numpy as np
from scripts.research.audit_masked_angular import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','run','audit','out']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();root=a.project;pp=root/'research/evidence/edit_mobility_chain_protocol_v1.json';protocol=json.loads(pp.read_text())
    values={};cost=0;provenance={};counts={};timings={};checked=dict(local=0,rotation=0,joint=0,inverse=0);all_caps=True
    for index in protocol['condition_indices']:
        rp=a.run/f'condition_{index:02d}/results.json';ap=a.audit/f'condition_{index:02d}/results.json';r=json.loads(rp.read_text());audit=json.loads(ap.read_text())
        assert r['complete'] and audit['complete'] and audit['full_replay']
        assert r['protocol_sha256']==audit['protocol_sha256']==sha(pp) and sha(rp)==audit['source_results_sha256']
        assert sha(rp.parent/'trace.pt')==r['trace_sha256']==audit['trace_sha256']
        assert r['new_raw_queries']==r['requested_raw_queries']==audit['raw_queries_in_producer']
        cost+=r['new_raw_queries'];all_caps&=r['all_caps_reached'];provenance[str(index)]=dict(results_sha256=sha(rp),audit_sha256=sha(ap),trace_sha256=r['trace_sha256'])
        for name,key in [('local','independent_local_MH_checks'),('rotation','independent_rotation_MH_checks'),('joint','independent_joint_MH_checks'),('inverse','inverse_checks')]:checked[name]+=audit[key]
        for c in r['chains']:
            key=f"{c['method']}_s{c['replica']}";entry=counts.setdefault(key,dict(attempted=0,accepted=0,joint_accepted=0,raw_calls=0))
            for k in ['attempted','accepted','joint_accepted']:entry[k]+=c[k]
            entry['raw_calls']+=sum(c['queries_per_parent']);timings[key]=timings.get(key,0)+r['method_seconds'][key]
            for cap,rows in c['readouts'].items():
                for row in rows:values[index,row['parent'],c['method'],c['replica'],int(cap)]=row
    assert cost<=protocol['maximum_new_raw_queries'] and len(values)==12*4*2*len(protocol['readouts'])
    if not all_caps:raise ValueError('Some chains missed their cap; inspect complete failure denominators before a matched-budget summary')
    assert cost==protocol['maximum_new_raw_queries']==12288
    ordered=[];groups=[]
    for index in protocol['condition_indices']:
        parents=sorted({pid for i,pid,m,s,c in values if i==index});assert len(parents)==3
        groups.append(list(range(len(ordered),len(ordered)+3)));ordered.extend((index,pid) for pid in parents)
    rng=random.Random(28431);samples=np.array([[j for group in groups for j in rng.choices(group,k=3)] for _ in range(5000)])
    results={}
    for cap in protocol['readouts']:
        array=np.array([[[values[i,pid,m,s,cap]['potential_change_eV'] for s in protocol['replicas']] for m in protocol['methods']] for i,pid in ordered])
        assert array.shape==(12,4,2)
        means={m:dict(mean_potential_change_eV=float(array[:,j].mean()),per_replica_mean_eV=array[:,j].mean(0).tolist(),
            mean_distinct_connectivities=float(np.mean([values[i,pid,m,s,cap]['distinct_connectivities'] for i,pid in ordered for s in protocol['replicas']]))) for j,m in enumerate(protocol['methods'])}
        comparisons={}
        for method,controls in [('collective',['physical_arc','scalar','fixed']),('scalar',['physical_arc','fixed']),('fixed',['physical_arc'])]:
            for control in controls:
                delta=array[:,protocol['methods'].index(method)]-array[:,protocol['methods'].index(control)]
                bootstrap=np.sort(delta[samples].mean((1,2)))
                comparisons[method+' minus '+control]=dict(potential_change_difference_eV=float(delta.mean()),per_replica_difference_eV=delta.mean(0).tolist(),
                    descriptive_fixed_composition_parent_bootstrap95=bootstrap[[125,4875]].tolist(),lower_is_better=True)
        results[str(cap)]=dict(methods=means,comparisons=comparisons,
            per_condition={str(i):{m:float(array[[j for j,(ii,pid) in enumerate(ordered) if ii==i],k].mean()) for k,m in enumerate(protocol['methods'])} for i in protocol['condition_indices']})
    report=dict(complete=True,protocol_sha256=sha(pp),provenance=provenance,new_raw_queries=cost,all96_chains_reached128_queries=True,
        checks=checked,counts=counts,chain_loop_seconds=timings,readouts=results,per_parent=[dict(index=i,parent=pid,
            data={str(cap):{m:[values[i,pid,m,s,cap] for s in protocol['replicas']] for m in protocol['methods']} for cap in protocol['readouts']}) for i,pid in ordered],
        scientific_submission_ready=False,scope='Strict128 actual raw calls per trajectory including fresh initialization. All12 INTERNAL parents and both paired model/sampling replicas retained. Same local MALA/force-vMF background; only joint edit differs. Query-index endpoints are not equilibrium samples. Training/preparation cost, amortized generation, fresh-composition transfer and final-test claims are not established. Descriptive intervals fix four compositions and resample parents with both replicas, without multiplicity correction.')
    if a.out.exists():raise FileExistsError(a.out)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(new_raw_queries=cost,checks=checked,counts=counts,final128=results['128']),indent=2))


if __name__=='__main__':main()
