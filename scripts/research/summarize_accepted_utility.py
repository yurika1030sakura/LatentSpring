#!/usr/bin/env python3
"""Parent-cluster uncertainty for the audited empirical accepted-utility pilot."""
import argparse,json,random
from pathlib import Path
from statistics import mean
from scripts.research.audit_masked_angular import sha


def parent_metrics(edges):
    result={}
    for index,parent in sorted({(e['index'],e['parent']) for e in edges}):
        group=[e for e in edges if e['index']==index and e['parent']==parent]
        values={}
        for key in ['utility_eV','expected_raw_cost','accepted_constitutional_flow']:
            values[key]=mean(mean(e[key] for e in group if e['replica']==replica) for replica in sorted({e['replica'] for e in group}))
        result[index,parent]=values
    return result


def aggregate(rows,selected):
    utility=mean(mean(rows[index,parent]['utility_eV'] for parent in parents) for index,parents in selected.items())
    cost=mean(mean(rows[index,parent]['expected_raw_cost'] for parent in parents) for index,parents in selected.items())
    return utility/cost


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','out']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();ap=args.project/'runs/accepted_utility_train_audit_v1/results.json'
    audit=json.loads(ap.read_text());assert audit['complete'] and audit['split_rebuilt']
    results=[]
    for row in audit['rows']:
        path=args.project/f"runs/accepted_utility_train_v1/replica_{row['replica']}/results.json"
        assert sha(path)==row['results_sha256'];report=json.loads(path.read_text())
        base=parent_metrics(report['baseline_validation']['edges']);final=parent_metrics(report['final_validation']['edges'])
        selected={i:sorted(p for ii,p in base if ii==i) for i in sorted({i for i,p in base})}
        assert len(base)==12 and all(len(v)==3 for v in selected.values())
        before=aggregate(base,selected);after=aggregate(final,selected)
        rng=random.Random(25631);values=[]
        for _ in range(5000):
            sample={i:rng.choices(parents,k=len(parents)) for i,parents in selected.items()}
            values.append(aggregate(final,sample)-aggregate(base,sample))
        values.sort()
        results.append(dict(replica=row['replica'],baseline_eV_per_expected_raw_call=before,final_eV_per_expected_raw_call=after,
            difference_eV_per_expected_raw_call=after-before,relative_gain=after/before-1,
            fixed_composition_parent_bootstrap95=[values[125],values[4875]],
            maximum_scored_importance=report['final_validation']['metrics']['maximum_scored_importance'],
            baseline_scored_edge_ESS=report['baseline_validation']['metrics']['weighted_scored_edge_ESS'],
            final_scored_edge_ESS=report['final_validation']['metrics']['weighted_scored_edge_ESS'],
            per_condition={str(i):dict(baseline=aggregate(base,{i:pids}),final=aggregate(final,{i:pids})) for i,pids in selected.items()}))
    output=dict(complete=True,audit_sha256=sha(ap),rows=results,new_physical_queries=0,scientific_submission_ready=False,
        scope='12 internal selection parents, four fixed compositions. Resample parents jointly for baseline/final and retain both behavioral trajectories. Confidence describes the empirical one-step proxy, not independently measured molecular performance or final-test generalization.')
    if args.out.exists():raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))


if __name__=='__main__':main()
