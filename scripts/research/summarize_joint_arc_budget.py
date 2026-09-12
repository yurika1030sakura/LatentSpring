#!/usr/bin/env python3
"""Summarize all audited TRAINING joint-arc chains without pseudo-replicating seeds."""
import argparse
import hashlib
import json
from pathlib import Path
import random
from statistics import mean


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_summary(per_condition, seed=25471, samples=4000):
    """Resample parents within each fixed composition; seeds were averaged first."""
    rows=[values for values in per_condition.values() if values]
    if len(rows)!=len(per_condition): return dict(available=False)
    rng=random.Random(seed)
    draws=sorted(mean(mean(rng.choices(values,k=len(values))) for values in rows) for _ in range(samples))
    return dict(available=True,mean=mean(mean(values) for values in rows),
        fixed_composition_parent_bootstrap_95_percent_interval=[draws[int(.025*samples)],draws[int(.975*samples)]],
        parents=sum(map(len,rows)),per_condition_mean={k:mean(v) for k,v in per_condition.items()},
        inference_scope='Descriptive TRAINING comparison; four composition identities fixed, algorithm seeds averaged within parent. Not an unseen-composition generalization interval.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','run','audit','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--protocol',type=Path)
    p.add_argument('--recovery-run',type=Path)
    args=p.parse_args();root=Path(__file__).resolve().parents[2]
    pp=args.protocol or root/'research/evidence/joint_arc_budget_protocol_v1.json';protocol=json.loads(pp.read_text())
    arms={};evidence={};source_cost=0;timing={};counts={};startup={};loading={}
    for index in protocol['condition_indices']:
        rp=args.run/f'condition_{index:02d}/results.json';ap=args.audit/f'condition_{index:02d}/results.json'
        if args.recovery_run is not None and index==5:
            old=rp
            rp=args.recovery_run/'results.json'
            recovered=json.loads(rp.read_text())
            assert recovered['recovery']['source_results_sha256']==sha(old)
            assert recovered['recovery']['all_cached_requests_replayed']
            assert recovered['new_raw_queries']==recovered['recovery']['previous_physical_calls']+recovered['recovery']['additional_physical_calls']
        report=json.loads(rp.read_text());audit=json.loads(ap.read_text())
        assert report['complete'] and audit['complete'] and audit['full_producer_replay']
        if 'recovery' in report: assert audit['recovery_prefix_verified']
        assert report['protocol_sha256']==audit['protocol_sha256']==sha(pp)
        assert audit['producer_results_sha256']==sha(rp)
        assert report['parent_ids']==audit['parent_ids']==protocol['parent_ids_by_condition'][str(index)]
        assert report['new_raw_queries']==report['requested_raw_queries']==audit['raw_queries_in_producer']
        assert audit['new_raw_queries']==0
        assert len(report['arms'])==len(audit['arms'])==len(protocol['methods'])*len(protocol['replicas'])
        source_cost+=report['new_raw_queries'];evidence[str(index)]=dict(producer_sha256=sha(rp),audit_sha256=sha(ap))
        startup[str(index)]=report['oracle_startup_seconds']
        for row,check in zip(report['arms'],audit['arms']):
            assert row['name']==check['name'] and row['raw_queries']==check['raw_queries']
            assert row['trace_sha256']==check['trace_sha256']==sha(rp.parent/row['file'])
            assert check['full_replay'] and check['independent_per_parent_ledger']
            key=(index,row['method'],row['replica']);assert key not in arms
            arms[key]=row
            timing.setdefault(row['method'],[]).append(row['sampling_seconds'])
            loading.setdefault(row['method'],[]).append(row.get('model_loading_seconds',0.))
            for kind,values in row['transition_counts'].items():
                dest=counts.setdefault(row['method'],{}).setdefault(kind,dict(attempted=0,scored=0,accepted=0))
                for name,value in values.items():dest[name]+=value
    assert len(arms)==len(protocol['condition_indices'])*len(protocol['methods'])*len(protocol['replicas'])
    assert source_cost<=protocol['maximum_total_new_raw_queries']
    rows_by_parent={}
    for (index,method,replica),arm in arms.items():
        for row in arm['endpoints']:
            key=(index,method,replica,row['parent'],row.get('readout',row['raw_queries']))
            assert key not in rows_by_parent;rows_by_parent[key]=row
    metrics=['potential_change_eV','unique_visited_smiles','final_graph_changed']
    readouts={};censored=[]
    for cap in protocol['readout_raw_queries']:
        aggregates={};pair_means={};groups={}
        for index in protocol['condition_indices']:
            for parent in protocol['parent_ids_by_condition'][str(index)]:
                rows=[rows_by_parent[index,m,s,parent,cap] for m in protocol['methods'] for s in protocol['replicas']]
                if not all(r['reached'] for r in rows):
                    censored.append(dict(index=index,parent=parent,cap=cap,failed=[dict(method=m,replica=s)
                        for m in protocol['methods'] for s in protocol['replicas'] if not rows_by_parent[index,m,s,parent,cap]['reached']]))
                    continue
                for method in protocol['methods']:
                    groups[index,method,parent]={k:mean(rows_by_parent[index,method,s,parent,cap][k] for s in protocol['replicas']) for k in metrics}
        for method in protocol['methods']:
            per_case={str(i):{k:mean(v[k] for (idx,m,parent),v in groups.items() if idx==i and m==method)
                for k in metrics} for i in protocol['condition_indices'] if any(idx==i for idx,m,parent in groups)}
            aggregates[method]=dict(per_condition=per_case,
                equal_condition_mean={k:mean(v[k] for v in per_case.values()) for k in metrics} if len(per_case)==len(protocol['condition_indices']) else None)
        comparisons=([('learned_work','arc_site'),('learned_work_force','arc_site'),('learned_work_force','learned_work')]
            if 'model_methods' in protocol else [('arc_uniform','legacy_site64'),('arc_site','legacy_site64'),('arc_site','arc_uniform')])
        for left,right in comparisons:
            values={str(i):[groups[i,left,parent]['potential_change_eV']-groups[i,right,parent]['potential_change_eV']
                for parent in protocol['parent_ids_by_condition'][str(i)] if (i,left,parent) in groups and (i,right,parent) in groups]
                for i in protocol['condition_indices']}
            pair_means[left+'_minus_'+right]=paired_summary(values)
            if 'model_methods' in protocol:pair_means[left+'_minus_'+right]['inference_scope']='INTERNAL development: composition identities fixed, seeds averaged within parent; descriptive unadjusted parent intervals. Not a reserved final benchmark.'
        readouts[str(cap)]=dict(methods=aggregates,paired_potential_change_eV=pair_means)
    result=dict(complete=True,protocol_sha256=sha(pp),evidence=evidence,
        new_raw_queries_in_experiment=source_cost,new_physical_queries_in_summary=0,
        transition_counts=counts,sampling_seconds_by_method={m:dict(total=sum(v),arms=len(v),mean=mean(v)) for m,v in timing.items()},
        oracle_startup_seconds_by_condition=startup,readouts=readouts,censored_parent_readouts=censored,
        all_48_parents_at_all_caps=not censored,zero_source_conditions_retained=protocol['zero_support_conditions_retained'],
        source_attempts_all_eight_conditions=protocol['all_eight_source_attempts_retained'],
        common_preparation_raw_queries_all_96_parents=protocol['common_preparation_raw_queries_all_96_parents'],
        no_new_model_fitted=True,scientific_submission_ready=False,
        scope=protocol.get('scope','Four fixed TRAINING FIT compositions only,48 parents,two algorithm seeds. Complete-chain finite-query progress. No learned advantage, equilibrium or unseen-composition superiority claim.'))
    if 'model_methods' in protocol:
        budgets={}
        for method in protocol['methods']:
            for replica in protocol['replicas']:
                selected=[row for (index,m,s),row in arms.items() if m==method and s==replica]
                inference=sum(row['raw_queries'] for row in selected)
                overhead=protocol['model_specific_raw_overhead_per_method_replica'] if method in protocol['model_methods'] else 0
                total=inference+overhead
                reached=all(row['all_caps_reached'] for row in selected)
                if reached:assert total==protocol['primary_total_raw_calls_per_method_replica']
                budgets[f'{method}_s{replica}']=dict(inference_raw_calls=inference,model_specific_data_raw_calls=overhead,
                    comparable_total_raw_calls=total,shared_preparation_raw_calls=protocol['common_selected_preparation_raw_queries'],
                    total_with_shared_preparation=total+protocol['common_selected_preparation_raw_queries'],all_caps_reached=reached)
        result.update(method_replica_costs=budgets,all_primary_budgets_matched=all(v['all_caps_reached'] for v in budgets.values()),
            model_loading_seconds_by_method={m:sum(v) for m,v in loading.items()},
            final_readout='Physical parent-specific436/438 versus learned128; exact total raw calls including model data costs. Same-inference readouts remain separate.')
    if args.recovery_run is not None:
        recovered=json.loads((args.recovery_run/'results.json').read_text())
        result['recovery']=dict(recovered['recovery'],results_sha256=sha(args.recovery_run/'results.json'),
            source_run=str(args.recovery_run),per_method_timing_complete=False,
            interpretation='The recovered partial arm sampling_seconds excludes its earlier failed-attempt time; previous elapsed time is retained in recovery provenance. No matched-wall-time claim is allowed. Raw-call accounting reuses the saved prefix once.')
    if args.out.exists():raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['readouts'].get('final',result['readouts']['128']),indent=2))


if __name__=='__main__':main()
