#!/usr/bin/env python3
"""Paired finite-cost outcomes and measured training-amortized physical controls."""
import argparse
import json
from pathlib import Path
import torch
from scripts.research.evaluate_chemical_policy import sha


def bootstrap_mean(values, generator, count):
    # Average the two algorithm/training replicas within a source parent first.
    x = torch.tensor(values, dtype=torch.float64).mean(0)
    indices = torch.randint(len(x), (count,len(x)), generator=generator)
    means = x[indices].mean(1)
    interval = means.quantile(x.new_tensor([.025,.975])).tolist()
    return dict(mean=float(x.mean()), parent_bootstrap_95_percent_interval=interval, independent_source_parents=len(x))


def endpoint(events, budget, reference):
    eligible = [e for e in events if e['queries'] <= budget]
    if not eligible or eligible[-1]['queries'] != budget:
        return None
    row = dict(eligible[-1])
    row['ever_reference'] = any(e['smiles'] == reference for e in eligible)
    row['initial_reference'] = events[0]['smiles'] == reference
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    root=args.project
    pp=root/'research/evidence/fresh_reuse_protocol_v1.json'
    protocol=json.loads(pp.read_text())
    analysis_path=root/'research/evidence/fresh_reuse_analysis_plan_v1.json'
    analysis=json.loads(analysis_path.read_text())
    assert analysis['protocol_sha256']==sha(pp) and analysis['frozen_before_evaluation_submission']
    prep=root/protocol['preparation_run']
    prepared=json.loads((prep/'results.json').read_text())
    pa=json.loads((root/protocol['preparation_audit']).read_text())
    assert prepared['complete'] and pa['complete'] and pa['results_sha256']==sha(prep/'results.json')
    parent_ids=prepared['parent_ids']
    curves={}; arms=[]
    for method in protocol['methods']:
        for rep in protocol['replicas']:
            key=f'{method}_s{rep}'
            directory=root/'runs/fresh_reuse_eval_v1'/key
            audit_path=root/'runs/fresh_reuse_audit_v1'/key/'results.json'
            report=json.loads((directory/'results.json').read_text())
            audit=json.loads(audit_path.read_text())
            assert audit['complete'] and report['complete'] and audit['full_producer_replay']
            assert audit['results_sha256']==sha(directory/'results.json')
            assert report['protocol_sha256']==audit['protocol_sha256']==sha(pp)
            assert report['parent_ids']==parent_ids
            by_parent={}
            for chunk in report['chunks']:
                path=directory/chunk['file']
                assert sha(path)==chunk['sha256']
                d=torch.load(path,map_location='cpu',weights_only=False)
                for i,parent in enumerate(d['parent_ids']):
                    events=[]
                    for step,(ids,counts) in enumerate(zip(d['history_state_ids'],d['query_count_history'])):
                        state=d['states'][ids[i]]
                        row=dict(step=step,queries=counts[i],potential_eV=float(state['potential_eV']),
                            energy_eV=float(state['energy_eV']),smiles=state['graph']['connectivity_smiles'])
                        if not events or counts[i]!=events[-1]['queries']:
                            events.append(row)
                        else:
                            assert row['potential_eV']==events[-1]['potential_eV'] and row['smiles']==events[-1]['smiles']
                    by_parent[parent]=events
            assert list(by_parent)==parent_ids
            curves[key]=by_parent
            arms.append(dict(method=method,replica=rep,raw_queries=report['new_raw_queries'],
                seconds=report['seconds'],audit_sha256=sha(audit_path),
                all_query_caps_reached=all(all(c['query_cap_reached']) for c in report['chunks'])))
    generator=torch.Generator().manual_seed(protocol['bootstrap_seed'])
    reference=protocol['reference_connectivity']
    comparisons=[]
    sizes=sorted(set([n for n in protocol['reuse_prefixes'] if n<=len(parent_ids)]+[len(parent_ids)]))
    for n in sizes:
        ids=parent_ids[:n]
        for budget in protocol['quality_query_budgets']:
            for full_cost in [False,True]:
                quotient,remainder=divmod(protocol['one_time_training_raw_queries']//2,n) if full_cost else (0,0)
                site_budgets=[budget+2*quotient+2*(i<remainder) for i in range(n)]
                row=dict(parents=n,learned_queries_per_parent=budget,match_training_cost=full_cost,
                    site_queries_per_parent=site_budgets,source_attempts_retained=protocol['source_attempts'],
                    learned_total_raw_queries=protocol['one_time_training_raw_queries']+n*budget+sum(prepared['raw_queries_per_parent'][:n]),
                    site_total_raw_queries=sum(site_budgets)+sum(prepared['raw_queries_per_parent'][:n]))
                if full_cost:
                    assert row['learned_total_raw_queries']==row['site_total_raw_queries']
                learned=[];site=[];missing=[]
                for rep in protocol['replicas']:
                    learned.append([endpoint(curves[f'learned_vector_s{rep}'][i],budget,reference) for i in ids])
                    site.append([endpoint(curves[f'site_s{rep}'][i],b,reference) for i,b in zip(ids,site_budgets)])
                    for i,a,b in zip(ids,learned[-1],site[-1]):
                        if a is None or b is None:
                            missing.append(dict(replica=rep,parent_id=i,learned_missing=a is None,site_missing=b is None))
                row['available']=not missing
                if missing:
                    row['missing_endpoints']=missing
                else:
                    difference=[[a['potential_eV']-b['potential_eV'] for a,b in zip(aa,bb)] for aa,bb in zip(learned,site)]
                    row['learned_minus_site_potential_eV']=bootstrap_mean(difference,generator,protocol['bootstrap_replicates'])
                    row['by_replica']=[dict(replica=rep,
                        mean_potential_difference_eV=sum(difference[rep])/n,
                        initially_reference=sum(a['initial_reference'] for a in learned[rep]),
                        learned_new_reference_hits=sum(a['ever_reference'] and not a['initial_reference'] for a in learned[rep]),
                        site_new_reference_hits=sum(a['ever_reference'] and not a['initial_reference'] for a in site[rep]),
                        learned_terminal_reference=sum(a['smiles']==reference for a in learned[rep]),
                        site_terminal_reference=sum(a['smiles']==reference for a in site[rep]),
                        learned_terminal_distinct_graphs=len({a['smiles'] for a in learned[rep]}),
                        site_terminal_distinct_graphs=len({a['smiles'] for a in site[rep]})) for rep in protocol['replicas']]
                    row['paired_potential_differences_eV']=difference
                comparisons.append(row)
    source=json.loads((root/protocol['source_audit']).read_text())
    result=dict(complete=True,protocol_sha256=sha(pp),arms=arms,comparisons=comparisons,
        analysis_plan_sha256=sha(analysis_path),primary_definition=analysis['primary'],
        primary_comparison=next(row for row in comparisons if row['parents']==len(parent_ids)
            and row['learned_queries_per_parent']==512 and row['match_training_cost']),
        attempted_source_parents=source['attempted'],supported_source_parents=source['chemically_supported'],
        source_validity_fraction=source['chemically_supported']/source['attempted'],
        source_generation_seconds=source['generation_seconds'],source_neural_field_calls=source['neural_field_calls'],
        source_validator_errors=source['validator_errors'],preparation_raw_queries=prepared['new_raw_queries'],
        total_new_physical_queries=prepared['new_raw_queries']+sum(a['raw_queries'] for a in arms),
        new_physical_queries_in_summary=0,scientific_submission_ready=False,
        inference='Paired finite-cost readout on one development composition. Source parents are bootstrap clusters; algorithm seeds are averaged within parent. Prefix/budget intervals are descriptive and not corrected for multiple comparisons. Unsupported/censored endpoints are not dropped to produce a favorable estimate. Per-query stopping does not establish stationary sampling.')
    if args.out.exists():raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    for row in comparisons:
        if row['parents']==len(parent_ids):
            print(json.dumps({k:v for k,v in row.items() if k not in ['paired_potential_differences_eV','site_queries_per_parent','missing_endpoints']}))


if __name__=='__main__':main()
