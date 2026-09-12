#!/usr/bin/env python3
"""Summarize all six frozen composition-transfer tests at one global cost budget."""
import argparse
import json
from pathlib import Path
import torch
from cfm_mol.hierarchical_uncertainty import hierarchical_paired_mean
from scripts.research.evaluate_chemical_policy import sha
from scripts.research.summarize_fresh_reuse import endpoint


def curves_from_trace(saved):
    result={}
    for i,parent in enumerate(saved['parent_ids']):
        events=[]
        for ids,counts in zip(saved['history_state_ids'],saved['query_count_history']):
            state=saved['states'][ids[i]]
            row=dict(queries=counts[i],potential_eV=float(state['potential_eV']),
                smiles=state['graph']['connectivity_smiles'])
            if not events or counts[i]!=events[-1]['queries']:
                events.append(row)
            else:
                assert row['potential_eV']==events[-1]['potential_eV'] and row['smiles']==events[-1]['smiles']
        result[parent]=events
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','run','audits','out']:
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();root=args.project
    mp=root/'research/evidence/transfer_reuse_protocol_v1.json';master=json.loads(mp.read_text())
    assert master['frozen'] and master['total_parents']==96 and master['parents_per_condition']==16
    cases=[];preparation_calls=0;sampling_calls=0
    for index in master['conditions']:
        cp=root/f'research/evidence/transfer_reuse_case_{index:02d}.json'
        assert sha(cp)==master['case_protocol_sha256'][str(cp.relative_to(root))]
        protocol=json.loads(cp.read_text());prep=root/protocol['preparation_run']
        prepared=json.loads((prep/'results.json').read_text())
        pa=json.loads((root/protocol['preparation_audit']).read_text())
        assert prepared['complete'] and pa['complete'] and pa['results_sha256']==sha(prep/'results.json')
        assert prepared['protocol_sha256']==sha(cp) and len(prepared['parent_ids'])==16
        preparation_calls+=prepared['new_raw_queries']
        curves={};arms=[]
        for method in master['methods']:
            for replica in master['replicas']:
                key=f'{method}_s{replica}';directory=args.run/f'condition_{index:02d}'/key
                ap=args.audits/f'condition_{index:02d}'/key/'results.json'
                report=json.loads((directory/'results.json').read_text());audit=json.loads(ap.read_text())
                assert report['complete'] and audit['complete'] and audit['full_producer_replay']
                assert report['protocol_sha256']==audit['protocol_sha256']==sha(cp)
                assert audit['results_sha256']==sha(directory/'results.json')
                assert report['preparation_results_sha256']==sha(prep/'results.json')
                assert report['parent_ids']==prepared['parent_ids']
                assert len(report['chunks'])==1
                path=directory/report['chunks'][0]['file'];assert sha(path)==report['chunks'][0]['sha256']
                data=torch.load(path,map_location='cpu',weights_only=False)
                curves[key]=curves_from_trace(data)
                assert list(curves[key])==prepared['parent_ids']
                arms.append(dict(method=method,replica=replica,raw_queries=report['new_raw_queries'],seconds=report['seconds'],
                    audit_sha256=sha(ap),query_cap_reached=report['chunks'][0]['query_cap_reached']))
                sampling_calls+=report['new_raw_queries']
        cases.append(dict(index=index,parent_ids=prepared['parent_ids'],condition=prepared['condition'],
            source_attempts=prepared['source_attempts'],source_supported=prepared['source_supported'],
            preparation_raw_queries=prepared['new_raw_queries'],arms=arms,curves=curves))
    comparisons=[]
    for full_cost in [True,False]:
        differences=[[],[]];missing=[];details=[];site_total=0
        q,r=divmod(master['one_time_training_raw_queries']//2,96) if full_cost else (0,0)
        for case in cases:
            budgets=[512+2*q+2*(16*case['index']+i<r) for i in range(16)]
            site_total+=sum(budgets)
            by_rep=[]
            for rep in master['replicas']:
                a=[endpoint(case['curves'][f'learned_vector_s{rep}'][pid],512,None) for pid in case['parent_ids']]
                b=[endpoint(case['curves'][f'site_s{rep}'][pid],budget,None) for pid,budget in zip(case['parent_ids'],budgets)]
                if any(v is None for v in a+b):
                    missing.append(dict(condition_index=case['index'],replica=rep))
                else:
                    diff=[x['potential_eV']-y['potential_eV'] for x,y in zip(a,b)]
                    differences[rep].append(diff)
                    by_rep.append(dict(replica=rep,paired_differences_eV=diff,mean_difference_eV=sum(diff)/16,
                        learned_distinct_graphs=len({x['smiles'] for x in a}),site_distinct_graphs=len({x['smiles'] for x in b})))
            details.append(dict(index=case['index'],site_query_budgets=budgets,replicas=by_rep))
        row=dict(match_training_cost=full_cost,available=not missing,missing_endpoints=missing,
            learned_total_raw_queries=9660+96*512+preparation_calls,site_total_raw_queries=site_total+preparation_calls,conditions=details)
        if full_cost:assert row['learned_total_raw_queries']==row['site_total_raw_queries']
        if not missing:
            row['paired_potential_difference_eV']=hierarchical_paired_mean(differences,
                generator=torch.Generator().manual_seed(master['bootstrap_seed']+(0 if full_cost else 1)),replicates=master['bootstrap_replicates'])
        comparisons.append(row)
    result=dict(complete=True,master_protocol_sha256=sha(mp),primary_comparison=comparisons[0],secondary_same_inference=comparisons[1],
        source_conditions=[{k:v for k,v in c.items() if k!='curves'} for c in cases],
        total_new_physical_queries=preparation_calls+sampling_calls,new_physical_queries_in_summary=0,
        scientific_submission_ready=False,scope='Six additional development compositions; fixed learned weights, hierarchical composition/parent uncertainty, finite-query endpoints and global one-time training cost. No equilibrium or universal chemistry claim.')
    if args.out.exists():raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result['primary_comparison'].items() if k!='conditions'}))


if __name__=='__main__':main()
