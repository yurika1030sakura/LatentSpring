#!/usr/bin/env python3
"""Paired source/destination mobility contrasts with all stops and costs retained."""
import argparse,json,random
from pathlib import Path
from collections import Counter
from statistics import mean
import numpy as np
import torch
from scripts.research.audit_masked_angular import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','run','audit','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--protocol',type=Path,default=Path('research/evidence/mobility_relaxation_protocol_v1.json'))
    a=p.parse_args();root=Path(__file__).resolve().parents[2];pp=root/a.protocol;protocol=json.loads(pp.read_text())
    physical_path=root/protocol['physical_protocol'];assert sha(physical_path)==protocol['physical_protocol_sha256']
    restraint=json.loads(physical_path.read_text())['restraint_eV_A2']
    pairs=[];arms=[];provenance={};calls=initial=checked=new_calls=0
    numerical=[];readouts={str(cap):[] for cap in protocol['options']['readouts']}
    for index in protocol['condition_indices']:
        directory=a.run/f'condition_{index:02d}';rp=directory/'results.json';ap=a.audit/f'condition_{index:02d}/results.json'
        r=json.loads(rp.read_text());audit=json.loads(ap.read_text())
        assert r['complete'] and audit['complete'] and audit['full_replay'] and audit['all_raw_potentials_and_forces_reconstructed']
        assert r['protocol_sha256']==audit['protocol_sha256']==sha(pp) and audit['source_results_sha256']==sha(rp)
        assert sha(directory/'trace.pt')==r['trace_sha256']==audit['trace_sha256']
        assert r['new_raw_queries']==r['requested_raw_queries']
        cumulative=r.get('cumulative_raw_queries',r['new_raw_queries'])
        assert cumulative==r.get('requested_cumulative_raw_queries',r['requested_raw_queries'])==audit['raw_queries_in_producer']
        if 'cumulative_raw_queries' in r:
            assert r['original_prefix_replayed_and_preserved'] and audit['original_prefix_replayed_and_preserved']
            assert cumulative==r['inherited_raw_queries']+r['new_raw_queries']
        saved=torch.load(directory/'trace.pt',map_location='cpu',weights_only=False)
        calls+=cumulative;new_calls+=r['new_raw_queries'];initial+=r['initial_raw_queries'];provenance[str(index)]=dict(results_sha256=sha(rp),audit_sha256=sha(ap),trace_sha256=r['trace_sha256'])
        checked+=audit['queried_trials_checked']
        numerical.append(dict(index=index,repeat_energy_error_eV=r['repeat_energy_error_eV'],repeat_force_error_eV_A=r['repeat_force_error_eV_A'],
            maximum_archived_gap_error_eV=max(v['archived_gap_error_eV'] for v in r['initial_comparisons']),
            maximum_archived_force_error_eV_A=max(v['archived_force_error_eV_A'] for v in r['initial_comparisons'])))
        for arm in saved['arms']:
            best=saved['states'][arm['best_state_id']];force=best['force_eV_A']-restraint*best['positions'];force-=force.mean(0)
            projected=arm['basis']@(arm['basis'].T@force);normal=force-projected
            arms.append(dict(index=index,pair_id=arm['pair_id'],parent=arm['parent'],endpoint=arm['endpoint'],mobility=arm['mobility'],
                status=arm['status'],evaluations=arm['evaluations'],raw_calls=2*arm['evaluations'],geometry_probes=arm['geometry_probes'],
                best_potential_eV=arm['best_potential_eV'],initial_potential_eV=arm['initial_potential_eV'],
                best_projected_force_max_eV_A=arm['best_projected_force_max_eV_A'],best_full_COM_force_max_eV_A=float(force.norm(dim=1).max()),
                best_orthogonal_force_max_eV_A=float(normal.norm(dim=1).max())))
        for pair in saved['pairs']:
            group=[v for v in arms if v['index']==index and v['pair_id']==pair['pair_id']];assert len(group)==4
            get=lambda end,mode:next(v for v in group if v['endpoint']==end and v['mobility']==mode)
            sr,dr,sc,dc=[get(end,mode) for end,mode in [('source','roots'),('destination','roots'),('source','collective'),('destination','collective')]]
            gap_r=dr['best_potential_eV']-sr['best_potential_eV'];gap_c=dc['best_potential_eV']-sc['best_potential_eV']
            pairs.append(dict(index=index,parent=pair['parent'],pair_id=pair['pair_id'],connectivity_changed=pair['connectivity_changed'],
                initial_gap_eV=dr['initial_potential_eV']-sr['initial_potential_eV'],roots_gap_eV=gap_r,collective_gap_eV=gap_c,
                gap_change_collective_minus_roots_eV=gap_c-gap_r,
                source_collective_minus_roots_eV=sc['best_potential_eV']-sr['best_potential_eV'],
                destination_collective_minus_roots_eV=dc['best_potential_eV']-dr['best_potential_eV'],
                all_four_converged=all(v['status'].startswith('converged') for v in group),
                all_four_best_force_qualified=all(v['best_projected_force_max_eV_A']<=protocol['options']['force_tolerance_eV_A'] for v in group)))
            raw_group=[v for v in saved['arms'] if v['pair_id']==pair['pair_id']]
            for cap in protocol['options']['readouts']:
                values={}
                for arm in raw_group:
                    eligible=[arm['initial_state_id']]+[e['state_id'] for e in arm['events'] if e['queried'] and e['evaluation']<=cap]
                    values[(arm['endpoint'],arm['mobility'])]=min(float(saved['states'][sid]['potential_eV']) for sid in eligible)
                root_gap=values['destination','roots']-values['source','roots']
                collective_gap=values['destination','collective']-values['source','collective']
                readouts[str(cap)].append(dict(index=index,pair_id=pair['pair_id'],roots_gap_eV=root_gap,collective_gap_eV=collective_gap,
                    gap_change_collective_minus_roots_eV=collective_gap-root_gap,
                    optimization_raw_queries=2*sum(min(cap,v['evaluations']) for v in raw_group),
                    early_stopped_arms=sum(v['evaluations']<cap for v in raw_group)))
    maximum=protocol.get('maximum_total_cumulative_raw_calls',protocol['maximum_total_new_raw_calls'])
    assert len(pairs)==32 and len(arms)==128 and calls<=maximum and new_calls<=protocol['maximum_total_new_raw_calls']
    assert checked*2==calls-initial==sum(v['raw_calls'] for v in arms)
    keys=['initial_gap_eV','roots_gap_eV','collective_gap_eV','gap_change_collective_minus_roots_eV','source_collective_minus_roots_eV','destination_collective_minus_roots_eV']
    summaries={k:mean(p[k] for p in pairs) for k in keys};rng=random.Random(26111)
    groups=[[j for j,p in enumerate(pairs) if p['index']==i] for i in protocol['condition_indices']];assert all(len(g)==8 for g in groups)
    samples=np.array([[j for group in groups for j in rng.choices(group,k=8)] for _ in range(5000)])
    intervals={}
    for key in keys:
        values=np.array([p[key] for p in pairs]);b=np.sort(values[samples].mean(1));intervals[key]=[float(b[125]),float(b[4875])]
    result=dict(complete=True,protocol_sha256=sha(pp),provenance=provenance,new_raw_queries=new_calls,cumulative_raw_queries=calls,
        inherited_raw_queries=calls-new_calls,initial_repeat_raw_queries=initial,
        optimization_raw_queries=calls-initial,maximum_raw_queries=maximum,maximum_new_raw_queries=protocol['maximum_total_new_raw_calls'],pairs=pairs,arms=arms,
        independently_checked_optimization_queries=checked,numerical_consistency=numerical,
        cap_readouts={cap:dict(pairs=values,mean={key:mean(v[key] for v in values) for key in ['roots_gap_eV','collective_gap_eV','gap_change_collective_minus_roots_eV']},
            optimization_raw_queries=sum(v['optimization_raw_queries'] for v in values),early_stopped_arms=sum(v['early_stopped_arms'] for v in values)) for cap,values in readouts.items()},
        optimization_calls_by_mobility={mode:sum(v['raw_calls'] for v in arms if v['mobility']==mode) for mode in ['roots','collective']},
        mean=summaries,descriptive_fixed_composition_parent_bootstrap95=intervals,
        statuses={mode:dict(Counter(v['status'] for v in arms if v['mobility']==mode)) for mode in ['roots','collective']},
        all_four_converged_pairs=sum(p['all_four_converged'] for p in pairs),all_four_best_force_qualified_pairs=sum(p['all_four_best_force_qualified'] for p in pairs),
        per_condition={str(i):{k:mean(p[k] for p in pairs if p['index']==i) for k in keys} for i in protocol['condition_indices']},
        scientific_submission_ready=False,scope='All32 FIT pairs and128 arms retained. Differences compare best feasible observed potentials under equal candidate-query caps, not necessarily equal actual counts or converged/global minima. Source controls remove generic relaxation as a sole explanation. This is teacher feasibility, not sampling, dynamics, a new MH proposal or AI advantage.')
    if a.out.exists():raise FileExistsError(a.out)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['new_raw_queries','mean','descriptive_fixed_composition_parent_bootstrap95','statuses','all_four_converged_pairs']},indent=2))


if __name__=='__main__':main()
