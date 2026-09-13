#!/usr/bin/env python3
"""Combine the primary and supplementary chains only after matching contracts."""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.research.audit_masked_angular import sha
from scripts.research.evaluate_chemical_policy import write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['primary','controls','primary-protocol','control-protocol','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--extra-control',type=Path);p.add_argument('--extra-protocol',type=Path)
    a=p.parse_args();primary=json.loads(a.primary.read_text());control=json.loads(a.controls.read_text())
    pp=json.loads(a.primary_protocol.read_text());cp=json.loads(a.control_protocol.read_text())
    assert primary['complete'] and control['complete'] and primary['all_caps_reached'] and control['all_caps_reached']
    assert primary['protocol_sha256']==sha(a.primary_protocol)==cp['paired_primary_protocol_sha256']
    assert control['protocol_sha256']==sha(a.control_protocol)
    fields=['sources','source_kind','physical_protocol','physical_protocol_sha256','condition_indices','replicas','evaluation_seeds',
            'schedule','local_scales','query_cap_per_parent','maximum_microsteps','readouts','policy','source_energy_tolerance_eV']
    for field in fields:assert pp[field]==cp[field],field
    parents={(row['index'],row['parent']):dict(row['data']) for row in primary['parents']}
    for row in control['parents']:
        key=(row['index'],row['parent']);assert key in parents
        for cap,methods in row['data'].items():
            assert not set(methods)&set(parents[key][cap]);parents[key][cap].update(methods)
    controls=[control];control_protocols=[cp]
    if a.extra_control is not None:
        extra=json.loads(a.extra_control.read_text());ep=json.loads(a.extra_protocol.read_text())
        assert extra['complete'] and extra['all_caps_reached'] and extra['protocol_sha256']==sha(a.extra_protocol)
        assert ep['paired_primary_protocol_sha256']==sha(a.primary_protocol)
        for field in fields:assert pp[field]==ep[field],field
        for row in extra['parents']:
            key=(row['index'],row['parent']);assert key in parents
            for cap,methods in row['data'].items():
                assert not set(methods)&set(parents[key][cap]);parents[key][cap].update(methods)
        controls.append(extra);control_protocols.append(ep)
    ordered=sorted(parents);assert len(ordered)==18
    methods=pp['methods']+[m for proto in control_protocols for m in proto['methods']];rng=np.random.default_rng(28911)
    groups=[[j for j,(i,pid) in enumerate(ordered) if i==index] for index in pp['condition_indices']]
    sample=np.concatenate([rng.choice(group,size=(10000,3),replace=True) for group in groups],axis=1)
    readouts={}
    for cap in pp['readouts']:
        means={};comparisons={}
        for metric in ['distinct_connectivities','potential_change_eV','connectivity_returns','max_typed_distance_change_A2','accepted_joint_moves']:
            x=np.array([[parents[key][str(cap)][m] for m in methods] for key in ordered],dtype=object)
            values=np.array([[[row[metric] for row in pair] for pair in arms] for arms in x],dtype=float)
            means[metric]={m:dict(mean=float(values[:,j].mean()),per_replica=values[:,j].mean(0).tolist()) for j,m in enumerate(methods)}
            for method in ['single_linear','panel_radial','panel_blind']:
                for baseline in [m for m in ['root_noise','single_uniform','single_force','single_restraint'] if m in methods]:
                    delta=values[:,methods.index(method)]-values[:,methods.index(baseline)]
                    comparisons.setdefault(method+' minus '+baseline,{})[metric]=dict(mean_difference=float(delta.mean()),per_replica_difference=delta.mean(0).tolist(),
                        descriptive_fixed_composition_parent_bootstrap95=np.quantile(delta[sample].mean((1,2)),[.025,.975]).tolist())
        readouts[str(cap)]=dict(methods=means,comparisons=comparisons)
    if a.out.exists():raise FileExistsError(a.out)
    runtime=dict(primary['runtime'])
    for c in controls:runtime.update(c['runtime'])
    write(a.out,dict(complete=True,primary_sha256=sha(a.primary),controls_sha256=sha(a.controls),extra_control_sha256=sha(a.extra_control) if a.extra_control else None,matched_fields=fields,parents=18,
        trajectories=primary['trajectories']+sum(c['trajectories'] for c in controls),new_raw_queries=primary['new_raw_queries']+sum(c['new_raw_queries'] for c in controls),
        readouts=readouts,runtime=runtime,scientific_submission_ready=False,
        scope='Supplementary physical single-edit controls were added after the primary outcomes. Same sources, target, background, seeds and budgets are verified. All matched controls and both replicas retained. Descriptive development intervals; no cold-start, mixing, final-test or novelty certificate.'))
    print(json.dumps({k:v for k,v in readouts[str(pp['readouts'][-1])]['methods'].items() if k in ['distinct_connectivities','potential_change_eV']}))


if __name__=='__main__':main()
