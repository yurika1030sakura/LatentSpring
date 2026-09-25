"""Recompute the main comparisons from the saved per-output score arrays."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from cfm_mol.replication_statistics import paired_intervals


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=Path('results'))
    p.add_argument('--out',type=Path,default=Path('reproduced_results.json'))
    a=p.parse_args();r=a.results
    final=dict(np.load(r/'geometry_candidate_five_fit_audit_v1.npz'))
    seed=dict(np.load(r/'seed_replication_audit_v2.npz'))
    metadata=json.loads((r/'seed_replication_audit_v2.json').read_text())
    other=dict(np.load(r/'other_baseline_transfer_audit_v1.npz'))
    additional=dict(np.load(r/'baseline_chemical_geometry_audit_v1.npz'))
    geometry=np.zeros((5,64,16),bool)
    with gzip.open(r/'round2_chemical_geometry_records_v1.jsonl.gz','rt') as f:
        for line in f:
            row=json.loads(line)
            if row['method']=='gaga_parent':geometry[row['fit'],row['condition'],row['sample']]=row['closed_shell_geometry_pass']
    metrics={}
    for name,key in [('Gaussian FM','gaussian_fm'),('EDM','edm')]:
        graph=other[key+'_graph'][:,0];force=other[key+'_force'][:,0];ok=other[key+'_success'][:,0]
        metrics[name]=dict(graph=graph,graph_force=graph&ok&(force<=5),geometry_force=additional[key+'_geometry'][:,0]&ok&(force<=5))
    gi=metadata['methods'].index('gaga_parent');low=seed['success'][:,gi]&(seed['force'][:,gi]<=5)
    metrics['GAGA']=dict(graph=seed['graph'][:,gi],graph_force=seed['graph'][:,gi]&low,geometry_force=geometry&low)
    metrics['LatentSpring']={key:final[key][:,1] for key in ['graph','graph_force','geometry_force']}
    summary={name:dict(attempts=int(row['graph'].size),fits=len(row['graph']),
        **{k:100*float(v.mean()) for k,v in row.items()}) for name,row in metrics.items()}
    expected=json.loads((r/'geometry_candidate_five_fit_audit_v1.json').read_text())
    for key in ['graph','graph_force','geometry_force']:
        np.testing.assert_allclose(summary['LatentSpring'][key]/100,expected['summary']['geometric_candidate'][key]['rate'],rtol=0,atol=1e-15)
    conditions=json.loads((r/'conditions.json').read_text())
    strata=np.array([len(c['numbers'])>28 for c in conditions])
    delta=(final['geometry_force'][2:,1].astype(float)-final['geometry_force'][2:,0]).mean(-1)
    improvement=paired_intervals(delta,strata=strata)
    reference=expected['contrasts']['three_additional']['geometry_force']
    for key in ['mean','by_fit','crossed_fit_composition_ci95','composition_ci95']:
        np.testing.assert_allclose(improvement[key],reference[key],atol=1e-12,rtol=0)
    transfer_arrays=dict(np.load(r/'geometric_correction_transfer_audit_v1.npz'))
    transfer={}
    for family in ['gaussian_fm','harmonic_fm','edm','gaga']:
        transfer[family]={key:(100*transfer_arrays[family+'_'+key].mean((0,2,3))).tolist()
            for key in ['graph','graph_force','geometry_force']}
    transfer_expected=json.loads((r/'geometric_correction_transfer_audit_v1.json').read_text())
    for family,row in transfer.items():
        for metric,values in row.items():
            wanted=[100*transfer_expected['summary'][family][arm][metric]['rate'] for arm in ['parent','geometric_physical']]
            np.testing.assert_allclose(values,wanted,atol=1e-12,rtol=0)
    source_intervals={}
    for metric in ['graph_force','geometry_force']:
        gaussian=transfer_arrays['gaussian_fm_'+metric];harmonic=transfer_arrays['harmonic_fm_'+metric]
        source_intervals[metric]={}
        for arm,label in [(0,'source_only'),(1,'source_with_correction')]:
            change=(harmonic[:,arm].astype(float)-gaussian[:,arm]).mean(-1)
            estimate=paired_intervals(change,strata=strata,seed=78602,repetitions=10000)
            reference=transfer_expected['source_contrasts'][metric][label]
            np.testing.assert_allclose(estimate['crossed_fit_composition_ci95'],reference['crossed_fit_composition_ci95'],atol=1e-12,rtol=0)
            source_intervals[metric][label]=estimate
    continuation_arrays=dict(np.load(r/'geometry_primary_audit_v1.npz'))
    continuation_expected=json.loads((r/'geometry_primary_audit_v1.json').read_text())
    ci=continuation_expected['methods'].index('continued_physical');fi=continuation_expected['methods'].index('full_geometry_physics')
    change=(continuation_arrays['geometry_force'][:,fi].astype(float)-continuation_arrays['geometry_force'][:,ci]).mean(-1)
    rng=np.random.default_rng(73401);indices=rng.integers(change.shape[1],size=(10000,change.shape[1]));fits=rng.integers(len(change),size=(10000,len(change)))
    continuation=dict(mean=float(change.mean()),crossed_fit_composition_ci95=np.quantile(change[fits[:,:,None],indices[:,None,:]].mean((1,2)),[.025,.975]).tolist())
    reference=continuation_expected['contrasts']['full_geometry_physics']['continued_physical']['geometry_force']
    np.testing.assert_allclose(continuation['crossed_fit_composition_ci95'],reference['crossed_fit_composition_ci95'],atol=1e-12,rtol=0)
    organic_arrays=dict(np.load(r/'organic_geometry_audit_v1.npz'))
    organic_expected=json.loads((r/'organic_geometry_audit_v1.json').read_text());organic={}
    for family,row in organic_expected['summary'].items():
        organic[family]={key:100*float(organic_arrays[family+'_'+key].mean()) for key in row}
        for key,value in organic[family].items():np.testing.assert_allclose(value,100*row[key]['rate'],atol=1e-12,rtol=0)
    result=dict(main_comparison=summary,additional_three_fit_improvement=improvement,
        transfer_parent_then_corrected=transfer,source_intervals=source_intervals,
        matched_continuation=continuation,organic_panel=organic,all_checks_passed=True)
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
