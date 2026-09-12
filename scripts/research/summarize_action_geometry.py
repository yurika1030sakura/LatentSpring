#!/usr/bin/env python3
"""Audited three-way action/geometry comparisons with paired parent uncertainty."""
import argparse
import json
import random
from pathlib import Path

from scripts.research.audit_masked_angular import sha
from scripts.research.summarize_accepted_utility import parent_metrics, aggregate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','run','audit','out']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[2]
    pp=root/'research/evidence/action_geometry_training_protocol_v1.json';protocol=json.loads(pp.read_text())
    reports={};provenance={};results=[]
    for variant in protocol['variants']:
        for replica in range(2):
            relative=Path(variant)/f'replica_{replica}'/'results.json'
            rp=a.run/relative;ap=a.audit/relative
            report=json.loads(rp.read_text());audit=json.loads(ap.read_text())
            assert report['complete'] and audit['complete'] and audit['split_rebuilt'] and audit['all_edge_metrics_replayed']
            assert audit['all1671_attempts_retained'] and audit['baseline_replayed'] and audit['independent_full_densities']==96
            assert report['protocol_sha256']==audit['protocol_sha256']==sha(pp)
            assert audit['results_sha256']==sha(rp) and audit['model_sha256']==report['model_sha256']==sha(rp.parent/'model.pt')
            assert report['variant']==audit['variant']==variant and report['replica']==audit['replica']==replica
            reports[variant,replica]=report
            provenance[f'{variant}_{replica}']=dict(results_sha256=sha(rp),audit_sha256=sha(ap),model_sha256=report['model_sha256'])
            results.append(dict(variant=variant,replica=replica,trainable_parameters=report['trainable_parameters'],
                elapsed_seconds=report['elapsed_seconds'],fit=report['final_fit']['metrics'],
                validation=report['final_validation']['metrics'],per_condition=report['final_validation']['per_condition']))
    comparisons=[]
    for replica in range(2):
        metrics={variant:parent_metrics(reports[variant,replica]['final_validation']['edges']) for variant in protocol['variants']}
        baseline=parent_metrics(reports['geometry',replica]['baseline_validation']['edges']);metrics['physical']=baseline
        selected={index:sorted(parent for i,parent in baseline if i==index) for index in sorted({i for i,p in baseline})}
        assert len(baseline)==12 and all(len(parents)==3 for parents in selected.values())
        for variant in protocol['variants']:
            other=parent_metrics(reports[variant,replica]['baseline_validation']['edges'])
            assert set(other)==set(baseline)
            for key in other:
                assert all(abs(other[key][m]-baseline[key][m])<1e-9 for m in other[key])
        for learned,control in [('action','physical'),('geometry','physical'),('joint','physical'),('joint','geometry'),('joint','action')]:
            left,right=metrics[learned],metrics[control]
            before=aggregate(right,selected);after=aggregate(left,selected)
            rng=random.Random(25731);draws=[]
            for _ in range(5000):
                sample={i:rng.choices(parents,k=len(parents)) for i,parents in selected.items()}
                draws.append(aggregate(left,sample)-aggregate(right,sample))
            draws.sort()
            comparisons.append(dict(replica=replica,learned=learned,control=control,
                control_eV_per_expected_raw_call=before,learned_eV_per_expected_raw_call=after,
                difference_eV_per_expected_raw_call=after-before,relative_difference=after/before-1 if before>0 else None,
                fixed_composition_parent_bootstrap95=[draws[125],draws[4875]]))
    result=dict(complete=True,protocol_sha256=sha(pp),provenance=provenance,models=results,comparisons=comparisons,
        new_physical_queries=0,scientific_submission_ready=False,
        scope='Frozen 12-parent internal selection cohort and four fixed compositions. Paired resampling retains both behavior trajectories and all failed attempts. Intervals are descriptive, not multiplicity-adjusted. No actual fresh-proposal or full-chain advantage follows from these empirical estimates.')
    if a.out.exists():raise FileExistsError(a.out)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(comparisons,indent=2))


if __name__=='__main__':main()
