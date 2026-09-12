#!/usr/bin/env python3
"""Audited screen utility/cost comparisons on the protected internal cohort."""
import argparse
import json
import random
from pathlib import Path
from scripts.research.audit_masked_angular import sha
from scripts.research.summarize_accepted_utility import parent_metrics,aggregate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','run','audit','out']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[2]
    pp=root/'research/evidence/delayed_screen_training_protocol_v1.json';protocol=json.loads(pp.read_text())
    reports={};provenance={};models=[]
    for variant in protocol['variants']:
        for replica in range(2):
            relative=Path(variant)/f'replica_{replica}'/'results.json'
            rp=a.run/relative;ap=a.audit/relative;report=json.loads(rp.read_text());audit=json.loads(ap.read_text())
            assert report['complete'] and audit['complete'] and audit['all_metrics_replayed'] and audit['fixed_controls_replayed']
            assert audit['all1671_attempts_retained'] and audit['reverse_factor_and_balance_verified'] and audit['independent_supported_pairs']==1597
            assert report['protocol_sha256']==audit['protocol_sha256']==sha(pp)
            assert report['data_sha256']==audit['data_sha256']==protocol['data_sha256']
            assert audit['results_sha256']==sha(rp) and audit['model_sha256']==report['model_sha256']==sha(rp.parent/'model.pt')
            assert report['variant']==audit['variant']==variant and report['replica']==audit['replica']==replica
            reports[variant,replica]=report
            provenance[f'{variant}_{replica}']=dict(results_sha256=sha(rp),audit_sha256=sha(ap),model_sha256=report['model_sha256'])
            base=report['zero_validation']['metrics'];final=report['final_validation']['metrics']
            models.append(dict(variant=variant,replica=replica,trainable_parameters=report['trainable_parameters'],
                elapsed_training_and_evaluation_seconds=report['elapsed_seconds'],fit=report['final_fit']['metrics'],
                validation=final,per_condition=report['final_validation']['per_condition'],
                internal_expected_query_cost_fraction=final['expected_raw_cost']/base['expected_raw_cost'],
                internal_signed_utility_fraction=final['utility_eV']/base['utility_eV'],
                internal_expected_acceptance_fraction=final['expected_acceptance']/base['expected_acceptance']))
    comparisons=[]
    for replica in range(2):
        report=reports['linear',replica]
        metrics={v:parent_metrics(reports[v,replica]['final_validation']['edges']) for v in protocol['variants']}
        for name in ['zero','physical']:metrics[name]=parent_metrics(report[f'{name}_validation']['edges'])
        baseline=metrics['zero'];selected={i:sorted(p for ii,p in baseline if ii==i) for i in sorted({ii for ii,p in baseline})}
        assert len(baseline)==12 and all(len(v)==3 for v in selected.values())
        for learned,control in ([('physical','zero')] if replica==0 else [])+[('linear','zero'),('linear','physical'),('neural','zero'),('neural','physical'),('neural','linear')]:
            left,right=metrics[learned],metrics[control];before=aggregate(right,selected);after=aggregate(left,selected)
            rng=random.Random(25831);draws=[]
            for _ in range(5000):
                sample={i:rng.choices(parents,k=len(parents)) for i,parents in selected.items()}
                draws.append(aggregate(left,sample)-aggregate(right,sample))
            draws.sort()
            comparisons.append(dict(replica=replica,learned=learned,control=control,control_rate=before,learned_rate=after,
                difference_eV_per_expected_raw_call=after-before,relative_difference=after/before-1 if before>0 else None,
                fixed_composition_parent_bootstrap95=[draws[125],draws[4875]]))
    result=dict(complete=True,protocol_sha256=sha(pp),provenance=provenance,models=models,comparisons=comparisons,
        zero_validation=reports['linear',0]['zero_validation']['metrics'],physical_validation=reports['linear',0]['physical_validation']['metrics'],
        new_physical_queries=0,actual_queries_saved=0,scientific_submission_ready=False,
        scope='Fixed internal12-parent physical-pair population. Corrected two-stage acceptance and expected cost, not an actual screened chain or achieved saving. Parent intervals retain both trajectories, fix compositions and are not multiplicity-adjusted. Complete-chain, learning and wall-time costs remain untested.')
    if a.out.exists():raise FileExistsError(a.out)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(comparisons,indent=2))


if __name__=='__main__':main()
