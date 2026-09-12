#!/usr/bin/env python3
"""FIT-only accounting of scored joint proposals and rejected oracle work.

Exact energies appear only in an explicitly inadmissible information diagnostic.
No fitted screen, actual saved queries, or chain speedup is measured here.
"""
import argparse
import json
import math
from pathlib import Path
import torch
from scripts.research.audit_masked_angular import sha
from scripts.research.train_accepted_utility import population_weights


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[2]
    pp=root/'research/evidence/accepted_utility_data_protocol_v1.json';protocol=json.loads(pp.read_text())
    dp=a.project/'runs/accepted_utility_data_v1/dataset';header=json.loads((dp/'results.json').read_text())
    assert header['complete'] and header['protocol_sha256']==sha(pp) and header['data_sha256']==sha(dp/'data.pt')
    data=torch.load(dp/'data.pt',map_location='cpu',weights_only=False)
    fit=[r for r in data if r['role']=='fit'];assert len({(r['index'],r['parent']) for r in fit})==36
    weights=population_weights(fit);expected_rejected_calls=0.;rates=[]
    for index in sorted({r['index'] for r in fit}):
        rows=[r for r in fit if r['index']==index];scored=[r for r in rows if r['valid']]
        alpha=[math.exp(min(0.,float(r['behavior_log_acceptance']))) for r in scored]
        possible=2*sum(1-v for v in alpha);expected_rejected_calls+=possible
        rates.append(dict(index=index,attempts=len(rows),scored=len(scored),joint_raw_queries=2*len(scored),
            expected_acceptances=sum(alpha),expected_rejected_scored_calls=possible))
    weight_scored=sum(float(w)*int(r['valid']) for w,r in zip(weights,fit))
    weight_accept=sum(float(w)*math.exp(min(0.,float(r['behavior_log_acceptance']))) for w,r in zip(weights,fit) if r['valid'])
    utility=sum(float(w)*r.get('baseline_expected_utility_eV',0.) for w,r in zip(weights,fit))
    # An oracle-informed bounded factor would preserve MH acceptance but cannot
    # be used before querying the energy. It is NOT a proposed cheap algorithm.
    bound=math.log(16)
    ideal_cost=sum(float(w)*2*math.exp(min(0.,max(-bound,float(r['behavior_log_acceptance']))))
                   for w,r in zip(weights,fit) if r['valid'])
    trajectory_calls=0;provenance=[]
    for source in protocol['source_arms']:
        rp=a.project/source['directory']/'results.json';assert sha(rp)==source['results_sha256']
        report=json.loads(rp.read_text());assert report['complete']
        arm=next(v for v in report['arms'] if v['method']=='arc_site' and v['replica']==source['replica'])
        assert arm['trace_sha256']==source['trace_sha256'] and arm['raw_queries']==source['raw_queries']
        fit_ids=set(protocol['splits'][str(source['index'])]['fit_parent_ids'])
        calls=sum(c for pid,c in zip(report['parent_ids'],arm['raw_queries_per_parent']) if pid in fit_ids)
        trajectory_calls+=calls;provenance.append(dict(index=source['index'],replica=source['replica'],fit_trajectory_raw_calls=calls,results_sha256=sha(rp)))
    result=dict(complete=True,fit_parents=36,attempts=len(fit),scored=sum(r['valid'] for r in fit),
        data_sha256=sha(dp/'data.pt'),data_protocol_sha256=sha(pp),per_condition=rates,provenance=provenance,
        fit_trajectory_raw_calls=trajectory_calls,fit_joint_raw_calls=2*sum(r['valid'] for r in fit),
        expected_rejected_joint_raw_calls=expected_rejected_calls,
        rejected_joint_fraction_of_recorded_trajectory_cost=expected_rejected_calls/trajectory_calls,
        balanced_scored_probability=weight_scored,balanced_expected_acceptance_per_attempt=weight_accept,
        balanced_expected_acceptance_given_scored=weight_accept/weight_scored,
        physical_utility_eV_per_expected_raw_call=utility/(2*weight_scored),
        inadmissible_oracle_informed_bounded_screen=dict(log_factor_bound=bound,expected_cost_per_attempt=ideal_cost,
            preserved_utility_eV_per_attempt=utility,utility_eV_per_expected_raw_call=utility/ideal_cost,
            deployable=False,uses_unavailable_candidate_oracle_energy=True),
        new_physical_queries=0,model_fitted=False,actual_queries_saved=0,
        separate_oracle_and_geometry_timing_available=False,scientific_submission_ready=False,
        scope='Only protected FIT behavior pairs. Rejected cost describes the recorded joint proposals, not all move families. Excludes preparation and learning overhead; a changed chain need not revisit this source population. No achievable speedup or learned contribution follows.')
    if a.out.exists():raise FileExistsError(a.out)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['provenance','scope','per_condition']},indent=2))


if __name__=='__main__':main()
