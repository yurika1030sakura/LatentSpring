#!/usr/bin/env python3
"""Join complete physical boundary checks to independent replay evidence."""
import argparse,json
from pathlib import Path
from statistics import mean,median
from scripts.research.audit_masked_angular import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run','audit','out']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[2];pp=root/'research/evidence/mobility_boundary_energy_protocol_v1.json';protocol=json.loads(pp.read_text())
    rows=[];calls=states=0;sources={};numeric=[]
    for index in protocol['condition_indices']:
        directory=a.run/f'condition_{index:02d}';rp=directory/'results.json';ap=a.audit/f'condition_{index:02d}/results.json'
        r=json.loads(rp.read_text());audit=json.loads(ap.read_text())
        assert r['complete'] and audit['complete'] and audit['full_replay']
        assert r['protocol_sha256']==audit['protocol_sha256']==sha(pp)
        assert sha(rp)==audit['source_results_sha256'] and sha(directory/'trace.pt')==r['trace_sha256']==audit['trace_sha256']
        assert r['new_raw_queries']==r['requested_raw_queries']==audit['raw_queries_in_producer']
        assert r['new_raw_queries']==r['initial_repeat_raw_queries']+r['candidate_raw_queries']
        sources[str(index)]=dict(results_sha256=sha(rp),trace_sha256=r['trace_sha256'],audit_sha256=sha(ap))
        calls+=r['new_raw_queries'];states+=audit['independently_reconstructed_physical_states'];rows.extend(dict(row,index=index) for row in r['rows']);numeric.append(dict(index=index,**r['numerical_consistency']))
    assert len(rows)==protocol['total_mode_rows']==28 and calls==protocol['maximum_total_new_raw_calls']==94
    tolerance=protocol['resolved_decrease_threshold_eV'];groups={}
    for mode in ['force','distance_projected_force']:
        selected=[v for v in rows if v['mode']==mode];values=[v['actual_decrease_eV'] for v in selected if v['feasible']]
        assert len(selected)==14
        groups[mode]=dict(all_stopped_arms=len(selected),geometry_feasible=len(values),geometry_failed=len(selected)-len(values),
            decrease_above_tolerance=sum(v>tolerance for v in values),increase_above_tolerance=sum(v<-tolerance for v in values),
            within_tolerance=sum(abs(v)<=tolerance for v in values),mean_decrease_among_feasible_eV=mean(values),median_decrease_among_feasible_eV=median(values),
            minimum_decrease_eV=min(values),maximum_decrease_eV=max(values))
    result=dict(complete=True,protocol_sha256=sha(pp),sources=sources,rows=rows,groups=groups,new_raw_queries=calls,independently_checked_physical_states=states,
        resolved_decrease_threshold_eV=tolerance,numerical_consistency=numeric,scientific_submission_ready=False,
        conclusion='Projected distance directions produce resolved finite decreases at12/14 stops, while2 increase energy. Raw force produces no resolved decreases under this frozen geometry-first step rule. This identifies nearby feasible lower-potential points and the need for actual-energy backtracking; it is not convergence, AI novelty or sampler superiority.')
    if a.out.exists():raise FileExistsError(a.out)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(groups=groups,new_raw_queries=calls,independently_checked_physical_states=states),indent=2))


if __name__=='__main__':main()
