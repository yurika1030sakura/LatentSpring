#!/usr/bin/env python3
"""Summarize qualified training labels without selecting favorable contexts."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


def quantiles(values):
    if not values:return None
    values=sorted(values)
    def at(p):
        x=p*(len(values)-1);i=int(x);f=x-i
        return values[i]*(1-f)+values[min(i+1,len(values)-1)]*f
    return {key:at(p) for key,p in [('min',0),('median',.5),('p90',.9),('p99',.99),('max',1)]}


def summarize(records):
    parents=defaultdict(list)
    for r in records:parents[(r['condition'],r['parent'])].append(r)
    aggregate={}
    for metric in ['force_mse','work_mae_over_kT']:
        aggregate[metric]={}
        for method in ['fitted','site64','site64_confinement']:
            by_condition=defaultdict(list)
            for (index,_),rows in parents.items():
                values=[r[metric][method] for r in rows if r[metric] is not None]
                if values:by_condition[index].append(statistics.mean(values))
            means={str(k):statistics.mean(v) for k,v in by_condition.items()}
            aggregate[metric][method]=dict(mean=statistics.mean(means.values()) if means else None,per_condition=means)
    return dict(contexts=len(records),parents=len(parents),conditions=sorted({r['condition'] for r in records}),
        intrinsic_concentration=quantiles([r['intrinsic_concentration'] for r in records]),
        direction_difference_degrees=quantiles([r['direction_difference_degrees'] for r in records if r['direction_difference_degrees'] is not None]),
        contexts_above_concentration2048=sum(r['intrinsic_concentration']>2048 for r in records),
        negative_center_axial_parameter=sum(r['negative_center_axial_parameter'] for r in records),
        contexts_without_valid_check=sum(r['force_mse'] is None for r in records),errors=aggregate)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();root=args.project
    ap=root/'runs/multicomposition_angular_audit_v1/results.json';audit=json.loads(ap.read_text())
    assert audit['complete'] and audit['independent_numpy_geometry_and_parameter_checks']
    pp=root/'research/evidence/multicomposition_angular_probe_protocol_v1.json';protocol=json.loads(pp.read_text())
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    assert audit['protocol_sha256']==sha(pp)
    groups=defaultdict(list);all_records=[];conditions=[]
    for ar in audit['rows']:
        path=root/f"runs/multicomposition_angular_probe_v1/condition_{ar['index']:02d}/results.json"
        assert sha(path)==ar['results_sha256'];report=json.loads(path.read_text());assert report['complete']
        index=ar['index'];split=protocol['condition_splits'][str(index)]
        conditions.append(dict(index=index,zero_support=report['zero_support'],contexts=report['contexts'],
            supported_exchanges=report.get('supported_exchanges',0),supported_probes=report.get('supported_probes',0),
            rank_failures=report['contexts']-report.get('full_rank_contexts',0),new_raw_queries=report['new_raw_queries']))
        for d in report['diagnostics']:
            if not d['full_rank']:continue
            parent=d['parent_id']
            role=('withheld_composition' if parent in split['withheld_composition_parent_ids'] else
                  'withheld_parent' if parent in split['withheld_parent_ids'] else 'fit')
            assert role!='fit' or parent in split['fit_parent_ids']
            intrinsic=[a-b for a,b in zip(d['fitted_parameter'],d['confinement_parameter'])]
            site=d['physical_site_parameter'];kn=math.sqrt(sum(a*a for a in intrinsic));sn=math.sqrt(sum(a*a for a in site))
            cosine=sum(a*b for a,b in zip(intrinsic,site))/(kn*sn) if min(kn,sn)>1e-12 else None
            angle=math.degrees(math.acos(max(-1,min(1,cosine)))) if cosine is not None else None
            checks=d['checks'];errors={}
            for metric in ['force_mse','work_mae_over_kT']:
                errors[metric]=({method:statistics.mean([c['force_score_mse'][method] if metric=='force_mse' else
                    abs(c['predicted_work_over_kT'][method]-c['actual_work_over_kT']) for c in checks])
                    for method in ['fitted','site64','site64_confinement']} if checks else None)
            r=dict(condition=index,context=d['context'],parent=parent,role=role,
                intrinsic_concentration=kn,direction_difference_degrees=angle,
                negative_center_axial_parameter=d['center_axial_parameter']<0,**errors)
            groups[role].append(r);all_records.append(r)
    result=dict(complete=True,protocol_sha256=sha(pp),audit_sha256=sha(ap),conditions=conditions,
        groups={key:summarize(value) for key,value in groups.items()},all_contexts=summarize(all_records),
        records=all_records,raw_queries_in_probes=audit['raw_queries_in_probes'],new_physical_queries=0,
        scientific_submission_ready=False,
        scope='Training-label diagnostics. Error means average contexts within parent, parents within composition, then compositions. Quantiles are over contexts. No neural fit, equilibrium or competitive sampling claim.')
    if args.out.exists():raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['records','conditions']},indent=2))


if __name__=='__main__':main()
