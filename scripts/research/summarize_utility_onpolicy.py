#!/usr/bin/env python3
"""Summarize audited fresh one-step oracle utilities with parent-cluster uncertainty."""
import argparse,json,random
from pathlib import Path
from statistics import mean
from scripts.research.audit_masked_angular import sha


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['project','run','audit','out']:p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();root=a.project.resolve()
 pp=root/'research/evidence/utility_onpolicy_protocol_v1.json';protocol=json.loads(pp.read_text())
 parents={};provenance={};total=0;counts={m:dict(attempted=0,scored=0,accepted=0) for m in protocol['methods']}
 for index in protocol['condition_indices']:
  rp=a.run/f'condition_{index:02d}/results.json';ap=a.audit/f'condition_{index:02d}/results.json'
  r=json.loads(rp.read_text());audit=json.loads(ap.read_text())
  assert r['complete'] and audit['complete'] and audit['full_replay']
  assert r['protocol_sha256']==audit['protocol_sha256']==sha(pp)
  assert sha(rp)==audit['source_results_sha256'] and sha(rp.parent/'trace.pt')==audit['trace_sha256']==r['trace_sha256']
  assert r['new_raw_queries']==r['requested_raw_queries']==audit['raw_queries_in_producer']
  total+=r['new_raw_queries'];provenance[str(index)]=dict(results_sha256=sha(rp),audit_sha256=sha(ap))
  for method in protocol['methods']:
   rows=[v for v in r['rows'] if v['method']==method];assert len(rows)==288
   for key in ['attempted','scored','accepted']:
    counts[method][key]+=len(rows) if key=='attempted' else sum(v['valid'] if key=='scored' else v['accepted'] for v in rows)
   for parent in sorted({v['parent'] for v in rows}):
    group=[v for v in rows if v['parent']==parent];assert len(group)==96
    parents[index,parent,method]={k:mean(v[k] for v in group) for k in ['expected_utility_eV','raw_cost','expected_constitutional_flow']}
 assert total<=protocol['maximum_new_raw_queries']
 groups={i:sorted(p for ii,p,m in parents if ii==i and m=='physical') for i in protocol['condition_indices']}
 assert all(len(v)==3 for v in groups.values())
 def metrics(method,selected):
  value={k:mean(mean(parents[i,p,method][k] for p in ids) for i,ids in selected.items())
   for k in ['expected_utility_eV','raw_cost','expected_constitutional_flow']}
  value['utility_eV_per_expected_raw_call']=value['expected_utility_eV']/value['raw_cost'];return value
 result=dict(complete=True,protocol_sha256=sha(pp),provenance=provenance,new_raw_queries=total,
  new_physical_queries_in_summary=0,counts=counts,methods={m:metrics(m,groups) for m in protocol['methods']},
  per_condition={str(i):{m:metrics(m,{i:ids}) for m in protocol['methods']} for i,ids in groups.items()},comparisons={},
  scientific_submission_ready=False,scope='Fresh one-step proposals on72 fixed INTERNAL states from12 validation parents. Actual paired-oracle utility and all failures retained; parent bootstrap keeps source slots, behavioral replicas and proposal trials together. Not full-chain or final-test evidence.')
 base=result['methods']['physical']['utility_eV_per_expected_raw_call']
 for method in protocol['methods'][1:]:
  point=result['methods'][method]['utility_eV_per_expected_raw_call'];rng=random.Random(25671);draws=[]
  for _ in range(5000):
   sample={i:rng.choices(ids,k=len(ids)) for i,ids in groups.items()}
   draws.append(metrics(method,sample)['utility_eV_per_expected_raw_call']-metrics('physical',sample)['utility_eV_per_expected_raw_call'])
  draws.sort();result['comparisons'][method]=dict(difference_eV_per_expected_raw_call=point-base,
   relative_gain=point/base-1 if base>0 else None,fixed_composition_parent_bootstrap95=[draws[125],draws[4875]])
 if a.out.exists():raise FileExistsError(a.out)
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:result[k] for k in ['new_raw_queries','counts','methods','comparisons']},indent=2))


if __name__=='__main__':main()
