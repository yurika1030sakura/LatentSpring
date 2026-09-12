#!/usr/bin/env python3
"""Separate accepted constitutional and same-connectivity moves at matched inference."""
import argparse,json,math
from pathlib import Path
from statistics import mean
import torch
from scripts.research.audit_masked_angular import sha


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['project','out']:p.add_argument('--'+name,type=Path,required=True)
 args=p.parse_args();root=Path(__file__).resolve().parents[2]
 summary_path=args.project/'runs/conditional_arc_chain_summary_v1/results.json'
 summary=json.loads(summary_path.read_text());assert summary['complete'] and summary['all_48_parents_at_all_caps']
 protocol=json.loads((root/'research/evidence/conditional_arc_chain_protocol_v1.json').read_text())
 rows=[]
 for index in protocol['condition_indices']:
  directory=args.project/('runs/conditional_arc_control_repair_v1' if index==5 else f'runs/conditional_arc_chain_v1/condition_{index:02d}')
  report=json.loads((directory/'results.json').read_text());assert sha(directory/'results.json')==summary['evidence'][str(index)]['producer_sha256']
  for arm in report['arms']:
   path=directory/arm['file'];assert sha(path)==arm['trace_sha256'];data=torch.load(path,map_location='cpu',weights_only=False)
   totals={p:dict(joint_attempts=0,scored=0,accepted_same_connectivity=0,accepted_changed_connectivity=0,
    expected_accepted_same_connectivity=0.,expected_accepted_changed_connectivity=0.,
    same_connectivity_accepted_potential_change_eV=0.,changed_connectivity_accepted_potential_change_eV=0.) for p in data['parent_ids']}
   for step,transitions in enumerate(data['transitions']):
    for offset,tr in zip(data['rounds'][step]['active_indices'],transitions):
     if data['query_count_history'][step][offset]>=128 or tr['kind']!='joint_exchange':continue
     item=totals[data['parent_ids'][offset]];item['joint_attempts']+=1
     if not tr['valid']:continue
     item['scored']+=1;old=data['states'][tr['old_state_id']];new=data['states'][tr['new_state_id']]
     same=old['graph']['connectivity_smiles']==new['graph']['connectivity_smiles']
     category='same_connectivity' if same else 'changed_connectivity'
     item['expected_accepted_'+category]+=math.exp(min(0.,tr['log_acceptance_ratio']))
     if tr['accepted']:
      item['accepted_'+category]+=1
      item[category+'_accepted_potential_change_eV']+=float(new['potential_eV']-old['potential_eV'])
   for parent,values in totals.items():rows.append(dict(index=index,parent=parent,method=arm['method'],replica=arm['replica'],**values))
 metrics=[k for k in rows[0] if k not in ['index','parent','method','replica']]
 averages={}
 for method in protocol['methods']:
  group=[r for r in rows if r['method']==method];cases={}
  for index in protocol['condition_indices']:
   selected=[r for r in group if r['index']==index]
   cases[str(index)]={k:mean(r[k] for r in selected) for k in metrics}
  averages[method]=dict(per_condition=cases,**{k:mean(v[k] for v in cases.values()) for k in metrics})
 result=dict(complete=True,source_summary_sha256=sha(summary_path),same_inference_raw_queries_per_parent=128,
  rows=rows,averages=averages,new_physical_queries=0,scientific_submission_ready=False,
  scope='Post-hoc mechanism diagnostic. Canonical connectivity strips stereochemistry. Same-connectivity moves may still change conformation; neither count defines independent molecular samples or mixing. No model or protocol is changed.')
 if args.out.exists():raise FileExistsError(args.out)
 args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({m:{k:v for k,v in a.items() if k!='per_condition'} for m,a in averages.items()},indent=2))


if __name__=='__main__':main()
