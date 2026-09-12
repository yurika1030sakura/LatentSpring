#!/usr/bin/env python3
"""Replay physical FIT trajectories and audit bounded observed proposal utilities."""
import argparse,json,math
from pathlib import Path
import torch

from cfm_mol.accepted_utility import accepted_importance_utility
from cfm_mol.bounded_arc_guide import BoundedArcGuide
from cfm_mol.chemical_sampler import ChemicalTarget
from cfm_mol.joint_arc_geometry import observed_joint_arc_density
from scripts.research.audit_masked_angular import ReplayOracle,equal,sha
from scripts.research.fresh_reuse import run_budget
from scripts.research.joint_arc_budget import inputs,audit_ratios


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['project','out']:p.add_argument('--'+name,type=Path,required=True)
 args=p.parse_args();root=Path(__file__).resolve().parents[2]
 pp=root/'research/evidence/accepted_utility_data_protocol_v1.json';protocol=json.loads(pp.read_text())
 source_pp=root/protocol['source_protocol'];assert sha(source_pp)==protocol['source_protocol_sha256']
 records=[];checks=[];max_error=0.;max_utility_error=0.;source_calls=0
 torch.manual_seed(25613);model=BoundedArcGuide(log_score_bound=.5).double()
 for arm in protocol['source_arms']:
  index,replica=arm['index'],arm['replica'];directory=args.project/arm['directory']
  assert sha(directory/'results.json')==arm['results_sha256'] and sha(directory/arm['trace'])==arm['trace_sha256']
  assert sha(args.project/arm['audit'])==arm['audit_sha256']
  _,old,header,warm,x,ids,physical=inputs(root,args.project,index,source_pp)
  saved=torch.load(directory/arm['trace'],map_location='cpu',weights_only=False)
  oracle=ReplayOracle(saved['query_trace']);oracle.evaluated=saved['raw_query_offset']
  target=ChemicalTarget(oracle,header['condition'],physical['kT_eV'],physical['restraint_eV_A2'])
  actual=dict(raw_query_offset=saved['raw_query_offset'])
  run_budget(target,x,ids,None,'arc_site',replica,index,old,old['shared'],actual)
  actual.update(states=target.states,query_trace=target.query_trace);equal(actual,saved)
  assert oracle.index==len(oracle.queries) and oracle.evaluated-saved['raw_query_offset']==arm['raw_queries']
  independent=audit_ratios(saved,target,'arc_site',old['shared'])
  z=torch.tensor(target.numbers);e=x.new_tensor([header['condition']['charge'],header['condition']['spin_multiplicity'],target.kT])
  n_attempts=n_valid=0;zero_reasons={}
  with torch.no_grad():
   for step,rows in enumerate(saved['transitions']):
    for offset,row in zip(saved['rounds'][step]['active_indices'],rows):
     if row['kind']!='joint_exchange':continue
     parent=ids[offset];state=target.states[row['old_state_id']]
     role='fit' if parent in protocol['splits'][str(index)]['fit_parent_ids'] else 'withheld_parent'
     assert parent in protocol['splits'][str(index)]['fit_parent_ids']+protocol['splits'][str(index)]['withheld_parent_ids']
     record=dict(index=index,replica=replica,parent=parent,step=step,role=role,valid=row['valid'],
      x=state['positions'],bonds=state['graph']['bond_orders'],numbers=z,electronic=e,radii=target.radii,
      action=row.get('action'),source_trace_sha256=arm['trace_sha256'])
     n_attempts+=1
     if row['valid']:
      n_valid+=1;end=target.states[row['new_state_id']]
      qf,_=observed_joint_arc_density(state['positions'],end['positions'],state['graph']['bond_orders'],z,e,target.radii,row['action'],
       kind='arc_bounded',model=model,**old['arc_options'])
      qr,_=observed_joint_arc_density(end['positions'],state['positions'],end['graph']['bond_orders'],z,e,target.radii,row['inverse_action'],
       kind='arc_bounded',model=model,**old['arc_options'])
      err=max(abs(float(qf-row['log_forward_coordinate'])),abs(float(qr-row['log_reverse_coordinate'])));max_error=max(max_error,err);assert err<1e-7
      delta=end['potential_eV']-state['potential_eV']
      target_ratio=-delta/target.kT;action_ratio=math.log(row['forward_count']/row['reverse_count'])
      utility,factor=accepted_importance_utility(qf,qr,row['log_forward_coordinate'],target_ratio,qf.new_tensor(action_ratio),-delta)
      expected=-float(delta)*math.exp(min(0.,row['log_acceptance_ratio']))
      err=abs(float(utility)-expected);max_utility_error=max(max_utility_error,err);assert err<1e-8
      record.update(y=end['positions'],new_bonds=end['graph']['bond_orders'],inverse_action=row['inverse_action'],
       log_behavior_forward=row['log_forward_coordinate'],log_behavior_reverse=row['log_reverse_coordinate'],
       target_log_ratio=target_ratio,action_log_ratio=qf.new_tensor(action_ratio),reward_eV=-delta,
       connectivity_changed=state['graph']['connectivity_smiles']!=end['graph']['connectivity_smiles'],
       behavior_log_acceptance=row['log_acceptance_ratio'],behavior_accepted=row['accepted'],baseline_expected_utility_eV=expected)
     else:
      reason=row['rejection_reason'];zero_reasons[reason]=zero_reasons.get(reason,0)+1
      assert reason in {'Passive context fails the desired contact/overlap graph','No supported angular draw or observed density'}
      record.update(failure_reason=reason,forward_trace=row.get('forward'),zero_utility=True)
     records.append(record)
  source_calls+=arm['raw_queries'];checks.append(dict(index=index,replica=replica,attempts=n_attempts,scored=n_valid,
   zero_reasons=zero_reasons,full_replay=True,independent_joint_MH=independent))
  print(json.dumps(checks[-1]),flush=True)
 assert len(records)==protocol['maximum_source_attempts'] and sum(r['valid'] for r in records)==protocol['expected_scored_attempts']
 assert source_calls==protocol['source_trajectory_raw_queries']
 report=dict(complete=True,protocol_sha256=sha(pp),attempts=len(records),scored=sum(r['valid'] for r in records),rows=checks,
  maximum_initialization_log_density_error=max_error,maximum_initial_utility_error_eV=max_utility_error,
  fitting_parents=len({(r['index'],r['parent']) for r in records if r['role']=='fit'}),
  withheld_parents=len({(r['index'],r['parent']) for r in records if r['role']=='withheld_parent'}),
  source_trajectory_raw_queries=source_calls,source_preparation_raw_queries=protocol['source_preparation_raw_queries'],
  new_physical_queries=0,model_fitted=False,scientific_submission_ready=False,
  scope='All behavior attempts retained; zero failures keep denominators. Bounded model initialization exactly reproduces physical site arcs and actual accepted-work factors. No training or molecular efficiency result follows.')
 if args.out.exists():raise FileExistsError(args.out)
 args.out.mkdir(parents=True);torch.save(records,args.out/'data.pt');report['data_sha256']=sha(args.out/'data.pt')
 (args.out/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'}))


if __name__=='__main__':main()
