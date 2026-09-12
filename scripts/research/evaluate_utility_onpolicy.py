#!/usr/bin/env python3
"""Fresh one-step proposals on fixed internal states; compare actual oracle utility."""
import argparse,json,math
from pathlib import Path
import torch
from cfm_mol.bounded_arc_guide import BoundedArcGuide
from cfm_mol.chemical_sampler import ChemicalTarget
from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.joint_chemical_geometry import joint_chemical_transition
from scripts.research.audit_masked_angular import ReplayOracle,equal,sha
from scripts.research.audit_joint_arc_support import independent_arc_q
from scripts.research.evaluate_chemical_policy import write


def inputs(root,project,index):
 pp=root/'research/evidence/utility_onpolicy_protocol_v1.json';protocol=json.loads(pp.read_text())
 assert index in protocol['condition_indices']
 ap=project/protocol['model_audit'];audit=json.loads(ap.read_text())
 assert audit['complete'] and sha(ap)==protocol['model_audit_sha256']
 physical_path=root/'research/evidence/parity_training_protocol_v1.json'
 assert sha(physical_path)==protocol['physical_protocol_sha256']
 physical=json.loads(physical_path.read_text());models={'physical':None}
 for row in audit['rows']:
  name=f"learned_s{row['replica']}";info=protocol['models'][name];path=project/info['path']
  assert sha(path)==row['model_sha256']==info['sha256']
  saved=torch.load(path,map_location='cpu',weights_only=False)
  model=BoundedArcGuide(**saved['configuration']).double();model.load_state_dict(saved['state_dict']);model.eval();model.requires_grad_(False)
  models[name]=model
 sources=[];condition=None;cache={}
 for spec in protocol['sources'][str(index)]:
  path=project/spec['trace']
  if path not in cache:
   assert sha(path)==spec['trace_sha256'] and sha(path.parent/'results.json')==spec['results_sha256']
   data=torch.load(path,map_location='cpu',weights_only=False)
   source_condition=json.loads((path.parent/'results.json').read_text())['condition']
   cache[path]=(data,source_condition,spec['trace_sha256'],spec['results_sha256'])
  data,source_condition,trace_hash,result_hash=cache[path]
  assert trace_hash==spec['trace_sha256'] and result_hash==spec['results_sha256']
  if condition is None:condition=source_condition
  assert condition==source_condition
  state=data['states'][spec['state_id']]
  assert data['parent_ids'][spec['parent_offset']]==spec['parent']
  assert data['history_state_ids'][spec['step']][spec['parent_offset']]==spec['state_id']
  sources.append((spec,state))
 assert len(sources)==18
 return pp,protocol,physical,models,condition,sources


def run(target,models,sources,protocol,index):
 initial=[]
 for spec,old in sources:
  state=target.coordinate_state(old['positions'])
  equal(state['graph'],old['graph'])
  for key in ['energy_eV','force_eV_A','potential_eV','score']:state[key]=old[key].clone()
  state.update(state_id=len(target.states),query_batch=None,query_row=None,inherited_source=spec)
  target.states.append(state);initial.append(state)
 attempts=[]
 for method in protocol['methods']:
  for trial in range(protocol['trials_per_source']):
   seeds=[protocol['seed']+100000000*index+100000*spec['parent']+1000*spec['behavior_replica']+100*spec['slot']+trial for spec,old in sources]
   generators=[torch.Generator().manual_seed(seed) for seed in seeds]
   _,rows=joint_chemical_transition(target,initial,kind='arc_site' if method=='physical' else 'arc_bounded',model=models[method],
    generator=None,row_generators=generators,phase=f'{method}_trial_{trial}',site_concentration=64.,arc_options=protocol['arc_options'])
   for source,(old,row,seed) in enumerate(zip(initial,rows,seeds)):
    row.update(method=method,trial=trial,source=source,parent=sources[source][0]['parent'],seed=seed,
     expected_utility_eV=0.,raw_cost=2*int(row['valid']),expected_constitutional_flow=0.)
    if row['valid']:
     new=target.states[row['new_state_id']];alpha=math.exp(min(0.,row['log_acceptance_ratio']))
     row['expected_utility_eV']=-alpha*float(new['potential_eV']-old['potential_eV'])
     row['expected_constitutional_flow']=alpha*int(new['graph']['connectivity_smiles']!=old['graph']['connectivity_smiles'])
    attempts.append(row)
 return dict(attempts=attempts,states=target.states,query_trace=target.query_trace)


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['project','out']:p.add_argument('--'+name,type=Path,required=True)
 for name in ['oracle-python','oracle-checkpoint','run']:p.add_argument('--'+name,type=Path)
 p.add_argument('--index',type=int,required=True);p.add_argument('--phase',choices=['evaluate','audit'],required=True)
 args=p.parse_args();root=Path(__file__).resolve().parents[2]
 pp,protocol,physical,models,condition,sources=inputs(root,args.project,args.index)
 args.out.mkdir(parents=True,exist_ok=True);output=args.out/'results.json'
 if output.exists():raise FileExistsError(output)
 report=dict(complete=False,index=args.index,phase=args.phase,protocol_sha256=sha(pp),condition=condition,
  sources=protocol['sources'][str(args.index)],new_raw_queries=0,scientific_submission_ready=False,
  scope='Fresh one-step proposals on fixed INTERNAL validation sources, with paired oracle and complete MH. No model fitting, long-chain benefit or reserved final evaluation.')
 write(output,report);oracle=target=None
 try:
  if args.phase=='audit':
   producer=json.loads((args.run/'results.json').read_text());assert producer['complete'] and producer['protocol_sha256']==sha(pp)
   assert sha(args.run/'trace.pt')==producer['trace_sha256']
   saved=torch.load(args.run/'trace.pt',map_location='cpu',weights_only=False);oracle=ReplayOracle(saved['query_trace'])
  else:
   assert sha(args.oracle_checkpoint)==physical['raw_oracle_sha256']
   oracle=EnergyOracle(args.oracle_python,root/'scripts/research/oracle_worker.py',args.oracle_checkpoint,
    numbers=condition['numbers'],charge=condition['charge'],spin_multiplicity=condition['spin_multiplicity'],device='cuda',batch_size=32)
   assert oracle.handshake['base_precision_dtype']=='torch.float32' and not oracle.handshake['tf32']
  target=ChemicalTarget(oracle,condition,physical['kT_eV'],physical['restraint_eV_A2'])
  actual=run(target,models,sources,protocol,args.index)
  assert len(actual['attempts'])==18*16*3 and oracle.evaluated==sum(r['raw_cost'] for r in actual['attempts'])
  if args.phase=='audit':
   equal(actual,saved);assert oracle.index==len(oracle.queries) and oracle.evaluated==producer['new_raw_queries']
   checked=0
   for row in saved['attempts']:
    if not row['valid']:continue
    old=saved['states'][row['old_state_id']];new=saved['states'][row['new_state_id']]
    q=saved['query_trace'][new['query_batch']];j=new['query_row']
    energy=(float(q['raw_energy_eV'][j])+float(q['inverted_energy_eV'][j]))/2
    delta=energy+target.restraint/2*float(new['positions'].square().sum())-float(old['potential_eV'])
    ratio=-delta/target.kT+independent_arc_q(row['reverse'])-independent_arc_q(row['forward'])+row['action_log_ratio']
    assert abs(ratio-row['log_acceptance_ratio'])<1e-7
    assert abs(-math.exp(min(0.,ratio))*delta-row['expected_utility_eV'])<1e-8;checked+=1
   report.update(complete=True,source_results_sha256=sha(args.run/'results.json'),trace_sha256=producer['trace_sha256'],
    full_replay=True,independent_MH_checks=checked,raw_queries_in_producer=oracle.evaluated,new_raw_queries=0)
  else:
   assert oracle.evaluated==oracle.requested_evaluations<=protocol['maximum_new_raw_queries']//4
   torch.save(actual,args.out/'trace.pt')
   compact=[{k:r[k] for k in ['method','trial','source','parent','seed','valid','accepted','expected_utility_eV','raw_cost','expected_constitutional_flow']} for r in actual['attempts']]
   report.update(complete=True,rows=compact,trace_sha256=sha(args.out/'trace.pt'),new_raw_queries=oracle.evaluated,
    requested_raw_queries=oracle.requested_evaluations,oracle_runtime=oracle.handshake)
  write(output,report);print(json.dumps({k:v for k,v in report.items() if k not in ['rows','sources','condition','oracle_runtime']}))
 except Exception as exc:
  if target is not None:torch.save(dict(states=target.states,query_trace=target.query_trace),args.out/'failed_trace.pt')
  report.update(failure=f'{type(exc).__name__}: {exc}')
  if oracle is not None and args.phase=='evaluate':report.update(new_raw_queries=oracle.evaluated,requested_raw_queries=oracle.requested_evaluations)
  write(output,report);raise
 finally:
  if oracle is not None and args.phase=='evaluate':oracle.close()


if __name__=='__main__':main()
