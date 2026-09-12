#!/usr/bin/env python3
"""Re-evaluate fixed accepted-utility learners, support bounds and trained gradients."""
import argparse,hashlib,json
from pathlib import Path
import torch
from cfm_mol.bounded_arc_guide import BoundedArcGuide
from scripts.research.audit_masked_angular import equal,sha
from scripts.research.train_accepted_utility import evaluate,population_weights,edge_values


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['project','run','out']:p.add_argument('--'+name,type=Path,required=True)
 args=p.parse_args();root=Path(__file__).resolve().parents[2]
 pp=root/'research/evidence/accepted_utility_training_protocol_v1.json';protocol=json.loads(pp.read_text())
 dp=args.project/protocol['data_run'];header=json.loads((dp/'results.json').read_text())
 assert header['complete'] and sha(dp/'results.json')==protocol['data_results_sha256']
 assert sha(dp/'data.pt')==header['data_sha256']==protocol['data_sha256']
 data=torch.load(dp/'data.pt',map_location='cpu',weights_only=False)
 data_pp=root/'research/evidence/accepted_utility_data_protocol_v1.json';data_protocol=json.loads(data_pp.read_text())
 assert header['protocol_sha256']==sha(data_pp)
 source_protocol=json.loads((root/data_protocol['source_protocol']).read_text())
 for index in [1,2,3,5]:
  ids=source_protocol['parent_ids_by_condition'][str(index)]
  ordered=sorted(ids,key=lambda pid:hashlib.sha256(f"{data_protocol['split_seed']}|{index}|{pid}".encode()).hexdigest())
  assert data_protocol['splits'][str(index)]==dict(fit_parent_ids=sorted(ordered[3:]),withheld_parent_ids=sorted(ordered[:3]))
 for row in data:
  expected='fit' if row['parent'] in data_protocol['splits'][str(row['index'])]['fit_parent_ids'] else 'withheld_parent'
  assert row['role']==expected
 assert len(data)==1671 and sum(r['valid'] for r in data)==1597
 fit=[r for r in data if r['role']=='fit'];validation=[r for r in data if r['role']=='withheld_parent']
 weights=population_weights(fit);cost=sum(float(weights[i])*2*r['valid'] for i,r in enumerate(fit))
 rate=sum(float(weights[i])*r.get('baseline_expected_utility_eV',0.) for i,r in enumerate(fit))/cost
 checks=[];summaries=[]
 for replica,seed in enumerate(protocol['seeds']):
  run=args.run/f'replica_{replica}';r=json.loads((run/'results.json').read_text())
  assert r['complete'] and r['protocol_sha256']==sha(pp) and r['steps']==protocol['steps']
  assert r['replica']==replica and r['seed']==seed and r['data_sha256']==protocol['data_sha256']
  assert abs(r['fixed_baseline_utility_per_query_eV']-rate)<1e-12 and abs(r['fixed_baseline_query_cost']-cost)<1e-12
  assert sha(run/'model.pt')==r['model_sha256']
  saved=torch.load(run/'model.pt',map_location='cpu',weights_only=False)
  assert saved['configuration']==protocol['model'] and saved['steps']==protocol['steps'] and saved['protocol_sha256']==sha(pp)
  model=BoundedArcGuide(**saved['configuration']).double();model.load_state_dict(saved['state_dict'])
  for role,records in [('fit',fit),('validation',validation)]:
   equal(evaluate(model,records,protocol['arc_options']),r['final_'+role])
  baseline=BoundedArcGuide(**protocol['model']).double()
  for role,records in [('fit',fit),('validation',validation)]:equal(evaluate(baseline,records,protocol['arc_options']),r['baseline_'+role])
  # Verify the actual cost-adjusted fitting objective at a trained coefficient.
  selected=max((v for v in fit if v['valid']),key=lambda v:abs(v['baseline_expected_utility_eV']))
  def objective():
   u,c,_,_,penalty,_=edge_values(model,selected,protocol['arc_options'])
   return -(u-rate*c)/cost+protocol['ratio_penalty_eV']*penalty
  loss=objective();weight=model.query_head[-1].weight;g=torch.autograd.grad(loss,weight)[0]
  flat=int(g.abs().argmax());i,j=flat//weight.shape[1],flat%weight.shape[1];h=1e-5
  with torch.no_grad():weight[i,j]+=h
  a=float(objective())
  with torch.no_grad():weight[i,j]-=2*h
  b=float(objective())
  with torch.no_grad():weight[i,j]+=h
  error=abs((a-b)/(2*h)-float(g[i,j]));assert error<1e-6+1e-5*abs(float(g[i,j]))
  checks.append(dict(replica=replica,results_sha256=sha(run/'results.json'),model_sha256=sha(run/'model.pt'),
   all_edge_metrics_replayed=True,baseline_replayed=True,trained_cost_adjusted_gradient_fd_error=error))
  summaries.append(dict(replica=replica,baseline_validation=r['baseline_validation']['metrics'],
   final_validation=r['final_validation']['metrics'],per_condition=r['final_validation']['per_condition']))
 result=dict(complete=True,protocol_sha256=sha(pp),data_sha256=sha(dp/'data.pt'),split_rebuilt=True,
  all1671_attempts_retained=True,rows=checks,summary=summaries,new_physical_queries=0,
  optimizer_trajectory_replayed=False,scientific_submission_ready=False,
  scope='Both final models and all empirical metrics re-evaluated; source splits and final objective gradients verified. No full-chain benefit or equilibrium result follows.')
 if args.out.exists():raise FileExistsError(args.out)
 args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(dict(complete=True,models=len(checks),gradient_errors=[r['trained_cost_adjusted_gradient_fd_error'] for r in checks])))


if __name__=='__main__':main()
