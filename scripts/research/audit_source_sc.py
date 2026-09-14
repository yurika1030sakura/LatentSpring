#!/usr/bin/env python3
"""Check matched source controls, reused shell reference and frozen development gate."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from flowmol.model_utils.load import model_from_config,read_config_file
from cfm_mol.radial_reference import prepare_research_backbone
from cfm_mol.source_checkpoint import prior_from_checkpoint
from scripts.research.tree_prior_fm import sample_source,geometry_counts
from scripts.research.audit_generator_output_support import assess
from scripts.research.audit_source_utility import intervals,flags
from scripts.research.train_electronic_fm import sha
from scripts.research.evaluate_chemical_policy import write


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['project','run','out']:p.add_argument('--'+key,type=Path,required=True)
 a=p.parse_args();torch.set_num_threads(2)
 if a.out.exists():raise FileExistsError(a.out)
 data_audit_path=a.project/'research/evidence/geometry_feedback_audit_v1.json';data_audit=json.loads(data_audit_path.read_text());assert data_audit['complete']
 summary={};reports={};costs={};artifacts=[];total=0
 for seed in [0,1]:
  path=a.project/f'research/evidence/source_sc_s{seed}_v1.json';spec=json.loads(path.read_text());ph=sha(path)
  root=a.run/f's{seed}/study';progress=json.loads((root/'progress.json').read_text())
  assert progress['complete'] and progress['protocol_sha256']==ph and progress['completed']==spec['methods']
  file=a.project/spec['data'];assert sha(file)==spec['data_sha256']
  old=next(r for r in data_audit['artifacts'] if r['seed']==seed);assert old['data_sha256']==spec['data_sha256']
  data=torch.load(file,map_location='cpu',weights_only=False);ids=[r['condition']['processed_index'] for r in data['training']]
  cfgpath=a.project/spec['config'];assert sha(cfgpath)==spec['config_sha256'];cfg=read_config_file(cfgpath);cfg['mol_fm'].pop('bgfm',None)
  assert cfg['dataset']['max_atoms']==200 and cfg['mol_fm']['total_loss_weights']['e']==0
  warmfile=a.project/spec['warm_checkpoint'];assert sha(warmfile)==spec['warm_checkpoint_sha256'];warm=torch.load(warmfile,map_location='cpu',weights_only=False)['state_dict']
  panelpath=a.project/spec['condition_manifest'];assert sha(panelpath)==spec['condition_manifest_sha256'];panel=json.loads(panelpath.read_text())
  summary[seed]={};reports[seed]={};costs[seed]={};initials=[]
  for method in spec['methods']:
   directory=root/method;training=json.loads((directory/'training.json').read_text());file=directory/'last.ckpt'
   assert training['complete'] and training['steps']==spec['fm_steps'] and sha(file)==training['checkpoint_sha256']
   if method=='fixed':assert sha(file)==spec['reference']['checkpoint_sha256']
   state=torch.load(file,map_location='cpu',weights_only=False);recipe=state['research_protocol'];feedback='clamped';ref=spec['reference'] if method=='fixed' else None
   expected_ph=ref['protocol_sha256'] if ref else ph
   assert recipe['geometry_feedback_protocol_sha256' if ref else 'source_sc_protocol_sha256']==expected_ph and recipe['position_parameterization']=='displacement'
   for key in ['config_sha256','warm_checkpoint_sha256','data_sha256','fm_seed','fm_steps','fm_lr','feedback_lr','edge_log_width']:assert recipe[key]==spec[key]
   assert recipe['source_prior_kind']==method
   assert not recipe.get('latent_tree_context') and not recipe.get('dynamic_tree_attention')
   if feedback:assert recipe['geometry_self_conditioning']==dict(edge_feedback=feedback,**spec['self_conditioning'])
   else:assert not recipe.get('geometry_self_conditioning')
   multiplier=2 if feedback else 1
   assert training['primitive_denoiser_training_forwards']==spec['fm_steps']*multiplier
   assert spec['midpoint_steps']*2*multiplier==spec['primitive_denoiser_evaluations_per_sample']
   metrics=[json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
   assert [r['processed_index'] for r in metrics]==ids and [r['step'] for r in metrics]==list(range(1,len(ids)+1))
   model=model_from_config(cfg);prepare_research_backbone(model,recipe);model.load_state_dict(state['state_dict'],strict=True);prior=prior_from_checkpoint(state)
   edge_keys=[k for k in warm if k.startswith('vector_field.to_edge_logits.')]
   edge_change=float(torch.stack([(state['state_dict'][k]-warm[k]).square().sum() for k in edge_keys]).sum().sqrt())
   assert edge_change==0.
   if feedback:
    initial=torch.load(directory/'feedback_initial.pt',map_location='cpu',weights_only=False);initials.append(initial)
   parameters=sum(p.numel() for p in model.parameters());del model,state
   reportpath=root/'evaluation'/f'{method}_results.json';report=json.loads(reportpath.read_text());assert report['complete'] and report['protocol_sha256']==expected_ph
   assert report['checkpoint_sha256']==training['checkpoint_sha256'] and len(report['rows'])==12
   if ref:assert sha(reportpath)==ref['report_sha256']
   for index,row in enumerate(report['rows']):
    file=root/'evaluation'/f'{method}_c{index}.pt';assert sha(file)==row['sample_sha256'];saved=torch.load(file,map_location='cpu',weights_only=False);c=saved['condition']
    assert c['composition_hex']==panel['rows'][index]['composition_hex'] and len(saved['positions'])==64
    expected=[spec['evaluation_seed']*1000003+index*100003+j for j in range(64)];assert saved['seeds']==expected
    for j,draw_seed in enumerate(expected):
     x,tree=sample_source(prior,c['numbers'],c['charge'],c['spin_multiplicity'],draw_seed)
     torch.testing.assert_close(x,saved['initial_positions'][j],atol=0,rtol=0);assert tree==saved['auxiliary_tree_edges'][j];total+=1
    result=assess(saved['positions'],c,list(range(64)));assert all(result[k]==row[k] for k in result)
    assert geometry_counts(saved['positions'],c['numbers'])==row['final_geometry']
   rows=report['rows'];reports[seed][method]=rows
   summary[seed][method]=dict(attempts=sum(r['attempted'] for r in rows),graph=sum(r['graph_supported'] for r in rows),geometry=sum(r['geometrically_supported'] for r in rows),
       distinct_connectivity=sum(r['distinct_connectivity'] for r in rows),validator_errors=sum(r['validator_errors'] for r in rows),edge_head_parameter_change=edge_change,parameters=parameters)
   costs[seed][method]=dict(training_seconds=training['seconds'],generation_seconds=sum(r['generation_seconds'] for r in rows),training_primitive_forwards=training['primitive_denoiser_training_forwards'],primitive_evaluations_per_sample=128,reused_training_and_draws=bool(ref))
   artifacts.append(dict(seed=seed,method=method,checkpoint_sha256=training['checkpoint_sha256'],report_sha256=sha(reportpath),metrics_sha256=sha(directory/'metrics.jsonl'),data_sha256=spec['data_sha256']))
   print(json.dumps(dict(seed=seed,method=method,**summary[seed][method])),flush=True)
  for item in initials[1:]:
   for group in ['sc','edge_head']:assert all(torch.equal(v,item[group][k]) for k,v in initials[0][group].items())
 strata=[0 if c['n_atoms']<=16 else 1 if c['n_atoms']<=28 else 2 for c in panel['rows']];rng=np.random.default_rng(34091);comparisons={}
 for left,right in [('fixed','gaussian'),('fixed','harmonic_tree'),('harmonic_tree','gaussian')]:
  key=f'{left} minus {right}';comparisons[key]={}
  for metric in ['graph_supported','geometrically_supported']:
   diff=np.stack([np.stack([flags(x,metric)-flags(y,metric) for x,y in zip(reports[s][left],reports[s][right])]) for s in [0,1]])
   comparisons[key][metric]=intervals(diff,strata,rng)
 point=all(summary[s]['fixed']['graph']>summary[s][m]['graph'] for s in [0,1] for m in ['gaussian','harmonic_tree'])
 bounds=all(comparisons[f'fixed minus {m}']['graph_supported']['paired_draw95'][0]>0 for m in ['gaussian','harmonic_tree'])
 distinct=min(sum(summary[s]['fixed']['distinct_connectivity']-summary[s][m]['distinct_connectivity'] for s in [0,1])/1536 for m in ['gaussian','harmonic_tree'])
 assert len({v['parameters'] for r in summary.values() for v in r.values()})==1
 write(a.out,dict(complete=True,summary=summary,comparisons=comparisons,costs=costs,artifacts=artifacts,
   source_and_structural_outputs_replayed=total,new_generated_outputs=3072,reused_generated_outputs=1536,data_provenance_audit_sha256=sha(data_audit_path),full_checkpoint_restoration=True,identical_feedback_initial_states=True,
   unchanged_edge_head_verified=True,development_gate_passed=point and bounds and distinct>=-.02,new_molecular_oracle_calls=0,scientific_submission_ready=False,
   scope='Reused12-composition development benchmark. All sources use identical geometry-SC structure,3000 two-pass updates and128 primitive inference calls. Same shell references reused without new sampling. Harmonic tree matches shell conditional covariance analytically; no new-composition, original-method or energy-law claim follows automatically.'))


if __name__=='__main__':main()
