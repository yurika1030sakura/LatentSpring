#!/usr/bin/env python3
"""Replay internal-geometry controls and compare them with matching saved baselines."""
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
 previous_path=a.project/'research/evidence/geometry_feedback_audit_v1.json';previous=json.loads(previous_path.read_text());assert previous['complete']
 summary={};reports={};costs={};artifacts=[];total=0
 for seed in [0,1]:
  path=a.project/f'research/evidence/dual_geometry_s{seed}_v1.json';spec=json.loads(path.read_text());ph=sha(path);root=a.run/f's{seed}/study'
  progress=json.loads((root/'progress.json').read_text());assert progress['complete'] and progress['protocol_sha256']==ph and progress['completed']==spec['methods']
  file=a.project/spec['data'];assert sha(file)==spec['data_sha256'];data=torch.load(file,map_location='cpu',weights_only=False)
  ids=[r['condition']['processed_index'] for r in data['training']]
  assert any(r['seed']==seed and r['data_sha256']==spec['data_sha256'] for r in previous['artifacts'])
  cfgpath=a.project/spec['config'];assert sha(cfgpath)==spec['config_sha256'];cfg=read_config_file(cfgpath);cfg['mol_fm'].pop('bgfm',None)
  panelpath=a.project/spec['condition_manifest'];assert sha(panelpath)==spec['condition_manifest_sha256'];panel=json.loads(panelpath.read_text())
  summary[seed]={};reports[seed]={};costs[seed]={};parameter_counts={};state_keys={}
  for method in spec['methods']:
   reference=spec['references'].get(method)
   if reference:
    file=a.project/reference['checkpoint'];assert sha(file)==reference['checkpoint_sha256']
    training=json.loads(file.with_name('training.json').read_text())
   else:
    file=root/method/'last.ckpt';training=json.loads((root/method/'training.json').read_text())
   digest=sha(file);assert training['complete'] and training['checkpoint_sha256']==digest and training['steps']==spec['fm_steps']
   state=torch.load(file,map_location='cpu',weights_only=False);recipe=state['research_protocol']
   for key in ['data_sha256','fm_seed','fm_steps','fm_lr','edge_log_width']:assert recipe[key]==spec[key]
   if not reference:assert recipe['dual_geometry']==dict(mode=spec['geometry_modes'][method]) and recipe['dual_geometry_protocol_sha256']==ph
   else:assert not recipe.get('dual_geometry')
   metrics=[json.loads(line) for line in file.with_name('metrics.jsonl').read_text().splitlines()];assert [r['processed_index'] for r in metrics]==ids
   model=model_from_config(cfg);prepare_research_backbone(model,recipe);model.load_state_dict(state['state_dict'],strict=True)
   parameter_counts[method]=sum(p.numel() for p in model.parameters());state_keys[method]=set(state['state_dict'])
   prior=prior_from_checkpoint(state);del model,state
   reportpath=root/'evaluation'/f'{method}_results.json';report=json.loads(reportpath.read_text());assert report['complete'] and report['protocol_sha256']==ph and report['checkpoint_sha256']==digest
   assert len(report['rows'])==12
   for index,row in enumerate(report['rows']):
    samplefile=root/'evaluation'/f'{method}_c{index}.pt';assert sha(samplefile)==row['sample_sha256'];saved=torch.load(samplefile,map_location='cpu',weights_only=False);c=saved['condition']
    assert c['composition_hex']==panel['rows'][index]['composition_hex'] and len(saved['positions'])==64
    expected=[spec['evaluation_seed']*1000003+index*100003+j for j in range(64)];assert saved['seeds']==expected
    for j,draw_seed in enumerate(expected):
     x,tree=sample_source(prior,c['numbers'],c['charge'],c['spin_multiplicity'],draw_seed)
     torch.testing.assert_close(x,saved['initial_positions'][j],atol=0,rtol=0);assert tree==saved['auxiliary_tree_edges'][j];total+=1
    result=assess(saved['positions'],c,list(range(64)));assert all(result[k]==row[k] for k in result)
    assert geometry_counts(saved['positions'],c['numbers'])==row['final_geometry']
   rows=report['rows'];reports[seed][method]=rows
   summary[seed][method]=dict(attempts=sum(r['attempted'] for r in rows),graph=sum(r['graph_supported'] for r in rows),geometry=sum(r['geometrically_supported'] for r in rows),
      distinct_connectivity=sum(r['distinct_connectivity'] for r in rows),validator_errors=sum(r['validator_errors'] for r in rows),parameters=parameter_counts[method])
   multiplier=2 if method=='geometry' else 1
   assert spec['method_midpoint_steps'][method]*2*multiplier==128
   costs[seed][method]=dict(training_seconds=training['seconds'],generation_seconds=sum(r['generation_seconds'] for r in rows),training_reused=bool(reference),primitive_evaluations_per_sample=128)
   artifacts.append(dict(seed=seed,method=method,checkpoint=str(file),checkpoint_sha256=digest,report_sha256=sha(reportpath),data_sha256=spec['data_sha256']))
   print(json.dumps(dict(seed=seed,method=method,**summary[seed][method])),flush=True)
  assert all(parameter_counts[m]==parameter_counts['plain'] and state_keys[m]==state_keys['plain'] for m in ['fixed_geometry','time_geometry'])
 rng=np.random.default_rng(35091);strata=[0 if c['n_atoms']<=16 else 1 if c['n_atoms']<=28 else 2 for c in panel['rows']];comparisons={}
 for left,right in [('time_geometry','plain'),('time_geometry','fixed_geometry'),('time_geometry','geometry'),('fixed_geometry','plain')]:
  key=f'{left} minus {right}';comparisons[key]={}
  for metric in ['graph_supported','geometrically_supported']:
   diff=np.stack([np.stack([flags(x,metric)-flags(y,metric) for x,y in zip(reports[s][left],reports[s][right])]) for s in [0,1]])
   comparisons[key][metric]=intervals(diff,strata,rng)
 point=all(summary[s]['time_geometry']['graph']>summary[s][m]['graph'] for s in [0,1] for m in ['plain','fixed_geometry'])
 bounds=all(comparisons[f'time_geometry minus {m}']['graph_supported']['paired_draw95'][0]>0 for m in ['plain','fixed_geometry'])
 distinct=sum(summary[s]['time_geometry']['distinct_connectivity']-summary[s]['plain']['distinct_connectivity'] for s in [0,1])/1536
 write(a.out,dict(complete=True,summary=summary,comparisons=comparisons,costs=costs,artifacts=artifacts,source_and_structural_outputs_replayed=total,
  matching_training_data_and_recipes=True,unchanged_parameter_count_and_state_shapes=True,full_checkpoint_restoration=True,
  provenance_audit_sha256=sha(previous_path),development_gate_passed=point and bounds and distinct>=-.02,new_molecular_oracle_calls=0,scientific_submission_ready=False,
  scope='Reused12-composition development benchmark with new source streams. Geometric SC has higher training compute; all inference arms use128 primitive denoiser calls. No final-density, Boltzmann-law or broad originality claim.'))


if __name__=='__main__':main()
