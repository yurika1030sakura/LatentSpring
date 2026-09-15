#!/usr/bin/env python3
"""Validate the new Gaussian covariance control against frozen source references."""
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
 for k in ['project','run','out']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();torch.set_num_threads(2)
 if a.out.exists():raise FileExistsError(a.out)
 calibration_path=a.project/'research/evidence/source_sc_covariance_calibration_v1.json';cal=json.loads(calibration_path.read_text());assert cal['complete']
 reports={key:{} for key in ['development','confirmation']};summary={key:{} for key in reports};artifacts=[];costs={};total=0
 for seed in [0,1]:
  path=a.project/f'research/evidence/source_sc_covariance_s{seed}_v1.json';spec=json.loads(path.read_text());ph=sha(path);root=a.run/f's{seed}/study'
  progress=json.loads((root/'progress.json').read_text());assert progress['complete'] and progress['protocol_sha256']==ph and progress['completed']==spec['methods']
  directory=root/'covariance_gaussian';ckpt=directory/'last.ckpt';digest=sha(ckpt);training=json.loads((directory/'training.json').read_text());assert training['complete'] and training['checkpoint_sha256']==digest and training['steps']==3000
  state=torch.load(ckpt,map_location='cpu',weights_only=False);recipe=state['research_protocol'];assert recipe['source_sc_protocol_sha256']==ph and recipe['source_prior_kind']=='covariance_gaussian'
  cfgpath=a.project/spec['config'];assert sha(cfgpath)==spec['config_sha256'];cfg=read_config_file(cfgpath);cfg['mol_fm'].pop('bgfm',None)
  model=model_from_config(cfg);prepare_research_backbone(model,recipe);model.load_state_dict(state['state_dict'],strict=True);assert sum(p.numel() for p in model.parameters())==5904369
  prior=prior_from_checkpoint(state);assert prior.configuration==cal['configuration'];del model,state
  datafile=a.project/spec['data'];assert sha(datafile)==spec['data_sha256'];data=torch.load(datafile,map_location='cpu',weights_only=False)
  ids=[r['condition']['processed_index'] for r in data['training']];metrics=[json.loads(r) for r in (directory/'metrics.jsonl').read_text().splitlines()];assert [r['processed_index'] for r in metrics]==ids
  olddir=a.project/f'runs/source_sc_v1/s{seed}/study/harmonic_tree';oldinit=torch.load(olddir/'feedback_initial.pt',map_location='cpu',weights_only=False);initial=torch.load(directory/'feedback_initial.pt',map_location='cpu',weights_only=False)
  for group in oldinit:assert all(torch.equal(v,initial[group][k]) for k,v in oldinit[group].items())
  oldspec=json.loads((a.project/f'research/evidence/source_sc_s{seed}_v1.json').read_text())
  for k in ['warm_checkpoint_sha256','data_sha256','fm_seed','fm_steps','fm_lr','feedback_lr','edge_log_width','self_conditioning']:assert spec[k]==oldspec[k]
  costs[seed]=dict(training_seconds=training['seconds'],training_forwards=training['primitive_denoiser_training_forwards']);assert costs[seed]['training_forwards']==6000
  for panel in reports:
   e=spec if panel=='development' else dict(spec,**spec['extra_evaluations']['confirmation'])
   destination=root/('evaluation' if panel=='development' else 'confirmation');manifest=a.project/e['condition_manifest'];assert sha(manifest)==e['condition_manifest_sha256'];conditions=json.loads(manifest.read_text())['rows']
   reportfile=destination/'covariance_gaussian_results.json';report=json.loads(reportfile.read_text());assert report['complete'] and report['protocol_sha256']==ph and report['checkpoint_sha256']==digest
   assert [r['condition_index'] for r in report['rows']]==e['conditions']
   for i,row in enumerate(report['rows']):
    sample=destination/f'covariance_gaussian_c{i}.pt';assert sha(sample)==row['sample_sha256'];saved=torch.load(sample,map_location='cpu',weights_only=False);c=saved['condition']
    assert c['composition_hex']==conditions[i]['composition_hex'] and c==row['condition'] and len(saved['positions'])==64
    expected=[e['evaluation_seed']*1000003+i*100003+j for j in range(64)];assert saved['seeds']==expected
    for j,draw_seed in enumerate(expected):
     x,t=sample_source(prior,c['numbers'],c['charge'],c['spin_multiplicity'],draw_seed);torch.testing.assert_close(x,saved['initial_positions'][j],atol=0,rtol=0);assert t==saved['auxiliary_tree_edges'][j];total+=1
    result=assess(saved['positions'],c,list(range(64)));assert all(row[k]==v for k,v in result.items());assert geometry_counts(saved['positions'],c['numbers'])==row['final_geometry']
   reports[panel][seed]={'covariance_gaussian':report['rows']};summary[panel][seed]={}
   parentfile=a.project/('research/evidence/source_sc_audit_v1.json' if panel=='development' else 'research/evidence/source_sc_confirmation_audit_v1.json');parent=json.loads(parentfile.read_text());assert parent['complete'] and sha(parentfile)==spec['parent_'+panel+'_audit_sha256']
   refroot=a.project/(f'runs/source_sc_v1/s{seed}/study/evaluation' if panel=='development' else f'runs/source_sc_confirmation_v1/s{seed}/evaluation')
   for method in ['gaussian','harmonic_tree']:
    file=refroot/f'{method}_results.json';old=json.loads(file.read_text());assert any(r['seed']==seed and r['method']==method and r['report_sha256']==sha(file) for r in parent['artifacts'])
    assert [r['condition']['composition_hex'] for r in old['rows']]==[r['composition_hex'] for r in conditions]
    reports[panel][seed][method]=old['rows']
   for m,rows in reports[panel][seed].items():summary[panel][seed][m]={k:sum(r[k] for r in rows) for k in ['attempted','graph_supported','geometrically_supported','distinct_connectivity','generation_seconds','validator_errors']}
   artifacts.append(dict(seed=seed,panel=panel,checkpoint_sha256=digest,report_sha256=sha(reportfile),data_sha256=spec['data_sha256'],protocol_sha256=ph))
   print(json.dumps(dict(seed=seed,panel=panel,summary=summary[panel][seed])),flush=True)
 comparisons={};rng=np.random.default_rng(36191)
 for panel in reports:
  comparisons[panel]={};rows=reports[panel][0]['gaussian'];strata=[0 if r['condition']['n_atoms']<=16 else 1 if r['condition']['n_atoms']<=28 else 2 for r in rows]
  for left,right in [('harmonic_tree','covariance_gaussian'),('covariance_gaussian','gaussian')]:
   comparisons[panel][left+' minus '+right]={}
   for metric in ['graph_supported','geometrically_supported']:
    d=np.stack([np.stack([flags(x,metric)-flags(y,metric) for x,y in zip(reports[panel][s][left],reports[panel][s][right])]) for s in [0,1]])
    comparisons[panel][left+' minus '+right][metric]=intervals(d,strata,rng)
 point=all(summary[p][s]['harmonic_tree']['graph_supported']>summary[p][s]['covariance_gaussian']['graph_supported'] for p in reports for s in [0,1])
 bounds=all(comparisons[p]['harmonic_tree minus covariance_gaussian']['graph_supported']['paired_draw95'][0]>0 for p in reports)
 write(a.out,dict(complete=True,summary=summary,comparisons=comparisons,costs=costs,artifacts=artifacts,new_outputs_replayed=total,source_calibration_sha256=sha(calibration_path),attribution_gate_passed=point and bounds,new_molecular_oracle_calls=0,scientific_submission_ready=False,scope='New source control on two already examined panels; not a new untouched confirmation. Identical SC architecture, continuation rows and optimization; covariance Gaussian is an explicit128-tree approximation calibrated against8192 trees. No exact-moment or external-generator superiority claim.'))


if __name__=='__main__':main()
