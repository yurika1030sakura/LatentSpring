#!/usr/bin/env python3
"""Audit exact parameter arithmetic and all matched physical-quality outputs."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from flowmol.model_utils.load import model_from_config,read_config_file
from cfm_mol.radial_reference import prepare_research_backbone
from cfm_mol.source_checkpoint import prior_from_checkpoint
from scripts.research.tree_prior_fm import sample_source
from scripts.research.audit_generator_output_support import assess
from scripts.research.audit_source_utility import flags,intervals
from scripts.research.audit_source_sc_energy import masked_intervals
from scripts.research.train_electronic_fm import sha
from scripts.research.evaluate_chemical_policy import write


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ['project','run','out']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();torch.set_num_threads(2)
 if a.out.exists():raise FileExistsError(a.out)
 summary={};reports={};energy={};force={};artifacts=[];generated=0;queries=0
 for seed in [0,1]:
  file=a.project/f'research/evidence/thermal_transfer_s{seed}_v1.json';spec=json.loads(file.read_text());ph=sha(file);root=a.run/f's{seed}/study';done=json.loads((root/'complete.json').read_text());assert done['complete'] and done['protocol_sha256']==ph and done['methods']==spec['methods'] and done['raw_queries']==2560
  parentfile=a.project/spec['parent_audit'];assert sha(parentfile)==spec['parent_audit_sha256'];parent=json.loads(parentfile.read_text());assert parent['complete']
  cfgfile=a.project/spec['config'];assert sha(cfgfile)==spec['config_sha256'];cfg=read_config_file(cfgfile);cfg['mol_fm'].pop('bgfm',None)
  states={}
  for name,ref in spec['checkpoints'].items():
   path=a.project/ref['path'];assert sha(path)==ref['sha256'];states[name]=torch.load(path,map_location='cpu',weights_only=False)
  warm=states['frozen'];prior=prior_from_checkpoint(warm);manifest=a.project/spec['condition_manifest'];assert sha(manifest)==spec['condition_manifest_sha256'];panel=json.loads(manifest.read_text())['rows']
  reports[seed]={};summary[seed]={};energy[seed]={};force[seed]={}
  for method in spec['methods']:
   directory=root/method;ckpt=directory/'last.ckpt';state=torch.load(ckpt,map_location='cpu',weights_only=False);transfer=json.loads((directory/'transfer.json').read_text());assert transfer['complete'] and transfer['checkpoint_sha256']==sha(ckpt) and transfer['coefficient']==1. and transfer['new_optimizer_steps']==0
   recipe=state['research_protocol'];assert recipe['paired_update_transfer_protocol_sha256']==ph and recipe['paired_update_transfer_method']==method
   model=model_from_config(cfg);prepare_research_backbone(model,recipe);model.load_state_dict(state['state_dict'],strict=True);names={n for n,_ in model.named_parameters()};assert sum(p.numel() for p in model.parameters())==5904369
   for name,v in warm['state_dict'].items():
    expected=v+(states[spec['updates'][method]]['state_dict'][name]-states['replay']['state_dict'][name]) if name in names else v
    torch.testing.assert_close(state['state_dict'][name],expected,atol=0,rtol=0)
   del model,state
   artifact=root/'evaluation'/f'{method}_results.json';report=json.loads(artifact.read_text());assert report['complete'] and report['protocol_sha256']==ph and report['checkpoint_sha256']==sha(ckpt)
   assert [r['condition_index'] for r in report['rows']]==list(range(10));es=[];fs=[]
   for i,row in enumerate(report['rows']):
    samplefile=root/'evaluation'/f'{method}_c{i}.pt';assert sha(samplefile)==row['sample_sha256'];sample=torch.load(samplefile,map_location='cpu',weights_only=False);c=sample['condition'];assert c==row['condition'] and c['composition_hex']==panel[i]['composition_hex']
    expected=[spec['evaluation_seed']*1000003+i*100003+j for j in range(64)];assert sample['seeds']==expected
    for j,s in enumerate(expected):
     x,t=sample_source(prior,c['numbers'],c['charge'],c['spin_multiplicity'],s);torch.testing.assert_close(x,sample['initial_positions'][j],atol=0,rtol=0);assert t==sample['auxiliary_tree_edges'][j];generated+=1
    assessment=assess(sample['positions'],c,list(range(64)));assert all(row[k]==v for k,v in assessment.items())
    p=root/'physical_eval'/f'{method}_c{i}.pt';d=torch.load(p,map_location='cpu',weights_only=False);assert d['source_sample_sha256']==sha(samplefile);torch.testing.assert_close(d['positions'],sample['positions'],atol=0,rtol=0)
    e=d['raw_energy_eV'];f=d['raw_force_eV_A'];assert e.shape==(128,) and f.shape==(128,c['n_atoms'],3) and torch.isfinite(e).all() and torch.isfinite(f).all()
    ep=(e[:64]+e[64:])/2;fp=(f[:64]-f[64:])/2;torch.testing.assert_close(d['even_energy_eV'],ep,atol=0,rtol=0);torch.testing.assert_close(d['even_force_eV_A'],fp,atol=0,rtol=0);queries+=128
    es.append(ep.numpy()/c['n_atoms']);fs.append(fp.square().sum(-1).mean(-1).sqrt().numpy())
   reports[seed][method]=report['rows'];energy[seed][method]=np.stack(es);force[seed][method]=np.stack(fs);artifacts.append(dict(seed=seed,method=method,checkpoint_sha256=sha(ckpt),report_sha256=sha(artifact),protocol_sha256=ph))
  quality=json.loads((root/'physical_eval/results.json').read_text());assert quality['complete'] and quality['raw_queries']==2560 and len(quality['rows'])==20
  for r in quality['rows']:assert sha(root/'physical_eval'/r['artifact'])==r['artifact_sha256']
  old=a.project/f'runs/thermal_distillation_v1/s{seed}/study'
  for method in ['frozen','escort','work']:
   file=old/'evaluation'/f'{method}_results.json';assert any(r['seed']==seed and r['method']==method and r['report_sha256']==sha(file) for r in parent['artifacts']);rows=json.loads(file.read_text())['rows'];reports[seed][method]=rows
   es=[];fs=[]
   for i,row in enumerate(rows):
    sample=torch.load(old/'physical_eval'/f'{method}_c{i}.pt',map_location='cpu',weights_only=False);es.append(sample['even_energy_eV'].numpy()/row['condition']['n_atoms']);fs.append(sample['even_force_eV_A'].square().sum(-1).mean(-1).sqrt().numpy())
   energy[seed][method]=np.stack(es);force[seed][method]=np.stack(fs)
  for m,rows in reports[seed].items():summary[seed][m]={k:sum(r[k] for r in rows) for k in ['attempted','graph_supported','geometrically_supported','distinct_connectivity','generation_seconds','validator_errors']}
  print(json.dumps(dict(seed=seed,summary=summary[seed])),flush=True)
 assert generated==2560 and queries==5120;comparisons={};rng=np.random.default_rng(37591)
 for left,right in [('escort_delta','frozen'),('work_delta','frozen'),('work_delta','escort_delta'),('work_delta','work')]:
  masks=np.stack([np.stack([flags(x,'graph_supported').astype(bool)&flags(y,'graph_supported').astype(bool) for x,y in zip(reports[s][left],reports[s][right])]) for s in [0,1]]);result={}
  for metric,source in [('energy_per_atom_eV',energy),('force_rms_eV_A',force)]:
   diff=np.stack([source[s][left]-source[s][right] for s in [0,1]]);result[metric]=dict(all_outputs=intervals(diff,[0]*10,rng),common_graph_supported=masked_intervals(diff,masks),common_graph_by_seed=[masked_intervals(diff[s:s+1],masks[s:s+1],seed=37592+s) for s in [0,1]])
  for metric in ['graph_supported','geometrically_supported']:
   diff=np.stack([np.stack([flags(x,metric)-flags(y,metric) for x,y in zip(reports[s][left],reports[s][right])]) for s in [0,1]]);result[metric]=intervals(diff,[0]*10,rng)
  comparisons[left+' minus '+right]=result
 def gate(right):
  p=comparisons['work_delta minus '+right]['energy_per_atom_eV'];point=all(r['mean'] is not None and r['mean']<0 for r in p['common_graph_by_seed']);bound=p['common_graph_supported']['paired_draw95'] is not None and p['common_graph_supported']['paired_draw95'][1]<0
  support=all(sum(summary[s]['work_delta'][metric]-summary[s][right][metric] for s in [0,1])/1280>=-.02 for metric in ['graph_supported','distinct_connectivity'])
  return bool(point and bound and support)
 write(a.out,dict(complete=True,summary=summary,comparisons=comparisons,artifacts=artifacts,new_generation_outputs_replayed=generated,new_raw_energy_force_rows_checked=queries,parameters_reconstructed_exactly=True,improvement_vs_frozen_gate=gate('frozen'),extra_work_vs_escort_gate=gate('escort_delta'),new_optimization_steps=0,scientific_submission_ready=False,
  scope='Single coefficient1 task-arithmetic intervention chosen after direct distillation results; two reused model seeds and the reused10-composition panel. Parameter arithmetic is empirical, not an exact functional or KL cancellation. No fresh-confirmation or global Boltzmann claim.'))


if __name__=='__main__':main()
