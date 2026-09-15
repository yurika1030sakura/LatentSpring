#!/usr/bin/env python3
"""Audit local escort work, FIT isolation and all learned-generator outcomes."""
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
from flowmol.model_utils.load import read_config_file,model_from_config
from cfm_mol.radial_reference import prepare_research_backbone
from cfm_mol.source_checkpoint import prior_from_checkpoint
from cfm_mol.chemical_moves import infer_chemical_graph
from scripts.research.tree_prior_fm import sample_source,geometry_counts
from scripts.research.audit_generator_output_support import assess
from scripts.research.audit_source_utility import flags,intervals
from scripts.research.audit_source_sc_energy import masked_intervals
from scripts.research.train_electronic_fm import sha
from scripts.research.evaluate_chemical_policy import write


def key(x,c):
 g=infer_chemical_graph(x,c['numbers'],c['charge'])
 return (tuple(g['bond_orders'].flatten().tolist()),tuple(g['formal_charges']),tuple(g['radical_electrons']))


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ['project','run','out']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();torch.set_num_threads(2)
 if a.out.exists():raise FileExistsError(a.out)
 summary={};reports={};energy={};force={};teacher=[];artifacts=[];costs={};total_fit=0;total_eval=0;raw_count=0
 for seed in [0,1]:
  path=a.project/f'research/evidence/thermal_distillation_s{seed}_v1.json';spec=json.loads(path.read_text());ph=sha(path);root=a.run/f's{seed}/study'
  done=json.loads((root/'complete.json').read_text());assert done['complete'] and done['protocol_sha256']==ph and done['methods']==spec['methods']
  datafile=a.project/spec['data'];assert sha(datafile)==spec['data_sha256'];data=torch.load(datafile,map_location='cpu',weights_only=False)
  warmfile=a.project/spec['warm_checkpoint'];assert sha(warmfile)==spec['warm_checkpoint_sha256'];warm=torch.load(warmfile,map_location='cpu',weights_only=False);prior=prior_from_checkpoint(warm)
  manifest=a.project/spec['condition_manifest'];assert sha(manifest)==spec['condition_manifest_sha256'];panel=json.loads(manifest.read_text())['rows'];excluded={r['composition_hex'] for r in panel}
  cfgpath=a.project/spec['config'];assert sha(cfgpath)==spec['config_sha256'];cfg=read_config_file(cfgpath);cfg['mol_fm'].pop('bgfm',None)
  pools=[];teacher_calls=0;teacher_stats=[]
  for i,row_index in enumerate(spec['teacher_training_rows']):
   c=data['training'][row_index]['condition'];assert c['composition_hex'] not in excluded and c['source_split']=='train'
   rawfile=root/'teacher'/f'raw_c{i}.pt';raw=torch.load(rawfile,map_location='cpu',weights_only=False);assert raw['role']=='FIT' and raw['condition']==c and raw['training_processed_index']==c['processed_index']
   assert raw['raw_positions'].shape==(32,c['n_atoms'],3)
   for j in range(32):
    x,_=sample_source(prior,c['numbers'],c['charge'],c['spin_multiplicity'],spec['teacher_seed']*1000003+i*100003+j);torch.testing.assert_close(x,raw['source_positions'][j],atol=0,rtol=0);total_fit+=1
   assessment=assess(raw['raw_positions'],c,list(range(32)));assert assessment==raw['assessment'];keep=torch.tensor([r['graph_supported'] for r in assessment['records']],dtype=torch.bool);assert torch.equal(keep,raw['graph_supported'])
   if not keep.any():continue
   file=root/'teacher'/f'refined_c{i}.pt';saved=torch.load(file,map_location='cpu',weights_only=False);assert saved['role']=='FIT' and saved['condition']==c and saved['raw_source_sha256']==sha(rawfile)
   r=saved['record'];anchor=raw['raw_positions'][keep];torch.testing.assert_close(r['anchor'],anchor,atol=0,rtol=0);n=len(anchor);k=spec['thermal_teacher']['particles'];kt=spec['thermal_teacher']['kT']
   ea=r['anchor_raw_energy_eV'];fa=r['anchor_raw_force_eV_A'];assert ea.shape==(2*n,) and fa.shape==(2*n,c['n_atoms'],3)
   e=(ea[:n]+ea[n:])/2;f=(fa[:n]-fa[n:])/2;torch.testing.assert_close(r['anchor_energy_eV'],e,atol=0,rtol=0);torch.testing.assert_close(r['anchor_force_eV_A'],f,atol=0,rtol=0)
   centered=f-f.mean(1,keepdim=True);norm=centered.square().sum((1,2)).sqrt();sigma=torch.minimum(torch.full_like(norm,spec['thermal_teacher']['max_sigma']),torch.sqrt(spec['thermal_teacher']['max_shift']*kt/norm.clamp_min(1e-12)));delta=sigma[:,None,None].square()*centered/kt
   noise=torch.randn((n,k,c['n_atoms'],3),dtype=torch.float64,generator=torch.Generator().manual_seed(spec['teacher_seed']*100003+i));noise-=noise.mean(-2,keepdim=True)
   x=anchor[:,None]+sigma[:,None,None,None]*noise;y=x+delta[:,None]
   for label,val in [('sigma',sigma),('shift',delta),('source',x),('proposal',y)]:torch.testing.assert_close(r[label],val,atol=1e-12,rtol=1e-12)
   graph_keys=[key(xx,c) for xx in anchor];assert graph_keys==r['graph_keys'];valid=torch.zeros(n,k,dtype=torch.bool)
   for b in range(n):
    for j in range(k):
     try:valid[b,j]=key(y[b,j],c)==graph_keys[b]
     except (ValueError,RuntimeError):pass
   assert torch.equal(valid,r['valid']);count=int(valid.sum());ey=r['proposal_raw_energy_eV'];fy=r['proposal_raw_force_eV_A'];assert ey.shape==(2*count,) and fy.shape==(2*count,c['n_atoms'],3)
   ep=torch.zeros(n,k,dtype=torch.float64);ep[valid]=(ey[:count]+ey[count:])/2;torch.testing.assert_close(ep,r['proposal_energy_eV'],atol=0,rtol=0)
   # Independently reconstruct both fully normalized intrinsic Gaussian logs.
   dimension=3*(c['n_atoms']-1);normalizer=-dimension/2*torch.log(2*math.pi*sigma.square())[:,None]
   logqx=normalizer-(x-anchor[:,None]).square().sum((-1,-2))/(2*sigma[:,None].square())
   logqy=normalizer-(y-anchor[:,None]).square().sum((-1,-2))/(2*sigma[:,None].square())
   logw=-(ep-e[:,None])/kt+logqy-logqx;logw[~valid]=-torch.inf;torch.testing.assert_close(logw,r['log_work_weights'],atol=1e-8,rtol=1e-10);torch.testing.assert_close(-kt*logw,r['work_eV'],atol=1e-8,rtol=1e-10)
   eligible=valid.any(1);weights=torch.zeros_like(logw);weights[eligible]=logw[eligible].softmax(-1);uniform=valid.double()/valid.sum(1).clamp_min(1)[:,None]
   torch.testing.assert_close(weights,r['weights'],atol=1e-9,rtol=1e-9);torch.testing.assert_close(uniform,r['uniform_weights'],atol=0,rtol=0);assert torch.equal(eligible,r['eligible'])
   ess=torch.zeros(n,dtype=torch.float64);ess[eligible]=1/weights[eligible].square().sum(1);torch.testing.assert_close(ess,r['local_particle_ess'],atol=1e-8,rtol=1e-9)
   for label,val in [('raw_positions',anchor[eligible]),('proposals',y[eligible]),('weights',weights[eligible]),('uniform_weights',uniform[eligible])]:torch.testing.assert_close(saved[label],val,atol=1e-8,rtol=1e-9)
   calls=2*n+2*count;assert saved['oracle_queries']==calls;teacher_calls+=calls
   if eligible.any():pools.append(saved)
   conditional=(ep-e[:,None]);teacher_stats.append(dict(condition=i,anchors=n,retained=int(eligible.sum()),mean_local_ess=float(ess.mean()),mean_energy_change_uniform_eV=float((uniform[eligible]*conditional[eligible]).sum(1).mean()) if eligible.any() else None,mean_energy_change_work_eV=float((weights[eligible]*conditional[eligible]).sum(1).mean()) if eligible.any() else None,artifact_sha256=sha(file)))
  teacher_progress=json.loads((root/'teacher_progress.json').read_text());assert teacher_progress['complete'] and teacher_progress['queries']==teacher_progress['requested']==teacher_calls==done['teacher_raw_queries']
  assert len(pools)>=spec['minimum_teacher_compositions'];teacher.append(dict(seed=seed,rows=teacher_stats,raw_queries=teacher_calls))
  reports[seed]={};summary[seed]={};energy[seed]={};force[seed]={};costs[seed]={};common_labels=None
  for method in spec['methods']:
   if method=='frozen':ckpt=warmfile
   else:
    directory=root/method;ckpt=directory/'last.ckpt';training=json.loads((directory/'training.json').read_text());assert training['complete'] and training['checkpoint_sha256']==sha(ckpt) and training['steps']==1000
    metrics=[json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()];assert len(metrics)==1000
    labels=[];rng=torch.Generator().manual_seed(spec['training_seed']+17)
    for step,metric in enumerate(metrics,1):
     if step%2:
      idx=int(torch.randint(len(data['training']),(1,),generator=rng));expected=dict(kind='reference',row=idx);c=data['training'][idx]['condition']
     else:
      idx=int(torch.randint(len(pools),(1,),generator=rng));entry=pools[idx];j=int(torch.randint(len(entry['raw_positions']),(1,),generator=rng));u=float(torch.rand((),generator=rng));c=entry['condition']
      particle=-1 if method=='replay' else int(torch.searchsorted(entry['weights' if method=='work' else 'uniform_weights'][j].cumsum(0),u,right=True).clamp_max(entry['weights'].shape[1]-1))
      expected=dict(kind='generated_fit',pool=idx,row=j,particle=particle,uniform_draw=u)
     assert metric['step']==step and metric['selection']==expected and metric['composition']==c['composition_hex'] and math.isfinite(metric['loss']) and math.isfinite(metric['gradient_norm'])
     labels.append({k:v for k,v in expected.items() if k!='particle'})
    if common_labels is None:common_labels=labels
    else:assert labels==common_labels
    state=torch.load(ckpt,map_location='cpu',weights_only=False);assert state['research_protocol']['thermal_distillation_protocol_sha256']==ph and state['research_protocol']['thermal_distillation_method']==method
    model=model_from_config(cfg);prepare_research_backbone(model,state['research_protocol']);model.load_state_dict(state['state_dict'],strict=True);assert sum(p.numel() for p in model.parameters())==5904369;del model,state
    costs[seed][method]=dict(training_seconds=training['seconds'],training_steps=1000)
   file=root/'evaluation'/f'{method}_results.json';report=json.loads(file.read_text());assert report['complete'] and report['protocol_sha256']==ph and report['checkpoint_sha256']==sha(ckpt) and len(report['rows'])==10
   e_rows=[];f_rows=[]
   for i,row in enumerate(report['rows']):
    samplefile=root/'evaluation'/f'{method}_c{i}.pt';assert sha(samplefile)==row['sample_sha256'];sample=torch.load(samplefile,map_location='cpu',weights_only=False);c=sample['condition'];assert c['composition_hex']==panel[i]['composition_hex'] and c==row['condition']
    expected=[spec['evaluation_seed']*1000003+i*100003+j for j in range(64)];assert sample['seeds']==expected
    for j,draw_seed in enumerate(expected):
     x,t=sample_source(prior,c['numbers'],c['charge'],c['spin_multiplicity'],draw_seed);torch.testing.assert_close(x,sample['initial_positions'][j],atol=0,rtol=0);assert t==sample['auxiliary_tree_edges'][j];total_eval+=1
    assessment=assess(sample['positions'],c,list(range(64)));assert all(row[k]==v for k,v in assessment.items())
    physical=root/'physical_eval'/f'{method}_c{i}.pt';d=torch.load(physical,map_location='cpu',weights_only=False);assert d['source_sample_sha256']==sha(samplefile);torch.testing.assert_close(d['positions'],sample['positions'],atol=0,rtol=0)
    raw_e=d['raw_energy_eV'];raw_f=d['raw_force_eV_A'];assert raw_e.shape==(128,) and raw_f.shape==(128,c['n_atoms'],3) and torch.isfinite(raw_e).all() and torch.isfinite(raw_f).all()
    e=(raw_e[:64]+raw_e[64:])/2;f=(raw_f[:64]-raw_f[64:])/2;torch.testing.assert_close(d['even_energy_eV'],e,atol=0,rtol=0);torch.testing.assert_close(d['even_force_eV_A'],f,atol=0,rtol=0)
    e_rows.append(e.numpy()/c['n_atoms']);f_rows.append(f.square().sum(-1).mean(-1).sqrt().numpy());raw_count+=128
   reports[seed][method]=report['rows'];energy[seed][method]=np.stack(e_rows);force[seed][method]=np.stack(f_rows)
   summary[seed][method]={k:sum(r[k] for r in report['rows']) for k in ['attempted','graph_supported','geometrically_supported','distinct_connectivity','validator_errors','generation_seconds']}
   artifacts.append(dict(seed=seed,method=method,checkpoint=str(ckpt),checkpoint_sha256=sha(ckpt),report_sha256=sha(file),protocol_sha256=ph))
   print(json.dumps(dict(seed=seed,method=method,summary=summary[seed][method])),flush=True)
  quality=json.loads((root/'physical_eval/results.json').read_text());assert quality['complete'] and quality['protocol_sha256']==ph and quality['raw_queries']==done['evaluation_raw_queries']==5120
  for row in quality['rows']:
   assert sha(root/'physical_eval'/row['artifact'])==row['artifact_sha256']
  assert len(quality['rows'])==40 and done['total_raw_queries']==teacher_calls+5120
 assert total_fit==512 and total_eval==5120 and raw_count==10240
 comparisons={};rng=np.random.default_rng(37391)
 for left,right in [('replay','frozen'),('escort','replay'),('work','escort'),('work','replay'),('work','frozen')]:
  result={};masks=np.stack([np.stack([flags(x,'graph_supported').astype(bool)&flags(y,'graph_supported').astype(bool) for x,y in zip(reports[s][left],reports[s][right])]) for s in [0,1]])
  for label,source in [('energy_per_atom_eV',energy),('force_rms_eV_A',force)]:
   diff=np.stack([source[s][left]-source[s][right] for s in [0,1]]);result[label]=dict(all_outputs=intervals(diff,[0]*10,rng),common_graph_supported=masked_intervals(diff,masks))
   result[label]['common_graph_by_seed']=[masked_intervals(diff[s:s+1],masks[s:s+1],seed=37392+s) for s in [0,1]]
  for metric in ['graph_supported','geometrically_supported']:
   diff=np.stack([np.stack([flags(x,metric)-flags(y,metric) for x,y in zip(reports[s][left],reports[s][right])]) for s in [0,1]]);result[metric]=intervals(diff,[0]*10,rng)
  comparisons[left+' minus '+right]=result
 primary=comparisons['work minus escort']['energy_per_atom_eV'];point=all(x['mean'] is not None and x['mean']<0 for x in primary['common_graph_by_seed']);bounds=primary['common_graph_supported']['paired_draw95'] is not None and primary['common_graph_supported']['paired_draw95'][1]<0
 support=all(sum(summary[s]['work'][metric]-summary[s][m][metric] for s in [0,1])/1280>=-.02 for metric in ['graph_supported','distinct_connectivity'] for m in ['frozen','replay','escort'])
 write(a.out,dict(complete=True,summary=summary,comparisons=comparisons,teacher=teacher,artifacts=artifacts,costs=costs,fit_generation_outputs_replayed=total_fit,evaluation_outputs_replayed=total_eval,teacher_raw_queries=sum(t['raw_queries'] for t in teacher),evaluation_raw_queries=raw_count,work_weighting_gate_passed=point and bounds and support,scientific_submission_ready=False,
  scope='Exact local escort/reference work reconstruction and all saved outputs audited; no independent oracle or neural-generation rerun. Finite-particle local ESS is not global ESS. Local Gaussian-restrained targets and distillation do not establish global Boltzmann law, chemical-isomer weights, quantum accuracy or new identity novelty.'))


if __name__=='__main__':main()
