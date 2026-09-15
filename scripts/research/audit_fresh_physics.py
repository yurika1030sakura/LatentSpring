#!/usr/bin/env python3
"""Audit fresh panel, every source/output and all eSEN readouts before GFN2."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from cfm_mol.source_checkpoint import prior_from_checkpoint
from scripts.research.tree_prior_fm import sample_source,geometry_counts
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
 panelpath=a.project/'research/evidence/fresh_physics_panel_v1.json';panel=json.loads(panelpath.read_text());poolfile=a.project/panel['source_pool'];assert sha(poolfile)==panel['source_pool_sha256'];pool=json.loads(poolfile.read_text());qfile=a.project/panel['qualification_audit'];assert sha(qfile)==panel['qualification_audit_sha256'];qualification=json.loads(qfile.read_text())
 overlapfile=a.project/'research/evidence/fresh_physics_overlap_v1.json';overlap=json.loads(overlapfile.read_text());assert overlap['complete'] and overlap['panel_sha256']==sha(panelpath) and all(r['overlapping_compositions']==0 for r in overlap['corpora'].values())
 excluded=set()
 for name,digest in pool['excluded_manifests'].items():
  path=a.project/'research/evidence'/f'{name}.json';assert sha(path)==digest;excluded.update(r['composition_hex'] for r in json.loads(path.read_text())['rows'])
 accepted=[]
 for i,r in enumerate(pool['rows']):
  c=dict(r['condition'],numbers=r['condition']['atomic_numbers']);key=np.bincount(np.asarray(c['numbers'])-1,minlength=83).astype(np.uint8).tobytes();assert key.hex()==c['composition_hex'] and c['composition_hex'] not in excluded
  assert int.from_bytes(hashlib.sha256(str(pool['old_partition_seed']).encode()+key).digest()[:8],'big')%5==0
  assert c['selection_rank']==hashlib.sha256(('37791|'+c['composition_hex']).encode()).hexdigest() and c['charge']==0 and c['spin_multiplicity']==1
  x=torch.tensor(r['reference_positions'],dtype=torch.float64)[None];result=assess(x,c,[0]);geom=geometry_counts(x,c['numbers']);decision=qualification['decisions'][i]
  assert result==decision['assessment'] and geom==decision['geometry'];passed=result['graph_supported']==1 and result['validator_errors']==0 and geom==dict(disconnected=0,overlap=0);assert passed==decision['qualified']
  if passed:accepted.append((c['selection_rank'],i,r['condition']))
 selected=[]
 for lo,hi in [(17,20),(21,24),(25,28)]:
  group=sorted((rank,i,c) for rank,i,c in accepted if lo<=c['n_atoms']<=hi)
  selected.extend(dict(c,pool_index=i,panel_stratum=[lo,hi,'neutral','singlet']) for _,i,c in group[:8])
 assert selected==panel['rows'] and len(selected)==24 and all(not any(k in c for k in ['positions','reference_positions','energy_eV','forces']) for c in selected)
 methods=['gaussian','harmonic_tree','escort_delta','work_delta'];shape=(2,4,24,32);energies=np.zeros(shape);forces=np.zeros(shape);graph=np.zeros(shape,dtype=bool);geometry=np.zeros(shape,dtype=bool);summary={};artifacts=[];counts=0;physical=0;reports={}
 for seed in [0,1]:
  path=a.project/f'research/evidence/fresh_physics_s{seed}_v1.json';spec=json.loads(path.read_text());ph=sha(path);root=a.run/f's{seed}/study';done=json.loads((root/'complete.json').read_text());assert spec['frozen'] and spec['methods']==methods and done['complete'] and done['protocol_sha256']==ph and done['new_training_steps']==0
  summary[seed]={};reports[seed]={};quality=json.loads((root/'physical_eval/results.json').read_text());assert quality['complete'] and quality['protocol_sha256']==ph and quality['raw_queries']==done['raw_esen_queries']==6192 and len(quality['rows'])==120
  for r in quality['rows']:assert sha(root/'physical_eval'/r['artifact'])==r['artifact_sha256']
  for i,c in enumerate(selected):
   ref=torch.load(root/'physical_eval'/f'reference_c{i}.pt',map_location='cpu',weights_only=False);torch.testing.assert_close(ref['positions'],torch.tensor(pool['rows'][c['pool_index']]['reference_positions'],dtype=torch.float64),atol=0,rtol=0);assert ref['raw_energy_eV'].shape==(2,) and ref['raw_force_eV_A'].shape==(2,c['n_atoms'],3);physical+=2
  for mi,method in enumerate(methods):
   checkpoint=a.project/spec['checkpoints'][method]['path'];digest=sha(checkpoint);assert digest==spec['checkpoints'][method]['sha256'];state=torch.load(checkpoint,map_location='cpu',weights_only=False);prior=prior_from_checkpoint(state);assert state['research_protocol']['source_prior_kind']==spec['source_kinds'][method];del state
   file=root/'evaluation'/f'{method}_results.json';report=json.loads(file.read_text());assert report['complete'] and report['protocol_sha256']==ph and report['checkpoint_sha256']==digest and [r['condition_index'] for r in report['rows']]==list(range(24))
   for i,row in enumerate(report['rows']):
    path=root/'evaluation'/f'{method}_c{i}.pt';assert sha(path)==row['sample_sha256'];sample=torch.load(path,map_location='cpu',weights_only=False);c=sample['condition'];assert c==row['condition'] and c['composition_hex']==selected[i]['composition_hex'] and c['atomic_numbers']==selected[i]['atomic_numbers']
    assert sample['positions'].shape==(32,c['n_atoms'],3);expected=[spec['evaluation_seed']*1000003+i*100003+j for j in range(32)];assert sample['seeds']==expected
    for j,draw_seed in enumerate(expected):
     x,t=sample_source(prior,c['numbers'],c['charge'],c['spin_multiplicity'],draw_seed);torch.testing.assert_close(x,sample['initial_positions'][j],atol=0,rtol=0);assert t==sample['auxiliary_tree_edges'][j];counts+=1
    result=assess(sample['positions'],c,list(range(32)));assert all(row[k]==v for k,v in result.items());assert geometry_counts(sample['positions'],c['numbers'])==row['final_geometry']
    d=torch.load(root/'physical_eval'/f'{method}_c{i}.pt',map_location='cpu',weights_only=False);assert d['source_sample_sha256']==sha(path);torch.testing.assert_close(d['positions'],sample['positions'],atol=0,rtol=0)
    e=d['raw_energy_eV'];f=d['raw_force_eV_A'];assert e.shape==(64,) and f.shape==(64,c['n_atoms'],3) and torch.isfinite(e).all() and torch.isfinite(f).all();ep=(e[:32]+e[32:])/2;fp=(f[:32]-f[32:])/2;physical+=64
    torch.testing.assert_close(d['even_energy_eV'],ep,atol=0,rtol=0);torch.testing.assert_close(d['even_force_eV_A'],fp,atol=0,rtol=0)
    energies[seed,mi,i]=ep.numpy()/c['n_atoms'];forces[seed,mi,i]=fp.square().sum(-1).mean(-1).sqrt().numpy();graph[seed,mi,i]=flags(row,'graph_supported').astype(bool);geometry[seed,mi,i]=flags(row,'geometrically_supported').astype(bool)
   reports[seed][method]=report['rows'];summary[seed][method]={k:sum(r[k] for r in report['rows']) for k in ['attempted','graph_supported','geometrically_supported','distinct_connectivity','validator_errors','generation_seconds']};artifacts.append(dict(seed=seed,method=method,checkpoint_sha256=digest,report_sha256=sha(file),protocol_sha256=ph,physical_manifest_sha256=sha(root/'physical_eval/results.json')))
   print(json.dumps(dict(seed=seed,method=method,summary=summary[seed][method])),flush=True)
 assert counts==6144 and physical==12384
 comparisons={};rng=np.random.default_rng(37991);strata=[i//8 for i in range(24)]
 for left,right in [('harmonic_tree','gaussian'),('escort_delta','harmonic_tree'),('work_delta','harmonic_tree'),('work_delta','escort_delta')]:
  li,ri=methods.index(left),methods.index(right);mask=graph[:,li]&graph[:,ri];r={}
  for name,array in [('energy_per_atom_eV',energies),('force_rms_eV_A',forces)]:
   diff=array[:,li]-array[:,ri];r[name]=dict(all_outputs=intervals(diff,strata,rng),common_graph_supported=masked_intervals(diff,mask),common_graph_by_seed=[masked_intervals(diff[s:s+1],mask[s:s+1],37992+s) for s in [0,1]])
  r['graph_supported']=intervals(graph[:,li].astype(float)-graph[:,ri].astype(float),strata,rng);r['geometrically_supported']=intervals(geometry[:,li].astype(float)-geometry[:,ri].astype(float),strata,rng);comparisons[left+' minus '+right]=r
 arrayfile=a.out.with_suffix('.npz');np.savez_compressed(arrayfile,energy_per_atom=energies,force_rms=forces,graph=graph,geometry=geometry)
 write(a.out,dict(complete=True,methods=methods,summary=summary,comparisons=comparisons,artifacts=artifacts,arrays_file=arrayfile.name,arrays_sha256=sha(arrayfile),generation_run=str(a.run.resolve()),panel_sha256=sha(panelpath),reference_pool_sha256=sha(poolfile),qualification_references_replayed=256,generated_outputs_replayed=counts,raw_esen_rows_checked=physical,independent_xtb_pending=True,scientific_submission_ready=False,scope='Frozen fresh24-composition confirmation in17-28-atom neutral singlets, two existing model seeds. eSEN only at this stage; independent evaluator and its failure coverage are required for the primary physical confirmation gate.'))


if __name__=='__main__':main()
