#!/usr/bin/env python3
"""Audit training isolation, exact source replay, support and projected controls."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from flowmol.model_utils.load import model_from_config,read_config_file
from cfm_mol.degree_tree import CoordinationTreePrior,tree_degrees,ORGANIC_CAPS
from cfm_mol.tree_manifold import geometric_tree,tree_geometry,to_coordinates
from cfm_mol.radial_reference import prepare_research_backbone
from cfm_mol.electronic_metadata import ElectronicMetadata
from scripts.research.run_tree_manifold import source
from scripts.research.tree_prior_fm import geometry_counts
from scripts.research.audit_generator_output_support import assess
from scripts.research.audit_source_utility import intervals,flags
from scripts.research.train_electronic_fm import sha
from scripts.research.evaluate_chemical_policy import write


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['project','run','out']:p.add_argument('--'+key,type=Path,required=True)
 a=p.parse_args();torch.set_num_threads(2)
 if a.out.exists():raise FileExistsError(a.out)
 metadata=ElectronicMetadata(a.project/'runs/raw_metadata_replay_v1','train',list(range(1,84)))
 first=json.loads((a.project/'research/evidence/tree_manifold_s0_v1.json').read_text())
 datafile=Path(first['processed_train']);metadata.verify_processed_file(datafile)
 raw=torch.load(str(datafile),mmap=True,map_location='cpu',weights_only=False)
 reports={};summaries={};artifacts=[];preparation={};unique_draws=0;method_outputs=0;training_rows=0;support=[]
 for seed in [0,1]:
  path=a.project/f'research/evidence/tree_manifold_s{seed}_v1.json';spec=json.loads(path.read_text());ph=sha(path)
  root=a.run/f's{seed}';study=root/'study';prepared=root/'data';progress=json.loads((study/'progress.json').read_text())
  assert progress['complete'] and progress['protocol_sha256']==ph and progress['completed']==spec['training_methods']
  selection=json.loads((prepared/'selection.json').read_text());assert selection['complete'] and selection['protocol_sha256']==ph
  assert sha(prepared/'data.pt')==selection['data_sha256']
  data=torch.load(prepared/'data.pt',map_location='cpu',weights_only=False)
  excluded=set()
  for name,digest in spec['training_exclusion_manifests'].items():
   file=a.project/name;assert sha(file)==digest
   excluded.update(r['composition_hex'] for r in json.loads(file.read_text())['rows'])
  for row in data['training']+data['prior_validation']:
   c=row['condition'];i=c['processed_index'];lo,hi=map(int,raw['node_idx_array'][i]);numbers=(raw['atom_types'][lo:hi].long()+1).tolist()
   assert numbers==c['numbers']
   expected=raw['positions'][lo:hi].double();expected=expected-expected.mean(0)
   torch.testing.assert_close(row['positions'],expected,atol=0,rtol=0)
   counts=np.zeros(83,dtype=np.uint16)
   for z in numbers:counts[z-1]+=1
   key=counts.astype(np.uint8).tobytes().hex();assert key==c['composition_hex'] and key not in excluded
   accepted=metadata.indices[i];assert metadata.values['total_charge'][accepted]==c['charge']==0 and metadata.values['spin_multiplicity'][accepted]==c['spin_multiplicity']==1
   tree=geometric_tree(expected,numbers,spec['radial_lower']+spec['reference_margin'],spec['radial_upper']-spec['reference_margin'])
   assert tree==row['tree']
   b,inv,length=tree_geometry(tree,numbers,like=expected)
   for actual,wanted in [(row['incidence'],b),(row['inverse'],inv),(row['lengths'],length)]:torch.testing.assert_close(actual,wanted)
   rebuilt=to_coordinates(row['radial_logits'],row['directions'],inv,length,spec['radial_lower'],spec['radial_upper'])
   torch.testing.assert_close(rebuilt,expected,atol=2e-10,rtol=1e-11)
   assessment=assess(expected[None],c,[0]);assert assessment['graph_supported']==1 and assessment['validator_errors']==0
   training_rows+=1
  ids=[r['condition']['processed_index'] for r in data['training']];assert len(ids)==spec['fm_steps']
  assert ids+[r['condition']['processed_index'] for r in data['prior_validation']]==selection['selected_indices']
  prior_state=torch.load(study/'prior.pt',map_location='cpu',weights_only=False)
  assert prior_state['kind']=='coordination_tree_v1' and prior_state['protocol_sha256']==ph
  prior=CoordinationTreePrior(**prior_state['configuration']).double();prior.load_state_dict(prior_state['state_dict'],strict=True);prior.requires_grad_(False)
  prior_report=json.loads((study/'prior_results.json').read_text());assert prior_report['checkpoint_sha256']==sha(study/'prior.pt')
  preparation[seed]=dict(data_preparation_seconds=selection['seconds'],prior_fit_and_validation_seconds=prior_report['seconds'],model_training={})
  cfgpath=a.project/spec['config'];assert sha(cfgpath)==spec['config_sha256'];cfg=read_config_file(cfgpath);cfg['mol_fm'].pop('bgfm',None)
  adapters=[];checkpoint_hashes={}
  for method in spec['training_methods']:
   directory=study/method;train=json.loads((directory/'training.json').read_text());file=directory/'last.ckpt'
   assert train['complete'] and train['steps']==spec['fm_steps'] and sha(file)==train['checkpoint_sha256']
   state=torch.load(file,map_location='cpu',weights_only=False);recipe=state['research_protocol']
   assert recipe['tree_manifold_protocol_sha256']==ph and state['global_step']==spec['fm_steps']
   assert recipe['position_parameterization']==('displacement' if method=='cartesian' else 'tree_product')
   assert all(torch.equal(state['source_prior']['state_dict'][k],v) for k,v in prior.state_dict().items())
   model=model_from_config(cfg);prepare_research_backbone(model,recipe);model.load_state_dict(state['state_dict'],strict=True);del model,state
   metrics=[json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
   assert [r['processed_index'] for r in metrics]==ids and [r['step'] for r in metrics]==list(range(1,len(ids)+1))
   adapters.append(torch.load(directory/'adapter_initial.pt',map_location='cpu',weights_only=False));checkpoint_hashes[method]=train['checkpoint_sha256']
   preparation[seed]['model_training'][method]=train['seconds']
  assert all(torch.equal(v,adapters[1][k]) for k,v in adapters[0].items())
  panelpath=a.project/spec['condition_manifest'];assert sha(panelpath)==spec['condition_manifest_sha256'];panel=json.loads(panelpath.read_text())
  reports[seed]={};summaries[seed]={}
  for method in spec['methods']:
   report_path=study/'evaluation'/f'{method}_results.json';report=json.loads(report_path.read_text());assert report['complete'] and report['protocol_sha256']==ph
   assert report['checkpoint_sha256']==checkpoint_hashes['cartesian' if method=='cartesian_projected' else method]
   assert len(report['rows'])==12
   for index,row in enumerate(report['rows']):
    file=study/'evaluation'/f'{method}_c{index}.pt';assert sha(file)==row['sample_sha256'];saved=torch.load(file,map_location='cpu',weights_only=False)
    c=saved['condition'];assert c['composition_hex']==panel['rows'][index]['composition_hex']
    assert len(saved['positions'])==len(saved['initial_positions'])==64 and torch.isfinite(saved['positions']).all()
    expected_seeds=[spec['evaluation_seed']*1000003+index*100003+j for j in range(64)];assert saved['seeds']==expected_seeds
    cart=None
    if method=='cartesian_projected':cart=torch.load(study/'evaluation'/f'cartesian_c{index}.pt',map_location='cpu',weights_only=False)
    for j,draw_seed in enumerate(expected_seeds):
     tree,state=source(prior,c,draw_seed,spec);y,u,b,inv,length=state
     assert tree==saved['auxiliary_tree_edges'][j]
     x0=to_coordinates(y,u,inv,length,spec['radial_lower'],spec['radial_upper'])
     torch.testing.assert_close(x0,saved['initial_positions'][j],atol=2e-10,rtol=1e-11)
     if cart is not None:
      edge=b@cart['positions'][j];r=edge.norm(dim=-1);direction=torch.where((r>1e-12)[:,None],edge/r.clamp_min(1e-12)[:,None],u)
      constrained=torch.maximum(torch.minimum(r,spec['radial_upper']*length),spec['radial_lower']*length)
      projected=inv@(constrained[:,None]*direction)
      torch.testing.assert_close(projected,saved['positions'][j],atol=2e-9,rtol=2e-10)
     if method!='cartesian':
      ratio=(b@saved['positions'][j]).norm(dim=-1)/length
      assert float(ratio.min())>=spec['radial_lower']-1e-8 and float(ratio.max())<=spec['radial_upper']+1e-8
     method_outputs+=1
     if method!='cartesian_projected':unique_draws+=1
    result=assess(saved['positions'],c,list(range(64)))
    assert all(result[k]==row[k] for k in result)
    assert geometry_counts(saved['positions'],c['numbers'])==row['final_geometry']
    if method!='cartesian':assert row['final_geometry']['disconnected']==0
   rows=report['rows'];reports[seed][method]=rows
   summaries[seed][method]=dict(attempts=sum(r['attempted'] for r in rows),graph=sum(r['graph_supported'] for r in rows),geometry=sum(r['geometrically_supported'] for r in rows),
      distinct_connectivity=sum(r['distinct_connectivity'] for r in rows),disconnected=sum(r['final_geometry']['disconnected'] for r in rows),overlap=sum(r['final_geometry']['overlap'] for r in rows),
      generation_seconds=sum(r['generation_seconds'] for r in rows),validator_errors=sum(r['validator_errors'] for r in rows))
   artifacts.append(dict(seed=seed,method=method,report_sha256=sha(report_path),checkpoint_sha256=report['checkpoint_sha256'],selection_sha256=sha(prepared/'selection.json'),data_sha256=sha(prepared/'data.pt')))
   print(json.dumps(dict(seed=seed,method=method,**summaries[seed][method])),flush=True)
 strata=[0 if c['n_atoms']<=16 else 1 if c['n_atoms']<=28 else 2 for c in panel['rows']];rng=np.random.default_rng(33191);comparisons={}
 for control in ['cartesian','cartesian_projected']:
  comparisons[control]={}
  for metric in ['graph_supported','geometrically_supported']:
   diff=np.stack([np.stack([flags(x,metric)-flags(y,metric) for x,y in zip(reports[s]['manifold'],reports[s][control])]) for s in [0,1]])
   comparisons[control][metric]=intervals(diff,strata,rng)
 point=all(summaries[s]['manifold']['graph']>summaries[s][m]['graph'] for s in [0,1] for m in ['cartesian','cartesian_projected'])
 bounds=all(comparisons[m]['graph_supported']['paired_draw95'][0]>0 for m in ['cartesian','cartesian_projected'])
 distinct=sum(summaries[s]['manifold']['distinct_connectivity']-summaries[s]['cartesian_projected']['distinct_connectivity'] for s in [0,1])/1536
 write(a.out,dict(complete=True,summary=summaries,comparisons=comparisons,preparation=preparation,artifacts=artifacts,
  original_training_rows_and_scaffolds_replayed=training_rows,unique_neural_outputs_replayed=unique_draws,reported_method_outputs_replayed=method_outputs,
  source_coordination_and_radial_support_passed=True,projection_control_replayed=True,full_checkpoint_restoration=True,matched_training_indices_and_initial_adapters=True,
  development_gate_passed=point and bounds and distinct>=-.02,new_molecular_oracle_calls=0,scientific_submission_ready=False,
  scope='Reused12-composition development panel. Contact connectivity does not guarantee graph validity, non-edge sterics or energy law. Projection shares Cartesian draws/timer; no extra neural evaluations. No independent optimizer or full neural-trajectory replay.'))


if __name__=='__main__':main()
