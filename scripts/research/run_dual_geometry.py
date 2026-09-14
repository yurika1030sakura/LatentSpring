#!/usr/bin/env python3
"""Zero-parameter internal geometry controls with existing matching baselines."""
import argparse,json,time
from pathlib import Path
import torch
from flowmol.model_utils.load import read_config_file
from cfm_mol.clamped_fm import clamped_fm_loss
from cfm_mol.dual_geometry import patch_dual_geometry
from cfm_mol.source_checkpoint import prior_from_checkpoint
from cfm_mol.tree_mixture_prior import TreeMixturePrior
from cfm_mol.chemical_moves import covalent_radii
from scripts.research.tree_prior_fm import restore_model,sample_source,evaluate
from scripts.research.run_tree_manifold import make_graph
from scripts.research.train_electronic_fm import sha
from scripts.research.evaluate_chemical_policy import write


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['project','protocol','out']:p.add_argument('--'+key,type=Path,required=True)
 a=p.parse_args();spec=json.loads(a.protocol.read_text());ph=sha(a.protocol);assert spec['frozen'];torch.set_num_threads(2)
 if a.out.exists():raise FileExistsError(a.out)
 a.out.mkdir(parents=True)
 datafile=a.project/spec['data'];assert sha(datafile)==spec['data_sha256'];data=torch.load(datafile,map_location='cpu',weights_only=False)
 assert len(data['training'])==spec['fm_steps']
 cfg_path=a.project/spec['config'];assert sha(cfg_path)==spec['config_sha256'];cfg=read_config_file(cfg_path);cfg['mol_fm'].pop('bgfm',None)
 assert cfg['dataset']['max_atoms']==200 and cfg['mol_fm']['total_loss_weights']['e']==0
 warm_path=a.project/spec['warm_checkpoint'];assert sha(warm_path)==spec['warm_checkpoint_sha256'];warm=torch.load(warm_path,map_location='cpu',weights_only=False)
 prior=TreeMixturePrior('fixed',width=spec['edge_log_width']).double()
 manifest=a.project/spec['condition_manifest'];assert sha(manifest)==spec['condition_manifest_sha256']
 evaluation=a.out/'evaluation';evaluation.mkdir();completed=[]
 for name,reference in spec['references'].items():
  file=a.project/reference['checkpoint'];assert sha(file)==reference['checkpoint_sha256']
  saved=torch.load(file,map_location='cpu',weights_only=False);recipe=saved['research_protocol']
  for key in ['data_sha256','fm_seed','fm_steps','fm_lr','edge_log_width']:
   assert recipe[key]==spec[key],('Unmatched baseline',key)
  model=restore_model(cfg,saved);reference_prior=prior_from_checkpoint(saved)
  protocol=dict(spec,midpoint_steps=spec['method_midpoint_steps'][name])
  evaluate(model,reference_prior,name,cfg,protocol,ph,evaluation,reference['checkpoint_sha256'],manifest)
  completed.append(name);del model,saved,reference_prior;torch.cuda.empty_cache()
 for method in spec['training_methods']:
  torch.manual_seed(spec['fm_seed']);model=restore_model(cfg,warm).train();model._research_prior_kind='fixed'
  patch_dual_geometry(model,mode=spec['geometry_modes'][method])
  directory=a.out/method;directory.mkdir()
  optimizer=torch.optim.AdamW(model.parameters(),lr=spec['fm_lr'],weight_decay=1e-12)
  torch.manual_seed(spec['fm_seed']+17);start=time.perf_counter()
  for step,row in enumerate(data['training'],1):
   c=row['condition'];g,nbi,uem=make_graph(c,cfg)
   g.ndata['x_1_true']=row['positions'].cuda().float();g.ndata['has_reference_geometry']=torch.ones(c['n_atoms'],1,dtype=torch.bool,device='cuda')
   x0,_=sample_source(prior,c['numbers'],c['charge'],c['spin_multiplicity'],spec['fm_seed']*3000017+step)
   loss=clamped_fm_loss(model,g,nbi,uem,terminal_time=1.,parameterization='displacement',prior_positions=x0,
       pairing='typed_rotation',pairing_radii=covalent_radii(c['numbers'],device='cuda',dtype=torch.float32),
       generator=torch.Generator(device='cuda').manual_seed(spec['fm_seed']*1000003+step),pairing_generator=torch.Generator(device='cuda').manual_seed(spec['fm_seed']*2000003+step))
   if not torch.isfinite(loss):raise FloatingPointError('Nonfinite feedback FM loss')
   optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
   record=dict(step=step,processed_index=c['processed_index'],fm_loss=float(loss.detach()),gradient_norm=float(norm),seconds=time.perf_counter()-start)
   with (directory/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
   if step%100==0:print(json.dumps(dict(method=method,**record)),flush=True)
  recipe={**warm['research_protocol'],**spec,'format':'dual_geometry_v1','source_prior_kind':'fixed','model_variant':method,
      'dual_geometry_protocol_sha256':ph,'position_parameterization':'displacement','dual_geometry':dict(mode=spec['geometry_modes'][method])}
  torch.save(dict(state_dict=model.state_dict(),research_protocol=recipe,global_step=spec['fm_steps'],optimizer_state_dict=optimizer.state_dict(),source_prior=dict(configuration=prior.configuration,state_dict=prior.state_dict())),directory/'last.ckpt')
  write(directory/'training.json',dict(complete=True,steps=spec['fm_steps'],seconds=time.perf_counter()-start,checkpoint_sha256=sha(directory/'last.ckpt'),primitive_denoiser_training_forwards=spec['fm_steps']))
  protocol=dict(spec,midpoint_steps=spec['method_midpoint_steps'][method])
  evaluate(model,prior,method,cfg,protocol,ph,evaluation,sha(directory/'last.ckpt'),manifest)
  completed.append(method);write(a.out/'progress.json',dict(complete=len(completed)==len(spec['methods']),protocol_sha256=ph,completed=completed))
  del model,optimizer;torch.cuda.empty_cache()


if __name__=='__main__':main()
