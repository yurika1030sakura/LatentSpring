#!/usr/bin/env python3
"""Matched Cartesian versus product-space tree flow, with a projection control."""
import argparse,json,time,math
from pathlib import Path
import numpy as np
import torch
import dgl
from flowmol.model_utils.load import read_config_file
from flowmol.data_processing.utils import get_batch_idxs,get_upper_edge_mask
from cfm_mol.degree_tree import CoordinationTreePrior,ORGANIC_CAPS
from cfm_mol.tree_manifold import sample_product_source,to_coordinates,product_path,cap_angular,midpoint_product
from cfm_mol.latent_tree_context import patch_latent_tree_context,tree_features,attach_context
from cfm_mol.clamped_density import deterministic_field,position_velocity,sample_clamped_flow
from cfm_mol.condition_systems import graph_from_condition,load_condition
from cfm_mol.orbit_pairing import proper_alignment,haar_rotation
from scripts.research.tree_prior_fm import restore_model,geometry_counts
from scripts.research.train_electronic_fm import sha
from scripts.research.audit_generator_output_support import assess
from scripts.research.evaluate_chemical_policy import write


def make_graph(c,cfg,batch=1):
 g=dgl.batch([graph_from_condition(c,cfg['dataset']['atom_map'],n_bond_classes=5 if cfg['mol_fm'].get('explicit_aromaticity',False) else 4)]*batch).to('cuda')
 return g,*get_batch_idxs(g)[:1],get_upper_edge_mask(g)


def fit_prior(data,spec,out,ph):
 torch.manual_seed(spec['prior_seed']);prior=CoordinationTreePrior().double()
 optimizer=torch.optim.Adam(prior.parameters(),lr=spec['prior_lr']);metrics=[];start=time.perf_counter()
 for step,row in enumerate(data['training'][:spec['prior_steps']],1):
  c=row['condition'];loss=-prior.log_prob(row['tree'],c['numbers'],c['charge'],c['spin_multiplicity'])/max(1,c['n_atoms']-1)
  if not torch.isfinite(loss):raise FloatingPointError('Nonfinite coordination prior objective')
  optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(prior.parameters(),5.,error_if_nonfinite=True);optimizer.step()
  metrics.append(dict(step=step,processed_index=c['processed_index'],nll_per_edge=float(loss.detach()),gradient_norm=float(norm)))
  if step%100==0:print(json.dumps(dict(phase='coordination_prior',**metrics[-1])),flush=True)
 prior.eval();prior.requires_grad_(False)
 validation=[]
 uniform=CoordinationTreePrior().double().requires_grad_(False)
 for row in data['prior_validation']:
  c=row['condition'];args=(row['tree'],c['numbers'],c['charge'],c['spin_multiplicity'])
  validation.append(dict(processed_index=c['processed_index'],learned_nll=float(-prior.log_prob(*args)/(c['n_atoms']-1)),uniform_nll=float(-uniform.log_prob(*args)/(c['n_atoms']-1))))
 torch.save(dict(kind='coordination_tree_v1',configuration=prior.configuration,state_dict=prior.state_dict(),protocol_sha256=ph),out/'prior.pt')
 write(out/'prior_results.json',dict(complete=True,metrics=metrics,validation=validation,seconds=time.perf_counter()-start,checkpoint_sha256=sha(out/'prior.pt')))
 return prior


def source(prior,c,seed,spec):
 tree=prior.sample(c['numbers'],c['charge'],c['spin_multiplicity'],rng=np.random.default_rng(seed+810000003))
 state=sample_product_source(tree,c['numbers'],generator=torch.Generator().manual_seed(seed),width=spec['edge_log_width'],lower=spec['radial_lower'],upper=spec['radial_upper'])
 return tree,state


def evaluate(model,method,prior,cfg,spec,ph,manifest,out,checkpoint_hash):
 model.eval();model.requires_grad_(False);results=[];projected_results=[];batch=spec['evaluation_batch']
 for index in spec['conditions']:
  c=load_condition(manifest,index);c.update(numbers=c['atomic_numbers'],requested_kT_eV=1.)
  g,nbi,uem=make_graph(c,cfg,batch);n=c['n_atoms'];positions=[];initial=[];projected=[];trees=[];seeds=[];start=time.perf_counter()
  for begin in range(0,spec['samples_per_condition'],batch):
   states=[];matrices=[]
   for j in range(batch):
    seed=spec['evaluation_seed']*1000003+index*100003+begin+j
    tree,state=source(prior,c,seed,spec);trees.append(tree);seeds.append(seed);states.append(state);matrices.append(tree_features(n,tree))
   y,u,b,inv,length=[torch.stack([state[k] for state in states]).cuda() for k in range(5)]
   initial_x=torch.einsum('bne,bed->bnd',inv,(length*(spec['radial_lower']+(spec['radial_upper']-spec['radial_lower'])*y.sigmoid()))[...,None]*u)
   initial.append(initial_x.cpu());attach_context(g,matrices,nbi)
   with torch.no_grad(),deterministic_field(model.vector_field):
    if method=='cartesian':
     final=sample_clamped_flow(model,g,nbi,uem,x0=initial_x.reshape(-1,3).float(),n_ode_steps=spec['midpoint_steps'],terminal_time=1.,parameterization='displacement').reshape(batch,n,3).double()
     edge=torch.einsum('ben,bnd->bed',b,final);r=edge.norm(dim=-1)
     direction=torch.where((r>1e-12)[...,None],edge/r.clamp_min(1e-12)[...,None],u)
     constrained=torch.maximum(torch.minimum(r,spec['radial_upper']*length),spec['radial_lower']*length)
     projected.append(torch.einsum('bne,bed->bnd',inv,constrained[...,None]*direction).cpu())
    else:
     def field(yy,uu,t):
      r=length*(spec['radial_lower']+(spec['radial_upper']-spec['radial_lower'])*yy.sigmoid())
      x=torch.einsum('bne,bed->bnd',inv,r[...,None]*uu)
      v=position_velocity(model,g,x.reshape(-1,3).float(),torch.full((batch,),t,device='cuda'),nbi,uem,parameterization='displacement').reshape(batch,n,3).double()
      e=torch.einsum('ben,bnd->bed',b,v)/length[...,None];dy=(e*uu).sum(-1)
      if not torch.isfinite(e).all():raise FloatingPointError('Nonfinite tree-chart neural field')
      return dy,e-dy[...,None]*uu
     for step in range(spec['midpoint_steps']):y,u=midpoint_product(y,u,step/spec['midpoint_steps'],1/spec['midpoint_steps'],field)
     r=length*(spec['radial_lower']+(spec['radial_upper']-spec['radial_lower'])*y.sigmoid())
     final=torch.einsum('bne,bed->bnd',inv,r[...,None]*u)
    final=final-final.mean(1,keepdim=True)
    if not torch.isfinite(final).all():raise FloatingPointError('Nonfinite generated positions')
    positions.append(final.cpu())
  torch.cuda.synchronize();seconds=time.perf_counter()-start;x=torch.cat(positions);x0=torch.cat(initial)
  pairs=[(method,x,results)]
  if method=='cartesian':pairs.append(('cartesian_projected',torch.cat(projected),projected_results))
  for label,coordinates,target in pairs:
   file=out/f'{label}_c{index}.pt'
   torch.save(dict(positions=coordinates,initial_positions=x0,auxiliary_tree_edges=trees,seeds=seeds,condition=c,
       derived_from='cartesian' if label=='cartesian_projected' else None),file)
   row=dict(method=label,condition_index=index,condition=c,sample_sha256=sha(file),checkpoint_sha256=checkpoint_hash,generation_seconds=seconds,
       final_geometry=geometry_counts(coordinates,c['numbers']),**assess(coordinates,c,list(range(len(coordinates)))))
   target.append(row)
   print(json.dumps({k:row[k] for k in ['method','condition_index','graph_supported','geometrically_supported','final_geometry']}),flush=True)
  if method=='manifold' and results[-1]['final_geometry']['disconnected']!=0:raise AssertionError('Tree manifold lost contact connectivity')
 for label,rows in [(method,results),('cartesian_projected',projected_results)]:
  if not rows:continue
  write(out/f'{label}_results.json',dict(complete=True,protocol_sha256=ph,rows=rows,checkpoint_sha256=checkpoint_hash,
      scope='Projection shares Cartesian neural draws and its timer; no extra neural generation. No final density, thermal-law or guaranteed non-edge/chemical-validity claim.',new_molecular_oracle_calls=0))


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['project','protocol','data','out']:p.add_argument('--'+key,type=Path,required=True)
 a=p.parse_args();spec=json.loads(a.protocol.read_text());ph=sha(a.protocol);assert spec['frozen'];torch.set_num_threads(2)
 assert {int(k):v for k,v in spec['coordination_caps'].items()}==ORGANIC_CAPS
 if a.out.exists():raise FileExistsError(a.out)
 a.out.mkdir(parents=True);selection=json.loads((a.data/'selection.json').read_text())
 assert selection['complete'] and selection['protocol_sha256']==ph and selection['data_sha256']==sha(a.data/'data.pt')
 data=torch.load(a.data/'data.pt',map_location='cpu',weights_only=False)
 cfg_path=a.project/spec['config'];assert sha(cfg_path)==spec['config_sha256'];cfg=read_config_file(cfg_path);cfg['mol_fm'].pop('bgfm',None)
 assert cfg['dataset']['max_atoms']==200 and cfg['mol_fm']['total_loss_weights']['e']==0
 prior=fit_prior(data,spec,a.out,ph)
 file=a.project/spec['warm_checkpoint'];assert sha(file)==spec['warm_checkpoint_sha256'];warm=torch.load(file,map_location='cpu',weights_only=False)
 manifest=a.project/spec['condition_manifest'];assert sha(manifest)==spec['condition_manifest_sha256']
 evaluation=a.out/'evaluation';evaluation.mkdir();completed=[]
 for method in ['cartesian','manifold']:
  torch.manual_seed(spec['fm_seed']);model=restore_model(cfg,warm).train()
  torch.manual_seed(spec['context_seed']);patch_latent_tree_context(model,hidden=spec['context_hidden'])
  directory=a.out/method;directory.mkdir();torch.save(model.vector_field.latent_tree_adapter.state_dict(),directory/'adapter_initial.pt')
  extra=list(model.vector_field.latent_tree_adapter.parameters());ids={id(p) for p in extra}
  optimizer=torch.optim.AdamW([dict(params=[p for p in model.parameters() if id(p) not in ids],lr=spec['fm_lr']),dict(params=extra,lr=spec['context_lr'])],weight_decay=1e-12)
  start=time.perf_counter();torch.manual_seed(spec['fm_seed']+17)
  for step,row in enumerate(data['training'],1):
   c=row['condition'];g,nbi,uem=make_graph(c,cfg)
   g.ndata['x_1_true']=row['positions'].cuda().float();g.ndata['has_reference_geometry']=torch.ones(c['n_atoms'],1,dtype=torch.bool,device='cuda')
   attach_context(g,[row['context']],nbi)
   y0,u0,b,inv,length=sample_product_source(row['tree'],c['numbers'],generator=torch.Generator().manual_seed(spec['fm_seed']*3000017+step),width=spec['edge_log_width'],lower=spec['radial_lower'],upper=spec['radial_upper'])
   x0=to_coordinates(y0,u0,inv,length,spec['radial_lower'],spec['radial_upper']);target=row['positions']
   rotation=proper_alignment(x0,target);augmentation=haar_rotation(x0,torch.Generator().manual_seed(spec['fm_seed']*2000003+step))
   u0=u0@rotation@augmentation;x0=x0@rotation@augmentation;target=target@augmentation
   u1=row['directions']@augmentation;y1=row['radial_logits']
   t=float(torch.rand((),generator=torch.Generator().manual_seed(spec['fm_seed']*1000003+step)))
   if method=='cartesian':x=(1-t)*x0+t*target;teacher=b@(target-x0)/length[:,None]
   else:
    y,u,dy,du=product_path(y0,u0,y1,u1,t);x=to_coordinates(y,u,inv,length,spec['radial_lower'],spec['radial_upper']);teacher=dy[:,None]*u+du
   with deterministic_field(model.vector_field):
    velocity=position_velocity(model,g,x.cuda().float(),torch.tensor([t],device='cuda'),nbi,uem,parameterization='displacement')
   edge=b.cuda()@velocity.double()/length.cuda()[:,None]
   if method=='manifold':
    unit=u.cuda();dyhat=(edge*unit).sum(-1);edge=dyhat[:,None]*unit+cap_angular(edge-dyhat[:,None]*unit)
   loss=(edge-teacher.cuda()).square().mean()
   if not torch.isfinite(loss):raise FloatingPointError('Nonfinite matched tree-flow objective')
   optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
   record=dict(step=step,processed_index=c['processed_index'],fm_loss=float(loss.detach()),gradient_norm=float(norm),seconds=time.perf_counter()-start)
   with (directory/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
   if step%100==0:print(json.dumps(dict(method=method,**record)),flush=True)
  recipe={**warm['research_protocol'],**spec,'format':'tree_manifold_v1','source_prior_kind':'coordination_tree',
     'position_parameterization':'displacement' if method=='cartesian' else 'tree_product',
     'latent_tree_context':dict(hidden=spec['context_hidden']),'model_variant':method,'tree_manifold_protocol_sha256':ph}
  torch.save(dict(state_dict=model.state_dict(),research_protocol=recipe,global_step=spec['fm_steps'],source_prior=dict(kind='coordination_tree_v1',configuration=prior.configuration,state_dict=prior.state_dict()),optimizer_state_dict=optimizer.state_dict()),directory/'last.ckpt')
  write(directory/'training.json',dict(complete=True,seconds=time.perf_counter()-start,checkpoint_sha256=sha(directory/'last.ckpt'),steps=spec['fm_steps']))
  evaluate(model,method,prior,cfg,spec,ph,manifest,evaluation,sha(directory/'last.ckpt'))
  completed.append(method);write(a.out/'progress.json',dict(complete=len(completed)==2,protocol_sha256=ph,completed=completed))
  del model,optimizer;torch.cuda.empty_cache()


if __name__=='__main__':main()
