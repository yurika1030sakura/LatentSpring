#!/usr/bin/env python3
"""One bounded replay, force-escort and full-work endpoint distillation comparison."""
import argparse,json,gc,time
from pathlib import Path
import torch
from flowmol.model_utils.load import read_config_file
from cfm_mol.source_checkpoint import prior_from_checkpoint
from cfm_mol.clamped_density import sample_clamped_flow
from cfm_mol.clamped_fm import clamped_fm_loss
from cfm_mol.chemical_moves import covalent_radii
from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.escorted_thermal_teacher import make_escorted_teacher
from scripts.research.run_tree_manifold import make_graph
from scripts.research.tree_prior_fm import restore_model,sample_source,evaluate
from scripts.research.audit_generator_output_support import assess
from scripts.research.train_electronic_fm import sha
from scripts.research.evaluate_chemical_policy import write


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ['project','protocol','out']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();torch.set_num_threads(2);spec=json.loads(a.protocol.read_text());ph=sha(a.protocol);assert spec['frozen']
 if a.out.exists():raise FileExistsError(a.out)
 a.out.mkdir(parents=True);teacher_dir=a.out/'teacher';teacher_dir.mkdir();start=time.perf_counter()
 datafile=a.project/spec['data'];assert sha(datafile)==spec['data_sha256'];data=torch.load(datafile,map_location='cpu',weights_only=False)
 warmfile=a.project/spec['warm_checkpoint'];assert sha(warmfile)==spec['warm_checkpoint_sha256'];warm=torch.load(warmfile,map_location='cpu',weights_only=False)
 cfgfile=a.project/spec['config'];assert sha(cfgfile)==spec['config_sha256'];cfg=read_config_file(cfgfile);cfg['mol_fm'].pop('bgfm',None)
 assert cfg['dataset']['max_atoms']==200 and cfg['mol_fm']['total_loss_weights']['e']==0
 rows=[data['training'][i] for i in spec['teacher_training_rows']];assert len({r['condition']['composition_hex'] for r in rows})==spec['teacher_compositions']
 manifest=a.project/spec['condition_manifest'];assert sha(manifest)==spec['condition_manifest_sha256'];excluded={r['composition_hex'] for r in json.loads(manifest.read_text())['rows']};assert not excluded.intersection(r['condition']['composition_hex'] for r in rows)
 prior=prior_from_checkpoint(warm);model=restore_model(cfg,warm).eval();model.requires_grad_(False);teacher=[]
 for i,row in enumerate(rows):
  c=row['condition'];batch=spec['teacher_draws_per_composition'];g,nbi,uem=make_graph(c,cfg,batch)
  x0=torch.stack([sample_source(prior,c['numbers'],c['charge'],c['spin_multiplicity'],spec['teacher_seed']*1000003+i*100003+j)[0] for j in range(batch)])
  with torch.no_grad():y=sample_clamped_flow(model,g,nbi,uem,x0=x0.reshape(-1,3).cuda().float(),n_ode_steps=32,terminal_time=1.,parameterization='displacement').reshape(batch,c['n_atoms'],3).cpu().double()
  noise=torch.randn(y.shape,dtype=torch.float64,generator=torch.Generator().manual_seed(spec['teacher_seed']+i));noise-=noise.mean(1,keepdim=True);y+=spec['terminal_noise_std_A']*noise;y-=y.mean(1,keepdim=True)
  assessment=assess(y,c,list(range(batch)));keep=torch.tensor([r['graph_supported'] for r in assessment['records']],dtype=torch.bool)
  item=dict(condition=c,raw_positions=y,source_positions=x0,graph_supported=keep,assessment=assessment,role='FIT',training_processed_index=c['processed_index'])
  torch.save(item,teacher_dir/f'raw_c{i}.pt');teacher.append(item)
  print(json.dumps(dict(phase='teacher_generation',condition=i,valid=int(keep.sum()),attempted=batch)),flush=True)
 del model,g,nbi,uem;gc.collect();torch.cuda.empty_cache()
 worker=Path(__file__).resolve().parent/'oracle_worker.py';assert sha(worker)==spec['oracle_worker_sha256'];assert sha(spec['oracle_checkpoint'])==spec['oracle_sha256'];queries=0;requested=0;pools=[]
 for i,item in enumerate(teacher):
  c=item['condition'];x=item['raw_positions'][item['graph_supported']]
  if not len(x):continue
  with EnergyOracle(spec['oracle_interpreter'],worker,spec['oracle_checkpoint'],numbers=c['numbers'],charge=c['charge'],spin_multiplicity=c['spin_multiplicity'],device='cuda',batch_size=16) as oracle:
   assert oracle.handshake['base_precision_dtype']=='torch.float32' and not oracle.handshake['tf32']
   record=make_escorted_teacher(x,c,oracle,**spec['thermal_teacher'],seed=spec['teacher_seed']*100003+i)
   queries+=oracle.evaluated;requested+=oracle.requested_evaluations
   saved=dict(condition=c,raw_positions=x[record['eligible']],proposals=record['proposal'][record['eligible']],weights=record['weights'][record['eligible']],uniform_weights=record['uniform_weights'][record['eligible']],role='FIT',record=record,raw_source_sha256=sha(teacher_dir/f'raw_c{i}.pt'),oracle_queries=oracle.evaluated)
   torch.save(saved,teacher_dir/f'refined_c{i}.pt')
   if len(saved['raw_positions']):pools.append(saved)
   print(json.dumps(dict(phase='thermal_teacher',condition=i,anchors=len(x),retained=int(record['eligible'].sum()),mean_local_ess=float(record['local_particle_ess'].mean()),queries=oracle.evaluated)),flush=True)
  write(a.out/'teacher_progress.json',dict(complete=False,queries=queries,requested=requested,conditions=len(pools)))
 assert len(pools)>=spec['minimum_teacher_compositions'] and queries==requested
 write(a.out/'teacher_progress.json',dict(complete=True,queries=queries,requested=requested,conditions=len(pools),seconds=time.perf_counter()-start,role='FIT',protocol_sha256=ph))
 evaluation=a.out/'evaluation';evaluation.mkdir();completed=[]
 for method in spec['methods']:
  torch.manual_seed(spec['training_seed']);model=restore_model(cfg,warm);directory=a.out/method;directory.mkdir()
  if method=='frozen':checkpoint=warmfile
  else:
   model.train();model.requires_grad_(True);extra=list(model.vector_field.self_conditioning_residual_layer.parameters())+list(model.vector_field.to_edge_logits.parameters());extra_ids={id(p) for p in extra}
   opt=torch.optim.AdamW([dict(params=[p for p in model.parameters() if id(p) not in extra_ids],lr=spec['fm_lr']),dict(params=extra,lr=spec['feedback_lr'])],weight_decay=1e-12)
   rng=torch.Generator().manual_seed(spec['training_seed']+17);training_start=time.perf_counter()
   for step in range(1,spec['training_steps']+1):
    if step%2:
     idx=int(torch.randint(len(data['training']),(1,),generator=rng));entry=data['training'][idx];c=entry['condition'];target=entry['positions'];label=dict(kind='reference',row=idx)
    else:
     idx=int(torch.randint(len(pools),(1,),generator=rng));entry=pools[idx];j=int(torch.randint(len(entry['raw_positions']),(1,),generator=rng));c=entry['condition'];u=float(torch.rand((),generator=rng))
     if method=='replay':target=entry['raw_positions'][j];particle=-1
     else:
      weights=entry['weights' if method=='work' else 'uniform_weights'][j];particle=int(torch.searchsorted(weights.cumsum(0),u,right=True).clamp_max(len(weights)-1));target=entry['proposals'][j,particle];assert weights[particle]>0
     label=dict(kind='generated_fit',pool=idx,row=j,particle=particle,uniform_draw=u)
    g,nbi,uem=make_graph(c,cfg);g.ndata['x_1_true']=target.cuda().float();g.ndata['has_reference_geometry']=torch.ones(c['n_atoms'],1,dtype=torch.bool,device='cuda')
    x0,_=sample_source(prior,c['numbers'],c['charge'],c['spin_multiplicity'],spec['training_seed']*3000017+step)
    loss=clamped_fm_loss(model,g,nbi,uem,terminal_time=1.,parameterization='displacement',prior_positions=x0,pairing='typed_rotation',pairing_radii=covalent_radii(c['numbers'],device='cuda',dtype=torch.float32),generator=torch.Generator(device='cuda').manual_seed(spec['training_seed']*1000003+step),pairing_generator=torch.Generator(device='cuda').manual_seed(spec['training_seed']*2000003+step))
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite distillation loss')
    opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
    with (directory/'metrics.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,selection=label,composition=c['composition_hex'],loss=float(loss.detach()),gradient_norm=float(norm)))+'\n')
    if step%100==0:print(json.dumps(dict(method=method,step=step,loss=float(loss.detach()))),flush=True)
   recipe={**warm['research_protocol'],'thermal_distillation_protocol_sha256':ph,'thermal_distillation_method':method,'thermal_distillation_steps':spec['training_steps'],'thermal_teacher':spec['thermal_teacher'],'source_prior_kind':'harmonic_tree'}
   checkpoint=directory/'last.ckpt';torch.save(dict(state_dict=model.state_dict(),research_protocol=recipe,source_prior=warm['source_prior'],optimizer_state_dict=opt.state_dict(),global_step=spec['training_steps']),checkpoint)
   write(directory/'training.json',dict(complete=True,seconds=time.perf_counter()-training_start,steps=spec['training_steps'],checkpoint_sha256=sha(checkpoint)))
   del opt
  evaluate(model,prior,method,cfg,spec,ph,evaluation,sha(checkpoint),manifest)
  completed.append(method);write(a.out/'progress.json',dict(complete=completed==spec['methods'],completed=completed,protocol_sha256=ph,teacher_queries=queries))
  del model;gc.collect();torch.cuda.empty_cache()


 quality=a.out/'physical_eval';quality.mkdir();quality_rows=[];quality_queries=0
 for i in spec['conditions']:
  source=torch.load(evaluation/f'frozen_c{i}.pt',map_location='cpu',weights_only=False);c=source['condition']
  with EnergyOracle(spec['oracle_interpreter'],worker,spec['oracle_checkpoint'],numbers=c['numbers'],charge=c['charge'],spin_multiplicity=c['spin_multiplicity'],device='cuda',batch_size=16) as oracle:
   assert oracle.handshake['base_precision_dtype']=='torch.float32' and not oracle.handshake['tf32']
   for method in spec['methods']:
    path=evaluation/f'{method}_c{i}.pt';sample=torch.load(path,map_location='cpu',weights_only=False);assert sample['condition']==c;x=sample['positions'];n=len(x)
    e,f=oracle.evaluate_chunked(torch.cat([x,-x]),max_request=32);even=(e[:n]+e[n:])/2;force=(f[:n]-f[n:])/2
    file=quality/f'{method}_c{i}.pt';torch.save(dict(source_sample_sha256=sha(path),positions=x,raw_energy_eV=e,raw_force_eV_A=f,even_energy_eV=even,even_force_eV_A=force),file)
    quality_rows.append(dict(method=method,condition_index=i,artifact=file.name,artifact_sha256=sha(file),source_sample_sha256=sha(path)))
   assert oracle.evaluated==oracle.requested_evaluations
   quality_queries+=oracle.evaluated
  write(quality/'results.json',dict(complete=False,rows=quality_rows,raw_queries=quality_queries,protocol_sha256=ph))
 assert quality_queries==2*len(spec['conditions'])*len(spec['methods'])*spec['samples_per_condition']
 write(quality/'results.json',dict(complete=True,rows=quality_rows,raw_queries=quality_queries,protocol_sha256=ph))
 write(a.out/'complete.json',dict(complete=True,protocol_sha256=ph,methods=completed,teacher_raw_queries=queries,evaluation_raw_queries=quality_queries,total_raw_queries=queries+quality_queries,seconds=time.perf_counter()-start,scientific_submission_ready=False))


if __name__=='__main__':main()
