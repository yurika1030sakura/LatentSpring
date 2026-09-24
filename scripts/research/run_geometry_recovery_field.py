"""Train and sample a direct geometry field with the original physical correction."""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from cfm_mol import matched_egnn as base
from cfm_mol.geometry_recovery_field import GeometryRecoveryField,GeometryThenPhysical,denoising_targets
from cfm_mol.matched_physical_connection import PhysicalFieldTransform
from scripts.research.run_matched_connection import load_parent,make_context,sample,score
from scripts.research.run_atomwise_connection import restore_head
from scripts.research.run_geometry_recovery import review_geometry
from scripts.research.audit_generator_output_support import assess
from scripts.research.run_matched_generators import batches,write
from scripts.research.run_gaga_feedback import atomic_save
from scripts.research.train_electronic_fm import sha


def evaluate(root,spec,ph,fit,variant,geometry,out):
    out.mkdir(parents=True,exist_ok=True);arm=spec['parents'][fit]
    model=load_parent(root,arm);source=base.HarmonicSource();context=make_context(arm,source)
    head,_=restore_head(root,spec['physical_heads'][fit]);parent_hash=base.state_hash(model);geo_hash=base.state_hash(geometry)
    if (out/'complete.json').exists():
        done=json.loads((out/'complete.json').read_text());assert done['protocol_sha256']==ph and done['geometry_head_state_sha256']==geo_hash;return
    folder=out/'generation';folder.mkdir(exist_ok=False);rows=[]
    for ci,entry in enumerate(spec['conditions']):
        c=entry['condition'];physical=PhysicalFieldTransform(model,arm['spec'],head,4.,strength_limit=4.)
        transform=GeometryThenPhysical(physical,geometry);positions=[];starts=[];core=[0];geo_before=geometry.forward_calls
        hook=model.dynamics.egnn.register_forward_hook(lambda *_:core.__setitem__(0,core[0]+1))
        for begin in [0,8]:
            seed=spec['evaluation_seeds'][fit]*1000003+ci*100003+begin
            x,x0=sample(model,arm,source,context,c,seed,8,128,transform)
            positions.append(x.cpu().double());starts.append(x0.cpu().double())
        hook.remove();x=torch.cat(positions)
        assert core[0]==256 and transform.calls==physical.calls==geometry.forward_calls-geo_before==128
        file=folder/f'{variant}_c{ci}.pt'
        atomic_save(dict(positions=x,initial_positions=torch.cat(starts),condition=c,protocol_sha256=ph,
            model_state_sha256=parent_hash,head_state_sha256=base.state_hash(head),geometry_head_state_sha256=geo_hash,
            backbone_calls_per_trajectory=128,geometry_calls_per_trajectory=64,physical_calls_per_trajectory=64),file)
        quality=assess(x,c,list(range(16)));rows.append(dict(method=variant,condition_index=ci,file=file.name,sha256=sha(file),**quality))
        print(json.dumps(dict(phase='generation',fit=fit,variant=variant,condition=ci,graph=quality['graph_supported'],attempted=16)),flush=True)
    assert base.state_hash(model)==parent_hash and base.state_hash(geometry)==geo_hash
    report=dict(complete=True,protocol_sha256=ph,rows=rows,new_neural_outputs=384,model_state_sha256=parent_hash,
        head_state_sha256=base.state_hash(head),geometry_head_state_sha256=geo_hash,new_oracle_queries=0)
    write(folder/'generation.json',report);checks=review_geometry(folder,report,out/'geometry.json')
    scoring=json.loads((root/spec['parent_protocol']).read_text());scoring.update(strength_limit=4.)
    score(scoring,ph,fit,[(folder,report)],out/'xtb')
    physical=json.loads((out/'xtb/results.json').read_text());lookup={(r['condition_index'],r['sample_index']):r for r in physical['rows']}
    joint=sum(r['closed_shell_geometry_pass'] and lookup[r['condition'],r['sample']]['success']
        and lookup[r['condition'],r['sample']]['rms_force']<=5 for r in checks)
    done=dict(complete=True,protocol_sha256=ph,fit=fit,variant=variant,attempted=len(checks),graph=sum(r['graph'] for r in checks),
        geometry=sum(r['closed_shell_geometry_pass'] for r in checks),geometry_force=joint,
        heavy_disconnected=sum(r['heavy_components']>1 for r in checks),model_state_sha256=parent_hash,
        head_state_sha256=report['head_state_sha256'],geometry_head_state_sha256=geo_hash,
        generation_sha256=sha(folder/'generation.json'),geometry_sha256=sha(out/'geometry.json'),physical_sha256=sha(out/'xtb/results.json'),
        new_generation_outputs=384,new_gfn2_attempts=384,new_esen_queries=0,geometry_optimized=False)
    write(out/'complete.json',done);print(json.dumps(dict(phase='evaluated',**done)),flush=True)


def main():
    p=argparse.ArgumentParser()
    for key in ['project','protocol','references','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--fit',type=int,choices=[0,1],required=True)
    p.add_argument('--variant',choices=['radial_geometry','moment_geometry'],required=True)
    a=p.parse_args();root=a.project.resolve();out=a.out.resolve();spec=json.loads(a.protocol.read_text());ph=sha(a.protocol)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    references=json.loads(a.references.read_text());assert references['complete'] and references['protocol_sha256']==ph
    datafile=root/spec['data'];assert sha(datafile)==spec['data_sha256']
    data=torch.load(datafile,map_location='cpu',weights_only=False)['training'];selected=[data[i] for i in references['indices']]
    assert not {r['condition']['composition_hex'] for r in selected}&{r['condition']['composition_hex'] for r in spec['conditions']}
    torch.manual_seed(spec['context_seeds'][a.fit]);mode='radial' if a.variant=='radial_geometry' else 'moments'
    config=dict(spec['field_configuration'],atomic_numbers=spec['parents'][a.fit]['spec']['atomic_numbers'],mode=mode)
    model=GeometryRecoveryField(**config).cuda().float();assert sum(p.numel() for p in model.parameters())==7747
    initial=base.state_hash(model);ema=copy.deepcopy(model).eval().requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=spec['learning_rate'],weight_decay=1e-12)
    schedule=batches(selected,spec['steps'],spec['batch_size'],spec['batch_seeds'][a.fit]);out.mkdir(parents=True,exist_ok=True)
    global_schedule=np.asarray(references['indices'])[schedule]
    if (out/'batch_indices.npy').exists():np.testing.assert_array_equal(np.load(out/'batch_indices.npy'),global_schedule)
    else:np.save(out/'batch_indices.npy',global_schedule)
    init=dict(protocol_sha256=ph,initial_state_sha256=initial,batch_schedule_sha256=sha(out/'batch_indices.npy'),
        fit=a.fit,variant=a.variant,trainable_parameters=7747,reference_audit_sha256=sha(a.references),configuration=config)
    if (out/'initialization.json').exists():assert json.loads((out/'initialization.json').read_text())==init
    else:write(out/'initialization.json',init)
    start=0;seconds=0.
    if (out/'last.ckpt').exists():
        saved=torch.load(out/'last.ckpt',map_location='cuda',weights_only=False);assert saved['protocol_sha256']==ph
        model.load_state_dict(saved['state_dict']);ema.load_state_dict(saved['ema_state_dict']);optimizer.load_state_dict(saved['optimizer_state_dict']);start=saved['global_step'];seconds=saved['training_seconds']
    attempt=len(list(out.glob('metrics_attempt*.jsonl')))
    with (out/f'metrics_attempt{attempt}.jsonl').open('w') as stream:
        for step,indices in enumerate(global_schedule[start:],start+1):
            tick=time.perf_counter();rows=[data[int(i)] for i in indices]
            clean=base.center(torch.stack([r['positions'] for r in rows]).cuda().float());z=torch.tensor([r['condition']['numbers'] for r in rows],device='cuda')
            rng=torch.Generator(device='cuda').manual_seed(spec['noise_seeds'][a.fit]*1000003+step)
            progress=.4+.55*torch.rand((len(clean),),generator=rng,device='cuda')
            noisy,target,_=denoising_targets(clean,progress,rng,velocity_scale=config['velocity_scale'],gate_power=config['gate_power'])
            prediction=model(noisy,z,progress);weights=torch.where(z==1,.25,1.).to(clean)
            objective=(((prediction-target).square().sum(-1)*weights).sum(-1)/weights.sum(-1)).mean()
            if not torch.isfinite(objective):raise FloatingPointError('Nonfinite recovery-field loss')
            optimizer.zero_grad(set_to_none=True);objective.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            decay=min(spec['ema_decay'],(1+step)/(10+step))
            with torch.no_grad():
                for avg,param in zip(ema.parameters(),model.parameters()):avg.lerp_(param,1-decay)
            torch.cuda.synchronize();seconds+=time.perf_counter()-tick
            if step<=3 or step%500==0:
                row=dict(step=step,loss=float(objective.detach()),gradient_norm=float(norm),training_seconds=seconds)
                stream.write(json.dumps(row)+'\n');stream.flush();print(json.dumps(row),flush=True)
            if step%spec['checkpoint_every']==0 or step==spec['steps']:
                atomic_save(dict(state_dict=model.state_dict(),ema_state_dict=ema.state_dict(),configuration=config,
                    optimizer_state_dict=optimizer.state_dict(),global_step=step,protocol_sha256=ph,initial_state_sha256=initial,training_seconds=seconds),out/'last.ckpt')
    done=dict(complete=True,protocol_sha256=ph,fit=a.fit,variant=a.variant,steps=spec['steps'],initial_state_sha256=initial,
        ema_state_sha256=base.state_hash(ema),checkpoint_sha256=sha(out/'last.ckpt'),training_seconds=seconds,new_optimizer_steps=spec['steps'],
        new_backbone_training_example_forwards=0,geometry_head_training_example_forwards=spec['steps']*spec['batch_size'],new_esen_queries=0)
    write(out/'training.json',done);evaluate(root,spec,ph,a.fit,a.variant,ema,out/'evaluation')
    write(out/'complete.json',dict(**done,evaluation_complete_sha256=sha(out/'evaluation/complete.json')))


if __name__=='__main__':main()
