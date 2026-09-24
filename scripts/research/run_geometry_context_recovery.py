"""Learn a small geometry context while preserving the parent and force head."""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from cfm_mol import matched_egnn as base, connectivity_feedback as feedback
from cfm_mol.geometry_recovery import recovery_loss
from cfm_mol.geometry_moment_context import GeometryMomentContext
from scripts.research.run_geometry_recovery import evaluate
from scripts.research.run_matched_connection import load_parent
from scripts.research.run_matched_generators import batches,write
from scripts.research.run_gaga_feedback import atomic_save
from scripts.research.train_electronic_fm import sha


def main():
    p=argparse.ArgumentParser()
    for key in ['project','protocol','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--fit',type=int,choices=[0,1],required=True)
    p.add_argument('--variant',choices=['radial_context','moment_context'],required=True)
    a=p.parse_args();root=a.project.resolve();out=a.out.resolve();protocol=json.loads(a.protocol.read_text());ph=sha(a.protocol)
    assert protocol['frozen'];torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out.mkdir(parents=True,exist_ok=True)
    datafile=root/protocol['data'];assert sha(datafile)==protocol['data_sha256']
    data=torch.load(datafile,map_location='cpu',weights_only=False)['training']
    assert not {r['condition']['composition_hex'] for r in data}&{r['condition']['composition_hex'] for r in protocol['conditions']}
    arm=protocol['parents'][a.fit];model=load_parent(root,arm).eval().requires_grad_(False)
    original_hash=base.state_hash(model)
    original={name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
    source=base.HarmonicSource(arm['spec']['edge_log_width'])
    torch.manual_seed(protocol['context_seeds'][a.fit])
    config=dict(protocol['context_configuration'],atomic_numbers=arm['spec']['atomic_numbers'],
        mode='radial' if a.variant=='radial_context' else 'moments')
    encoder=GeometryMomentContext(**config).cuda().float()
    model.add_module('geometry_context_encoder',encoder)
    initial=base.state_hash(model);ema=copy.deepcopy(encoder).eval().requires_grad_(False)
    trainable=[name for name,param in model.named_parameters() if param.requires_grad]
    assert trainable and all(name.startswith('geometry_context_encoder.') for name in trainable)
    assert sum(param.numel() for param in encoder.parameters())==7489
    optimizer=torch.optim.AdamW(encoder.parameters(),lr=protocol['learning_rate'],amsgrad=True,weight_decay=1e-12)
    schedule=batches(data,protocol['steps'],protocol['batch_size'],protocol['batch_seeds'][a.fit])
    indices_file=out/'batch_indices.npy'
    if indices_file.exists():np.testing.assert_array_equal(np.load(indices_file),schedule)
    else:np.save(indices_file,schedule)
    initialization=dict(protocol_sha256=ph,initial_state_sha256=initial,original_parent_state_sha256=original_hash,
        batch_schedule_sha256=sha(indices_file),fit=a.fit,variant=a.variant,parameters=sum(param.numel() for param in model.parameters()),
        trainable_parameters=7489,context_configuration=config,device=torch.cuda.get_device_name())
    if (out/'initialization.json').exists():
        old=json.loads((out/'initialization.json').read_text())
        for key in ['protocol_sha256','initial_state_sha256','batch_schedule_sha256']:assert old[key]==initialization[key]
    else:write(out/'initialization.json',initialization)
    start=0;seconds=0.
    if (out/'last.ckpt').exists():
        saved=torch.load(out/'last.ckpt',map_location='cuda',weights_only=False)
        assert saved['protocol_sha256']==ph and saved['initial_state_sha256']==initial
        encoder.load_state_dict(saved['context_state_dict']);ema.load_state_dict(saved['context_ema_state_dict'])
        optimizer.load_state_dict(saved['optimizer_state_dict']);start=saved['global_step'];seconds=saved['training_seconds'];del saved
    attempt=len(list(out.glob('metrics_attempt*.jsonl')))
    with (out/f'metrics_attempt{attempt}.jsonl').open('w') as stream:
        for step,indices in enumerate(schedule[start:],start+1):
            tick=time.perf_counter();rows=[data[int(i)] for i in indices]
            clean=torch.stack([row['positions'] for row in rows]).cuda().float()
            numbers=torch.tensor([row['condition']['numbers'] for row in rows],device='cuda')
            seed=protocol['noise_seeds'][a.fit]*1000003+step
            if step%2:
                objective=feedback.loss(model,clean,numbers,arm['spec'],source,encoder,seed);extra=dict(kind='original_cfm')
            else:
                objective,extra=recovery_loss(model,clean,numbers,arm['spec'],source,encoder,seed,corrupt=True,local_geometry=True)
                extra['kind']='endpoint_recovery'
            if not torch.isfinite(objective):raise FloatingPointError('Nonfinite context recovery objective')
            optimizer.zero_grad(set_to_none=True);objective.backward()
            norm=torch.nn.utils.clip_grad_norm_(encoder.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            assert all(param.grad is None for name,param in model.named_parameters() if not name.startswith('geometry_context_encoder.'))
            decay=min(protocol['ema_decay'],(1+step)/(10+step))
            with torch.no_grad():
                for average,current in zip(ema.parameters(),encoder.parameters()):average.lerp_(current,1-decay)
            torch.cuda.synchronize();seconds+=time.perf_counter()-tick
            if step<=4 or step%100==0:
                record=dict(step=step,loss=float(objective.detach()),gradient_norm=float(norm),training_seconds=seconds,**extra)
                stream.write(json.dumps(record)+'\n');stream.flush();print(json.dumps(record),flush=True)
            if step%protocol['checkpoint_every']==0 or step==protocol['steps']:
                atomic_save(dict(context_state_dict=encoder.state_dict(),context_ema_state_dict=ema.state_dict(),
                    context_configuration=config,optimizer_state_dict=optimizer.state_dict(),global_step=step,
                    protocol_sha256=ph,initial_state_sha256=initial,training_seconds=seconds),out/'last.ckpt')
    for name,value in original.items():torch.testing.assert_close(model.state_dict()[name].cpu(),value,atol=0,rtol=0)
    encoder.load_state_dict(ema.state_dict());encoder.eval().requires_grad_(False)
    done=dict(complete=True,protocol_sha256=ph,fit=a.fit,variant=a.variant,steps=protocol['steps'],
        original_parent_state_sha256=original_hash,initial_state_sha256=initial,ema_state_sha256=base.state_hash(model),
        checkpoint_sha256=sha(out/'last.ckpt'),training_seconds=seconds,trainable_parameters=7489,
        frozen_parent_tensors_identical=True,new_optimizer_steps=protocol['steps'],
        training_forward_examples=protocol['steps']*protocol['batch_size']*2,new_esen_queries=0)
    write(out/'training.json',done)
    evaluate(root,protocol,ph,a.fit,a.variant,model,source,encoder,out/'evaluation')
    for name,value in original.items():torch.testing.assert_close(model.state_dict()[name].cpu(),value,atol=0,rtol=0)
    write(out/'complete.json',dict(**done,evaluation_complete_sha256=sha(out/'evaluation/complete.json')))


if __name__=='__main__':main()
