"""Train a small H-coordinate flow shared by both compared parent generators."""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
import torch
from cfm_mol import matched_egnn as base
from cfm_mol.hydrogen_completion import loss
from scripts.research.run_matched_generators import batches,write
from scripts.research.run_gaga_feedback import atomic_save
from scripts.research.train_electronic_fm import sha


@torch.no_grad()
def diagnostics(model,spec,rows):
    values=[]
    for i,row in enumerate(rows):
        clean=row['positions'][None].cuda().float();z=torch.tensor(row['condition']['numbers'],device='cuda')[None]
        value=loss(model,clean,z,spec,seed=spec['diagnostic_seed']*1000003+i);values.append(float(value))
    return dict(mean_hydrogen_velocity_mse=float(np.mean(values)),per_row=values)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','protocol','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--seed-index',type=int,choices=[0,1],required=True);a=p.parse_args();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    spec=json.loads(a.protocol.read_text());ph=sha(a.protocol);si=a.seed_index;assert spec['frozen'] and sha(a.project/spec['data'])==spec['data_sha256']
    data=torch.load(a.project/spec['data'],map_location='cpu',weights_only=False);training=data['training'];validation=[data['validation'][i] for i in spec['reference_validation_rows']]
    assert len(training)==20000 and not {r['condition']['composition_hex'] for r in training}&{r['condition']['composition_hex'] for r in validation}
    a.out.mkdir(parents=True,exist_ok=False);net=copy.deepcopy(spec['network_spec']);net['initialization_seed']=spec['seeds'][si]
    model=base.initialize(net,'cuda').train();assert sum(p.numel() for p in model.parameters())==spec['parameter_count']
    initial=base.state_hash(model);frozen={n:p.detach().cpu().clone() for n,p in model.named_parameters() if not p.requires_grad}
    ema=copy.deepcopy(model).eval().requires_grad_(False);optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=spec['learning_rate'],amsgrad=True,weight_decay=1e-12)
    schedule=batches(training,spec['steps'],spec['batch_size'],spec['batch_seeds'][si]);np.save(a.out/'batch_indices.npy',schedule)
    write(a.out/'initialization.json',dict(protocol_sha256=ph,initial_state_sha256=initial,parameter_count=spec['parameter_count'],schedule_sha256=sha(a.out/'batch_indices.npy'),frozen_parameters=list(frozen)))
    history=[];tick=time.perf_counter()
    with (a.out/'metrics.jsonl').open('w') as stream:
        for step,indices in enumerate(schedule,1):
            rows=[training[int(i)] for i in indices];clean=torch.stack([r['positions'] for r in rows]).cuda().float();z=torch.tensor([r['condition']['numbers'] for r in rows],device='cuda')
            value=loss(model,clean,z,spec,seed=spec['noise_seeds'][si]*1000003+step)
            if not torch.isfinite(value):raise FloatingPointError('Nonfinite conditional H objective')
            optimizer.zero_grad(set_to_none=True);value.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            decay=min(spec['ema_decay'],(1+step)/(10+step))
            with torch.no_grad():
                for average,current in zip(ema.parameters(),model.parameters()):average.lerp_(current,1-decay)
            if step%100==0 or step==1:
                row=dict(step=step,loss=float(value.detach()),gradient_norm=float(norm),seconds=time.perf_counter()-tick)
                stream.write(json.dumps(row)+'\n');stream.flush();print(json.dumps(row),flush=True)
            if step in spec['diagnostic_steps']:
                record=dict(step=step,validation=diagnostics(ema,spec,validation));history.append(record)
                print(json.dumps(dict(phase='reference_diagnostic',step=step,mse=record['validation']['mean_hydrogen_velocity_mse'])),flush=True)
            if step%spec['checkpoint_every']==0:
                file=a.out/f'step_{step}.pt';atomic_save(dict(state_dict=model.state_dict(),ema_state_dict=ema.state_dict(),optimizer_state_dict=optimizer.state_dict(),
                    network_spec=net,protocol_sha256=ph,step=step,seed=spec['seeds'][si],initial_state_sha256=initial),file)
    for n,p in model.named_parameters():
        if n in frozen:torch.testing.assert_close(p.cpu(),frozen[n],atol=0,rtol=0)
    write(a.out/'complete.json',dict(complete=True,protocol_sha256=ph,seed_index=si,steps=spec['steps'],parameter_count=spec['parameter_count'],
        checkpoint_sha256=sha(a.out/f'step_{spec["steps"]}.pt'),history=history,new_training_forward_examples=spec['steps']*spec['batch_size'],
        diagnostic_forward_examples=len(validation)*len(history),new_neural_generation_outputs=0,new_physical_queries=0,seconds=time.perf_counter()-tick))


if __name__=='__main__':main()
