"""Repeat the frozen parent recipe with new seeds and no outcome-based selection."""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
import torch
from cfm_mol import matched_egnn as base,connectivity_feedback as feedback
from scripts.research.run_matched_generators import batches,write
from scripts.research.run_gaga_feedback import atomic_save
from scripts.research.train_electronic_fm import sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','protocol','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--fit',type=int,choices=[2,3,4],required=True);p.add_argument('--family',choices=['fm','gaga'],required=True);a=p.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    campaign=json.loads(a.protocol.read_text());assert campaign['frozen'];entry=campaign['fits'][a.fit]['parents'][a.family];spec=entry['spec'];file=a.project/entry['protocol'];ph=sha(file)
    assert ph==entry['protocol_sha256'] and sha(a.project/spec['data'])==spec['data_sha256'];data=torch.load(a.project/spec['data'],map_location='cpu',weights_only=False)
    assert len(data['training'])==20000 and not {r['condition']['composition_hex'] for r in data['training']}&{r['condition']['composition_hex'] for r in data['validation']}
    a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'complete.json').exists():
        done=json.loads((a.out/'complete.json').read_text());assert done['protocol_sha256']==ph and done['checkpoint_sha256']==sha(a.out/'last.ckpt');return
    model=base.initialize(spec,'cuda');initial=base.state_hash(model)
    if a.family=='fm':feedback.install(model)
    assert base.state_hash(model)==initial and sum(q.numel() for q in model.parameters())==2381566
    source=base.HarmonicSource(spec['edge_log_width']);context=feedback.GeometryContext(source,'distance') if a.family=='fm' else None
    model.train();ema=copy.deepcopy(model).eval().requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=spec['learning_rate'],amsgrad=True,weight_decay=1e-12)
    full=batches(data['training'],30000,spec['batch_size'],spec['batch_seed']);schedule=full[:spec['training_steps']]
    if (a.out/'batch_indices.npy').exists():np.testing.assert_array_equal(np.load(a.out/'batch_indices.npy'),schedule)
    else:np.save(a.out/'batch_indices.npy',schedule)
    init=dict(protocol_sha256=ph,state_sha256=initial,parameter_count=2381566,batch_schedule_sha256=sha(a.out/'batch_indices.npy'),batch_prefix_steps=15000,pretrained=False)
    if (a.out/'initialization.json').exists():assert json.loads((a.out/'initialization.json').read_text())==init
    else:write(a.out/'initialization.json',init)
    start=0;seconds=0.
    if (a.out/'last.ckpt').exists():
        saved=torch.load(a.out/'last.ckpt',map_location='cuda',weights_only=False);assert saved['protocol_sha256']==ph and saved['initial_state_sha256']==initial
        model.load_state_dict(saved['state_dict'],strict=True);ema.load_state_dict(saved['ema_state_dict'],strict=True);optimizer.load_state_dict(saved['optimizer_state_dict']);start=saved['global_step'];seconds=saved['training_seconds'];del saved
    number=len(list(a.out.glob('metrics_attempt*.jsonl')));write(a.out/f'attempt{number}.json',dict(start_step=start,device=torch.cuda.get_device_name(),discarded_work_preserved=True))
    with (a.out/f'metrics_attempt{number}.jsonl').open('w') as stream:
        for step,indices in enumerate(schedule[start:],start+1):
            tick=time.perf_counter();rows=[data['training'][int(i)] for i in indices];clean=torch.stack([r['positions'] for r in rows]).cuda().float();z=torch.tensor([r['condition']['numbers'] for r in rows],device='cuda');seed=spec['noise_seed']*1000003+step
            objective=feedback.loss(model,clean,z,spec,source,context,seed) if context else base.loss(model,clean,z,spec['kind'],spec,source,seed)
            if not torch.isfinite(objective):raise FloatingPointError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True);objective.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            decay=min(spec['ema_decay'],(1+step)/(10+step))
            with torch.no_grad():
                for average,current in zip(ema.parameters(),model.parameters()):average.lerp_(current,1-decay)
            torch.cuda.synchronize();seconds+=time.perf_counter()-tick
            if step==1 or step%250==0:
                record=dict(step=step,loss=float(objective.detach()),gradient_norm=float(norm),training_seconds=seconds);stream.write(json.dumps(record)+'\n');stream.flush();print(json.dumps(record),flush=True)
            if step%spec['checkpoint_every']==0 or step==spec['training_steps']:
                atomic_save(dict(state_dict=model.state_dict(),ema_state_dict=ema.state_dict(),optimizer_state_dict=optimizer.state_dict(),global_step=step,protocol=spec,protocol_sha256=ph,initial_state_sha256=initial,training_seconds=seconds),a.out/'last.ckpt')
    write(a.out/'complete.json',dict(complete=True,protocol_sha256=ph,campaign_sha256=sha(a.protocol),fit=a.fit,family=a.family,steps=spec['training_steps'],initial_state_sha256=initial,ema_state_sha256=base.state_hash(ema),checkpoint_sha256=sha(a.out/'last.ckpt'),training_seconds=seconds,training_forward_examples=960000,optimizer_steps=spec['training_steps'],new_physical_queries=0,new_generation_outputs=0,selection='Final fixed-step EMA; no validation-based model selection.'))

if __name__=='__main__':main()
