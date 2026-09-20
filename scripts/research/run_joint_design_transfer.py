"""Train structured diffusion, attach a frozen FM head, and evaluate both designs."""
import argparse,copy,gc,json,time
from pathlib import Path
import numpy as np
import torch
from cfm_mol import harmonic_diffusion as hd,matched_egnn as base,connectivity_feedback as feedback
from cfm_mol.matched_physical_connection import PhysicalFieldTransform
from scripts.research.run_matched_generators import batches,write
from scripts.research.run_gaga_feedback import atomic_save
from scripts.research.run_atomwise_connection import restore_head
from scripts.research.run_matched_connection import score
from scripts.research.audit_generator_output_support import assess
from scripts.research.audit_seed_replication import verify_physics
from scripts.research.train_electronic_fm import sha


@torch.no_grad()
def validation(model,spec,noise,rows,seed):
    values=[];model.eval()
    for i,row in enumerate(rows):
        clean=row['positions'][None].float().cuda();z=torch.tensor([row['condition']['numbers']],device='cuda')
        values.append(float(hd.loss(model,clean,z,spec,noise,seed+i)))
    return float(np.mean(values))


def train(root,protocol,ph,target,fit,out,data):
    arm=protocol['parents'][fit][target];spec=arm['spec'];out.mkdir(parents=True,exist_ok=True)
    assert sha(root/arm['baseline_checkpoint'])==arm['baseline_checkpoint_sha256']
    original=torch.load(root/arm['baseline_checkpoint'],map_location='cpu',weights_only=False)
    model=feedback.install(base.initialize(spec,'cuda'));initial=base.state_hash(model)
    assert initial==original['initial_state_sha256'] and sum(p.numel() for p in model.parameters())==2381566
    del original
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['learning_rate'],amsgrad=True,weight_decay=1e-12)
    schedule=batches(data['training'],protocol['training_steps'],protocol['batch_size'],spec['batch_seed'])
    assert sha(root/arm['baseline_batch_indices'])==arm['baseline_batch_indices_sha256']
    np.testing.assert_array_equal(schedule,np.load(root/arm['baseline_batch_indices']))
    if (out/'batch_indices.npy').exists():np.testing.assert_array_equal(schedule,np.load(out/'batch_indices.npy'))
    else:np.save(out/'batch_indices.npy',schedule)
    noise=hd.HarmonicNoise(protocol['source_width']);start=0;seconds=0.;checkpoint=out/'last.ckpt'
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cuda',weights_only=False)
        assert saved['protocol_sha256']==ph and saved['initial_state_sha256']==initial
        model.load_state_dict(saved['state_dict'],strict=True);ema.load_state_dict(saved['ema_state_dict'],strict=True)
        optimizer.load_state_dict(saved['optimizer_state_dict']);start=saved['global_step'];seconds=saved['training_seconds'];del saved
    selected=data['validation'][:protocol['validation_examples']]
    if not (out/'validation_initial.json').exists():
        write(out/'validation_initial.json',dict(step=0,loss=validation(ema,spec,noise,selected,82001),model_selection=False))
    attempt=len(list(out.glob('metrics_attempt*.jsonl')));log=out/f'metrics_attempt{attempt}.jsonl'
    write(out/f'attempt_{attempt}.json',dict(start_step=start,protocol_sha256=ph))
    model.train()
    with log.open('w') as f:
        for step,index in enumerate(schedule[start:],start+1):
            tick=time.perf_counter();rows=[data['training'][int(i)] for i in index]
            clean=torch.stack([r['positions'] for r in rows]).float().cuda()
            z=torch.tensor([r['condition']['numbers'] for r in rows],device='cuda')
            objective=hd.loss(model,clean,z,spec,noise,spec['noise_seed']*1000003+step)
            if not torch.isfinite(objective):raise FloatingPointError('Nonfinite harmonic diffusion loss')
            optimizer.zero_grad(set_to_none=True);objective.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            decay=min(protocol['ema_decay'],(1+step)/(10+step))
            with torch.no_grad():
                for average,value in zip(ema.parameters(),model.parameters()):average.lerp_(value,1-decay)
            torch.cuda.synchronize();seconds+=time.perf_counter()-tick
            record=dict(step=step,loss=float(objective.detach()),gradient_norm=float(norm),training_seconds=seconds)
            f.write(json.dumps(record)+'\n')
            if step==1 or step%250==0:f.flush();print(json.dumps(dict(phase='training',target=target,fit=fit,**record)),flush=True)
            if step%protocol['checkpoint_every']==0 or step==protocol['training_steps']:
                atomic_save(dict(protocol_sha256=ph,network_spec=spec,state_dict=model.state_dict(),ema_state_dict=ema.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),initial_state_sha256=initial,global_step=step,training_seconds=seconds),checkpoint)
            if step in protocol['validation_steps']:
                value=validation(ema,spec,noise,selected,82001);write(out/f'validation_{step}.json',dict(step=step,loss=value,model_selection=False));model.train()
    assert (start==protocol['training_steps']) or step==protocol['training_steps']
    write(out/'complete.json',dict(complete=True,protocol_sha256=ph,target=target,fit=fit,checkpoint_sha256=sha(checkpoint),
        initial_state_sha256=initial,ema_state_sha256=base.state_hash(ema),steps=protocol['training_steps'],
        training_example_forwards=protocol['training_steps']*protocol['batch_size'],validation_example_forwards=48,
        training_seconds=seconds,batch_indices_sha256=sha(out/'batch_indices.npy')))
    del model,optimizer;gc.collect();torch.cuda.empty_cache()
    return ema.eval().requires_grad_(False),noise


def generate(root,p,ph,target,fit,out,model,noise):
    spec=p['parents'][fit][target]['spec'];info=p['heads'][fit];head,_=restore_head(root,info);head.requires_grad_(False)
    mh,hh=base.state_hash(model),base.state_hash(head);assert hh==info['head_state_sha256']
    folder=out/'generation';folder.mkdir(parents=True,exist_ok=True);report=[];coordinates={};graphs={}
    c=p['test_rows'][0];seed=p['evaluation_seeds'][fit]*1000003
    direct=hd.sample(model,c['numbers'],spec,noise,seed,8,128)
    zero=PhysicalFieldTransform(model,spec,head,0.,strength_limit=4.)
    checked=hd.sample(model,c['numbers'],spec,noise,seed,8,128,field_transform=zero)
    for x,y in zip(direct[:2],checked[:2]):torch.testing.assert_close(x,y,atol=0,rtol=0)
    for key in direct[2]:torch.testing.assert_close(direct[2][key],checked[2][key],atol=0,rtol=0)
    write(out/'sampler_verification.json',dict(complete=True,protocol_sha256=ph,verification_outputs=16,native_equals_zero=True))
    for i,c in enumerate(p['test_rows']):
        file=folder/f'both_c{i}.pt'
        if file.exists():
            saved=torch.load(file,map_location='cpu',weights_only=False)
            assert saved['protocol_sha256']==ph and saved['model_state_sha256']==mh and saved['head_state_sha256']==hh and saved['condition']==c
        else:
            transform=PhysicalFieldTransform(model,spec,head,p['strength'],strength_limit=4.)
            xx=[];zz=[];components=[];calls=[0];tick=time.perf_counter()
            handle=model.dynamics.egnn.register_forward_hook(lambda *_:calls.__setitem__(0,calls[0]+1))
            for begin in range(0,16,8):
                seed=p['evaluation_seeds'][fit]*1000003+i*100003+begin
                x,x0,component=hd.sample(model,c['numbers'],spec,noise,seed,8,128,field_transform=transform)
                xx.append(x.cpu().double());zz.append(x0.cpu().double());components.append(component)
            handle.remove();torch.cuda.synchronize();assert calls[0]==256 and transform.calls==256
            saved=dict(positions=torch.cat(xx),initial_positions=torch.cat(zz),components=components,condition=c,
                strength=p['strength'],protocol_sha256=ph,model_state_sha256=mh,head_state_sha256=hh,
                core_calls=256,head_calls=256,generation_seconds=time.perf_counter()-tick)
            atomic_save(saved,file)
        quality=assess(saved['positions'],c,list(range(16)));coordinates[i]=saved['positions'];graphs[i]=quality
        report.append(dict(method='both',strength=p['strength'],condition_index=i,file=file.name,sha256=sha(file),**quality))
        print(json.dumps(dict(phase='generation',target=target,fit=fit,condition=i,graph=quality['graph_supported'],attempted=16)),flush=True)
    assert base.state_hash(model)==mh and base.state_hash(head)==hh
    value=dict(complete=True,protocol_sha256=ph,rows=report,new_neural_outputs=1024,model_state_sha256=mh,head_state_sha256=hh)
    write(folder/'generation.json',value);del head;gc.collect();torch.cuda.empty_cache()
    if not (out/'xtb/results.json').exists():score(p,ph,fit,[(folder,value)],out/'xtb')
    physical=json.loads((out/'xtb/results.json').read_text());assert physical['complete'] and physical['protocol_sha256']==ph
    arrays=dict(graph=np.zeros((64,16),bool),success=np.zeros((64,16),bool),force=np.full((64,16),np.inf),energy=np.full((64,16),np.nan))
    cache=set();seen=set()
    for r in physical['rows']:
        i,j=r['condition_index'],r['sample_index'];assert r['method']=='both' and (i,j) not in seen;seen.add((i,j))
        ok,energy,force=verify_physics(root,r,out/'xtb/details'/r['task_id'],coordinates[i][j],p['test_rows'][i],cache)
        graph=graphs[i]['records'][j]['graph_supported'];assert r['joint']==bool(graph and ok and force<=5)
        for k,v in [('graph',graph),('success',ok),('force',force),('energy',energy)]:arrays[k][i,j]=v
    assert len(seen)==1024
    np.savez_compressed(out/'audit.npz',**arrays)
    write(out/'complete.json',dict(complete=True,protocol_sha256=ph,target=target,fit=fit,
        arrays_sha256=sha(out/'audit.npz'),generation_sha256=sha(folder/'generation.json'),physical_sha256=sha(out/'xtb/results.json'),
        training_sha256=sha(out/'training/complete.json'),model_state_sha256=mh,head_state_sha256=hh,
        new_evaluation_outputs=1024,new_gfn2_attempts=1024,new_parent_updates=30000,new_head_updates=0,new_esen_queries=0,
        verification_replays=16,summary=physical['summary']))


def main():
    q=argparse.ArgumentParser(description=__doc__)
    for k in ['project','protocol','out']:q.add_argument('--'+k,type=Path,required=True)
    q.add_argument('--target',choices=['edm','gaga'],required=True);q.add_argument('--fit',type=int,choices=[0,1],required=True);a=q.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    p=json.loads(a.protocol.read_text());ph=sha(a.protocol);assert p['frozen'];a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'complete.json').exists():assert json.loads((a.out/'complete.json').read_text())['protocol_sha256']==ph;return
    spec=p['parents'][a.fit][a.target]['spec'];assert sha(a.project/spec['data'])==spec['data_sha256']
    data=torch.load(a.project/spec['data'],map_location='cpu',weights_only=False)
    keys={c['composition_hex'] for c in p['test_rows']};assert len(keys)==64
    assert not keys&{r['condition']['composition_hex'] for r in data['training']+data['validation']}
    model,noise=train(a.project,p,ph,a.target,a.fit,a.out/'training',data)
    generate(a.project,p,ph,a.target,a.fit,a.out,model,noise)


if __name__=='__main__':main()
