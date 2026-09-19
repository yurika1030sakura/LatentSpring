"""Fit only the correction head to cached frozen-parent states and physical targets."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from cfm_mol.physical_connection import PhysicalConnection
from scripts.research.train_electronic_fm import sha
from scripts.research.run_matched_generators import write
from scripts.research.run_gaga_feedback import atomic_save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','protocol','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--method',choices=['connection_force','connection_work'],required=True)
    a=p.parse_args();spec=json.loads(a.protocol.read_text());ph=sha(a.protocol);assert spec['frozen'] and a.seed in spec['training_seeds']
    bank=a.project/spec['trajectory_bank'];complete=json.loads((bank.parent/'complete.json').read_text())
    assert complete['complete'] and complete['protocol_sha256']==ph and complete['bank_sha256']==sha(bank)
    data=torch.load(bank,map_location='cpu',weights_only=False);assert data['protocol_sha256']==ph and len(data['rows'])==512
    parentpath=a.project/spec['checkpoint'];assert sha(parentpath)==spec['checkpoint_sha256'];parent=torch.load(parentpath,map_location='cpu',weights_only=False)
    torch.set_num_threads(2);torch.manual_seed(a.seed+spec['connection_init_seed_offset'])
    head=PhysicalConnection(**spec['physical_connection']).cuda().float().train();optimizer=torch.optim.AdamW(head.parameters(),lr=spec['feedback_learning_rate'],weight_decay=1e-12)
    atom_types={z:i for i,z in enumerate(spec['physical_connection']['atomic_numbers'])};cache=[]
    for row in data['rows']:
        n=len(row['x']);edges=torch.where(~torch.eye(n,dtype=torch.bool,device='cuda'))
        cache.append((row['x'].cuda(),row['endpoint'].cuda(),torch.tensor([atom_types[z] for z in row['condition']['numbers']],device='cuda'),
            row['t'].reshape(1).cuda(),torch.zeros(n,dtype=torch.long,device='cuda'),*edges,row[a.method.split('_')[-1]].cuda()))
    rng=np.random.default_rng(a.seed);a.out.mkdir(parents=True,exist_ok=False);tick=time.perf_counter()
    with (a.out/'metrics.jsonl').open('w') as stream:
        for step in range(1,spec['training_steps']+1):
            index=int(rng.integers(len(cache)));*inputs,target=cache[index];prediction=head(*inputs)
            loss=(prediction-target).square().sum(-1).mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite trajectory regression')
            optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            value=dict(step=step,bank_row=index,training_row=data['rows'][index]['training_row'],loss=float(loss.detach()),gradient_norm=float(norm))
            stream.write(json.dumps(value)+'\n')
            if step%100==0:stream.flush();print(json.dumps(dict(method=a.method,seed=a.seed,**value)),flush=True)
    state={**parent['state_dict'],**{'vector_field.physical_connection.'+n:v.detach().cpu() for n,v in head.state_dict().items()}}
    for name,value in parent['state_dict'].items():assert torch.equal(state[name],value)
    target=dict(mode='parent_trajectory_physical_moment',model_kT_eV=1.,teacher_temperature_K=300.,terminal_noise_std_A=0.,
        protocol_sha256=ph,bank_sha256=sha(bank),method=a.method,target=spec['target'],thermal_distribution_claim=False,adaptation='frozen_parent_connection')
    recipe={**parent['research_protocol'],'direct_endpoint_training':target,'physical_connection':spec['physical_connection']}
    checkpoint=a.out/'last.ckpt';atomic_save(dict(state_dict=state,source_prior=parent['source_prior'],research_protocol=recipe,
        global_step=spec['training_steps'],optimizer_state_dict=optimizer.state_dict(),training_seed=a.seed),checkpoint)
    # Training-set fit diagnostics only; no holdout or quality selection here.
    with torch.no_grad():
        before=[];after=[]
        for *inputs,target in cache:
            before.append(float(target.square().sum(-1).mean()));after.append(float((head(*inputs)-target).square().sum(-1).mean()))
    write(a.out/'complete.json',dict(complete=True,protocol_sha256=ph,bank_sha256=sha(bank),checkpoint_sha256=sha(checkpoint),
        seed=a.seed,method=a.method,steps=spec['training_steps'],backbone_training_forwards=0,connection_training_forwards=spec['training_steps'],
        diagnostic_head_forwards=len(cache),trainable_parameters=sum(p.numel() for p in head.parameters()),frozen_parent_verified=True,
        seconds=time.perf_counter()-tick,oracle_queries=0,training_mse_before=float(np.mean(before)),training_mse_after=float(np.mean(after))))


if __name__=='__main__':main()
