"""Generate compositions with the shared EGNN physical head and conditional H flow.

No graph labels, energy calls, or output selection are used by this entry point.
"""
import argparse,json,time
from pathlib import Path
import torch
from cfm_mol import matched_egnn as base
from cfm_mol.hydrogen_completion import complete
from cfm_mol.matched_physical_connection import PhysicalFieldTransform
from scripts.research.run_matched_connection import load_parent,make_context,sample
from scripts.research.run_atomwise_connection import restore_head
from scripts.research.train_electronic_fm import sha


def validate_condition(value,spec):
    numbers=value['numbers'];charge=value['charge'];spin=value['spin_multiplicity']
    if not isinstance(numbers,list) or not 2<=len(numbers)<=200 or any(type(z) is not int for z in numbers):raise ValueError('Provide 2--200 integer atomic numbers')
    if charge!=0 or spin!=1:raise ValueError('This checkpoint was evaluated only for neutral singlets')
    if 1 not in numbers or 6 not in numbers or not set(numbers)<=set(spec['network_spec']['atomic_numbers']):raise ValueError('Unsupported composition for this checkpoint')
    return dict(numbers=numbers,charge=charge,spin_multiplicity=spin)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--recipe',type=Path,default=Path('configs/research/latentspring_hydrogen_flow_v1.json'))
    group=p.add_mutually_exclusive_group(required=True);group.add_argument('--condition',type=Path);group.add_argument('--replay-parent',type=Path)
    p.add_argument('--family',choices=['fm','gaga'],default='fm');p.add_argument('--fit',type=int,choices=[0,1],default=0)
    p.add_argument('--samples',type=int,default=16);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();root=a.project.resolve();torch.set_num_threads(2)
    if a.out.exists():raise FileExistsError(a.out)
    if a.samples<1:raise ValueError('samples must be positive')
    recipe_path=root/a.recipe;recipe=json.loads(recipe_path.read_text());spec_file=root/recipe['protocol'];assert sha(spec_file)==recipe['protocol_sha256']
    spec=json.loads(spec_file.read_text());entry=recipe['decoder_checkpoints'][str(a.fit)];file=root/entry['file'];assert sha(file)==entry['sha256']
    saved=torch.load(file,map_location='cpu',weights_only=False);assert saved['protocol_sha256']==spec['training_protocol_sha256'] and saved['step']==10000
    decoder=base.initialize(saved['network_spec'],a.device).eval().requires_grad_(False);decoder.load_state_dict(saved['ema_state_dict'],strict=True)
    decoder_hash=base.state_hash(decoder);provenance=dict(recipe_sha256=sha(recipe_path),decoder_checkpoint_sha256=sha(file),decoder_state_sha256=decoder_hash,
        protocol_sha256=sha(spec_file),family=a.family,fit=a.fit)
    initial=None;parent_seconds=0.;backbone_calls=0;physical_calls=0
    if a.replay_parent:
        parent=torch.load(a.replay_parent,map_location='cpu',weights_only=False);c=validate_condition(parent['condition'],spec);x=parent['positions']
        if x.ndim!=3 or x.shape[1:]!=(len(c['numbers']),3) or not torch.isfinite(x).all():raise ValueError('Invalid parent coordinate batch')
        torch.testing.assert_close(x.float().double(),x.double(),atol=0,rtol=0)
        provenance.update(replay_parent_sha256=sha(a.replay_parent),replay_parent_file=str(a.replay_parent.resolve()))
    else:
        if a.device!='cuda':raise ValueError('Native parent sampling requires CUDA; CPU supports replay of saved parents')
        c=validate_condition(json.loads(a.condition.read_text()),spec);arm=spec['parents'][a.fit][a.family]
        model=load_parent(root,arm);source=base.HarmonicSource();context=make_context(arm,source)
        head,unused=restore_head(root,spec['physical_heads'][str(a.fit)][a.family]);transform=PhysicalFieldTransform(model,arm['spec'],head,4.,strength_limit=4.)
        provenance.update(parent_checkpoint_sha256=arm['checkpoint_sha256'],parent_state_sha256=base.state_hash(model),head_state_sha256=base.state_hash(head))
        outputs=[];starts=[];clock=time.perf_counter();counter=[0];hook=model.dynamics.egnn.register_forward_hook(lambda *_:counter.__setitem__(0,counter[0]+1))
        try:
            for begin in range(0,a.samples,spec['evaluation_batch']):
                n=min(spec['evaluation_batch'],a.samples-begin)
                final,x0=sample(model,arm,source,context,c,a.seed*1000003+begin,n,128,transform)
                outputs.append(final.cpu().double());starts.append(x0.cpu().double())
            torch.cuda.synchronize();parent_seconds=time.perf_counter()-clock
        finally:hook.remove()
        x=torch.cat(outputs);initial=torch.cat(starts);backbone_calls=counter[0];physical_calls=transform.calls
        batch_count=(a.samples+spec['evaluation_batch']-1)//spec['evaluation_batch']
        assert backbone_calls==128*batch_count and physical_calls==(64 if a.family=='fm' else 128)*batch_count
        assert base.state_hash(model)==provenance['parent_state_sha256'] and base.state_hash(head)==provenance['head_state_sha256']
        del model,head,context,transform
    results=[];infos=[];clock=time.perf_counter()
    for begin in range(0,len(x),spec['evaluation_batch']):
        xx=x[begin:begin+spec['evaluation_batch']].to(a.device).float();numbers=torch.tensor(c['numbers'],device=a.device)[None].expand(len(xx),-1)
        y,info=complete(decoder,xx,numbers,spec,mode='molecule',start_time=0.,steps=4,velocity_cap=2.)
        results.append(y.cpu().double());infos.append({k:v.cpu() if isinstance(v,torch.Tensor) else v for k,v in info.items()})
    if a.device=='cuda':torch.cuda.synchronize()
    decoder_seconds=time.perf_counter()-clock;y=torch.cat(results);changed=torch.cat([i['changed'] for i in infos])
    torch.testing.assert_close(y[~changed],x[~changed].double(),atol=0,rtol=0)
    heavy=torch.tensor(c['numbers'])!=1
    torch.testing.assert_close(y[:,heavy]-y[:,heavy].mean(1,keepdim=True),x[:,heavy]-x[:,heavy].mean(1,keepdim=True),atol=3e-6,rtol=1e-6)
    assert base.state_hash(decoder)==decoder_hash
    a.out.mkdir(parents=True)
    torch.save(dict(condition=c,positions=y,parent_positions=x,initial_positions=initial,changed=changed,
        active_atoms=torch.cat([i['active_atoms'] for i in infos]),provenance=provenance),a.out/'samples.pt')
    from rdkit import Chem
    periodic=Chem.GetPeriodicTable()
    with (a.out/'samples.xyz').open('w') as f:
        for index,pos in enumerate(y):
            f.write(f'{len(pos)}\nLatentSpring conditional H flow sample {index}; charge=0 spin=1\n')
            for z,row in zip(c['numbers'],pos.tolist()):f.write(f'{periodic.GetElementSymbol(z)} {row[0]:.12f} {row[1]:.12f} {row[2]:.12f}\n')
    receipt=dict(complete=True,attempted=len(x),selected_outputs=len(x),changed=int(changed.sum()),new_parent_trajectories=0 if a.replay_parent else len(x),
        energy_queries=0,geometry_optimizer_steps=0,backbone_batch_calls=backbone_calls,physical_head_batch_calls=physical_calls,
        decoder_batch_calls=sum(i['network_calls'] for i in infos),decoder_example_calls=sum(i['network_example_calls'] for i in infos),
        parent_seconds=parent_seconds,decoder_seconds=decoder_seconds,device=a.device,seed=a.seed,provenance=provenance,
        samples_sha256=sha(a.out/'samples.pt'),xyz_sha256=sha(a.out/'samples.xyz'))
    (a.out/'manifest.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
