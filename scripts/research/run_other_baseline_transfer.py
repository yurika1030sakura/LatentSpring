"""Frozen FM-head transfer to three archived, separately trained baselines."""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from cfm_mol import matched_egnn as base
from cfm_mol.matched_physical_connection import PhysicalFieldTransform
from scripts.research.run_atomwise_connection import restore_head
from scripts.research.run_matched_connection import load_parent, score
from scripts.research.run_matched_generators import write
from scripts.research.run_gaga_feedback import atomic_save
from scripts.research.audit_generator_output_support import assess
from scripts.research.audit_seed_replication import verify_physics
from scripts.research.train_electronic_fm import sha


def run(root, protocol, target, fit, out):
    spec=json.loads(protocol.read_text()); ph=sha(protocol)
    assert spec['frozen'] and target in spec['targets'] and fit in spec['fits']
    assert sha(root/spec['panel'])==spec['panel_sha256']
    out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():
        done=json.loads((out/'complete.json').read_text())
        assert done['complete'] and done['protocol_sha256']==ph
        assert sha(out/'audit.npz')==done['arrays_sha256']
        return
    arm=spec['parents'][fit][target]; model=load_parent(root,arm)
    head,_=restore_head(root,spec['heads'][fit]);head.requires_grad_(False)
    mh,hh=base.state_hash(model),base.state_hash(head)
    assert hh==spec['heads'][fit]['head_state_sha256']
    source=base.HarmonicSource(arm['spec']['edge_log_width'])
    folder=out/'generation';folder.mkdir(exist_ok=True);rows=[];positions={};graphs={}
    for ci,c in enumerate(spec['test_rows']):
        initial_by_arm=[]
        for ai,strength in enumerate(spec['strengths']):
            label='parent' if ai==0 else 'transferred';file=folder/f'{label}_c{ci}.pt'
            if file.exists():
                saved=torch.load(file,map_location='cpu',weights_only=False)
                assert saved['protocol_sha256']==ph and saved['model_state_sha256']==mh and saved['head_state_sha256']==hh
                assert saved['condition']==c and saved['strength']==strength
            else:
                transform=PhysicalFieldTransform(model,arm['spec'],head,strength,strength_limit=4.)
                xx=[];zz=[];calls=[0];tick=time.perf_counter()
                handle=model.dynamics.egnn.register_forward_hook(lambda *_:calls.__setitem__(0,calls[0]+1))
                for begin in range(0,16,8):
                    seed=spec['evaluation_seeds'][fit]*1000003+ci*100003+begin
                    x,z=base.sample(model,c['numbers'],target,arm['spec'],source,seed,8,128,field_transform=transform)
                    assert torch.isfinite(x).all();xx.append(x.cpu().double());zz.append(z.cpu().double())
                handle.remove();torch.cuda.synchronize()
                assert calls[0]==256 and transform.calls==256
                saved=dict(positions=torch.cat(xx),initial_positions=torch.cat(zz),condition=c,strength=strength,
                    protocol_sha256=ph,model_state_sha256=mh,head_state_sha256=hh,core_calls=256,head_calls=256,
                    generation_seconds=time.perf_counter()-tick)
                atomic_save(saved,file)
            assert saved['core_calls']==saved['head_calls']==256 and len(saved['positions'])==16
            if ci==0 and ai==0:
                seed=spec['evaluation_seeds'][fit]*1000003
                direct,initial=base.sample(model,c['numbers'],target,arm['spec'],source,seed,8,128)
                torch.testing.assert_close(direct.cpu().double(),saved['positions'][:8],atol=0,rtol=0)
                torch.testing.assert_close(initial.cpu().double(),saved['initial_positions'][:8],atol=0,rtol=0)
            quality=assess(saved['positions'],c,list(range(16)))
            rows.append(dict(method=label,strength=strength,condition_index=ci,file=file.name,sha256=sha(file),**quality))
            initial_by_arm.append(saved['initial_positions']);positions[ai,ci]=saved['positions']
            graphs[ai,ci]=[v['graph_supported'] for v in quality['records']]
            print(json.dumps(dict(target=target,fit=fit,condition=ci,method=label,graph=quality['graph_supported'],attempted=16)),flush=True)
        torch.testing.assert_close(*initial_by_arm,atol=0,rtol=0)
    assert base.state_hash(model)==mh and base.state_hash(head)==hh
    report=dict(complete=True,protocol_sha256=ph,rows=rows,model_state_sha256=mh,head_state_sha256=hh,new_neural_outputs=2048)
    write(folder/'generation.json',report)
    del model,head;gc.collect();torch.cuda.empty_cache()
    if not (out/'xtb/results.json').exists():score(spec,ph,fit,[(folder,report)],out/'xtb')
    physical=json.loads((out/'xtb/results.json').read_text())
    assert physical['complete'] and physical['protocol_sha256']==ph and physical['new_gfn2_attempts']==2048
    arrays=dict(graph=np.zeros((2,64,16),bool),success=np.zeros((2,64,16),bool),
        force=np.full((2,64,16),np.inf),energy=np.full((2,64,16),np.nan))
    cache=set();seen=set()
    for row in physical['rows']:
        ai=['parent','transferred'].index(row['method']);ci,j=row['condition_index'],row['sample_index']
        assert (ai,ci,j) not in seen;seen.add((ai,ci,j))
        ok,energy,force=verify_physics(root,row,out/'xtb/details'/row['task_id'],positions[ai,ci][j],spec['test_rows'][ci],cache)
        graph=bool(graphs[ai,ci][j]);assert row['graph']==graph and row['joint']==bool(graph and ok and force<=5)
        for key,value in [('graph',graph),('success',ok),('energy',energy),('force',force)]:arrays[key][ai,ci,j]=value
    assert len(seen)==2048
    np.savez_compressed(out/'audit.npz',**arrays)
    write(out/'complete.json',dict(complete=True,protocol_sha256=ph,target=target,fit=fit,
        arrays_sha256=sha(out/'audit.npz'),generation_sha256=sha(folder/'generation.json'),physical_sha256=sha(out/'xtb/results.json'),
        model_state_sha256=mh,head_state_sha256=hh,new_parent_trajectories=2048,new_gfn2_attempts=2048,
        new_esen_queries=0,new_optimizer_steps=0,verification_replays=8,summary=physical['summary']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','protocol','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--target',choices=['edm','gaussian_fm','harmonic_fm'],required=True)
    a=p.parse_args();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    for fit in [0,1]:run(a.project.resolve(),a.protocol.resolve(),a.target,fit,a.out/f's{fit}')


if __name__=='__main__':main()
