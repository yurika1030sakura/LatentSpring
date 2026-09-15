#!/usr/bin/env python3
"""Record an existing molecular generation trajectory for scientific figures."""
import argparse
import json
from pathlib import Path
import torch
from flowmol.model_utils.load import read_config_file

import cfm_mol.clamped_density as density
from cfm_mol.nonequilibrium import centered_orthonormal_basis
from scripts.research.run_tree_manifold import make_graph
from scripts.research.tree_prior_fm import restore_model
from scripts.research.train_electronic_fm import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['project','out']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(2);assert not args.out.exists()
    protocol=args.project/'research/evidence/fresh_physics_s0_v1.json';spec=json.loads(protocol.read_text())
    directory=args.project/'runs/fresh_physics_v1/s0/study/evaluation';report=json.loads((directory/'escort_delta_results.json').read_text())
    assert report['complete'] and report['protocol_sha256']==sha(protocol)
    # A deterministic structural display rule, fixed before physical readouts
    # of this figure: first CHON composition and first valid draw in panel order.
    selected=next(r for r in report['rows'] if set(r['condition']['numbers'])<={1,6,7,8} and r['graph_supported'])
    index=selected['condition_index'];sample_index=next(i for i,r in enumerate(selected['records']) if r['graph_supported'])
    file=directory/f'escort_delta_c{index}.pt';assert sha(file)==selected['sample_sha256']
    saved=torch.load(file,weights_only=False,map_location='cpu');c=saved['condition'];batch=spec['evaluation_batch'];begin=sample_index//batch*batch
    checkpoint=args.project/spec['checkpoints']['escort_delta']['path']
    assert sha(checkpoint)==spec['checkpoints']['escort_delta']['sha256']
    warm=torch.load(checkpoint,weights_only=False,map_location='cpu')
    cfg=read_config_file(args.project/spec['config']);cfg['mol_fm'].pop('bgfm',None)
    model=restore_model(cfg,warm).eval().requires_grad_(False);graph,node_batch,upper=make_graph(c,cfg,batch)
    initial=saved['initial_positions'][begin:begin+batch];frames=[];calls=0
    original=density.position_velocity
    def record(*arguments,**keywords):
        nonlocal calls
        if calls%2==0:frames.append(arguments[2].detach().reshape(batch,c['n_atoms'],3).cpu().double().clone())
        calls+=1
        return original(*arguments,**keywords)
    density.position_velocity=record
    try:
        with torch.no_grad():
            final=density.sample_clamped_flow(model,graph,node_batch,upper,x0=initial.reshape(-1,3).cuda().float(),
                n_ode_steps=spec['midpoint_steps'],terminal_time=1.,parameterization='displacement')
    finally:density.position_velocity=original
    final=final.reshape(batch,c['n_atoms'],3).double();final-=final.mean(1,keepdim=True);frames.append(final.cpu().clone())
    basis=centered_orthonormal_basis(c['n_atoms'],device='cuda')
    seed=spec['evaluation_seed']*2000003+index*100003+begin
    noise=torch.randn((batch,c['n_atoms']-1,3),device='cuda',dtype=torch.float64,generator=torch.Generator(device='cuda').manual_seed(seed))
    final+=spec['terminal_noise_std_A']*torch.einsum('nk,bkd->bnd',basis,noise)
    expected=saved['positions'][begin:begin+batch]
    error=float((final.cpu()-expected).abs().max())
    torch.testing.assert_close(final.cpu(),expected,atol=2e-5,rtol=0)
    assert calls==64 and len(frames)==33
    args.out.mkdir(parents=True)
    torch.save(dict(positions=torch.stack(frames),raw_output=expected,initial_positions=initial,
        auxiliary_tree_edges=saved['auxiliary_tree_edges'][begin:begin+batch],condition=c,
        times=torch.linspace(0,1,33,dtype=torch.float64),selected_sample=sample_index-begin,
        original_sample_sha256=sha(file),checkpoint_sha256=sha(checkpoint)),args.out/'trajectory.pt')
    manifest=dict(complete=True,protocol_sha256=sha(protocol),condition_index=index,sample_index=sample_index,
        original_sample=str(file.relative_to(args.project)),original_sample_sha256=sha(file),
        checkpoint=str(checkpoint.relative_to(args.project)),checkpoint_sha256=sha(checkpoint),
        trajectory_sha256=sha(args.out/'trajectory.pt'),maximum_replay_error_A=error,
        replayed_existing_outputs=batch,new_unique_neural_outputs=0,primitive_network_evaluations_per_replay=128,
        new_energy_queries=0,selection='First CHON composition with a valid raw output in the fixed panel; first valid draw, without energy ranking.',
        scope='Actual numerical flow-matching trajectory, with unoptimized final coordinates. Flow time is generation progress, not physical molecular dynamics. Final Gaussian noise is a separately recorded terminal step.')
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))


if __name__=='__main__':main()
