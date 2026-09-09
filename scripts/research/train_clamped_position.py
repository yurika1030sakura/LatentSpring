#!/usr/bin/env python3
"""Train the position factor whose sampler and conditional density agree.

Reuses FlowMol's backbone and processed OMol25 graphs, but trains an unaligned
composition-clamped FM path ending at T. The original checkpoint remains the
separate composition prior. Optimizer gradients from FM and energy are added
sequentially before one update, avoiding simultaneous activation storage.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import dgl
import torch
from flowmol.model_utils.load import read_config_file,model_from_config
from flowmol.data_processing.dataset import MoleculeDataset
from flowmol.data_processing.utils import get_batch_idxs,get_upper_edge_mask
from cfm_mol.clamped_fm import clamped_fm_loss
from cfm_mol.bgfm_density import energy_consistency_loss_per_mol
from cfm_mol.perturbation_loader import PerturbationLoader


def finite_json(value):
    """Keep failed numerical diagnostics as null while preserving skip records."""
    if isinstance(value, dict):
        return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--warm-checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--seed',type=int,default=9004)
    p.add_argument('--terminal-time',type=float,default=.8)
    p.add_argument('--lr',type=float,default=2e-5)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--device',default='cuda')
    p.add_argument('--lambda-energy',type=float,default=0.)
    p.add_argument('--energy-every',type=int,default=4)
    p.add_argument('--energy-steps',type=int,default=16)
    p.add_argument('--energy-solver',choices=['midpoint','rk4'],default='midpoint')
    p.add_argument('--energy-cap',type=float,default=1500.)
    p.add_argument('--energy-estimator',choices=['squared','replica_product'],default='replica_product')
    p.add_argument('--common-probes',action='store_true')
    p.add_argument('--trace-distribution',choices=['rademacher','gaussian'],default='rademacher')
    p.add_argument('--energy-exact-trace',action='store_true')
    p.add_argument('--checkpoint-energy',action='store_true')
    p.add_argument('--discrete-adjoint-energy',action='store_true')
    p.add_argument('--perturbation-indices',type=int,nargs='+')
    p.add_argument('--energy-control',choices=['value','shuffle','zero'],default='value')
    p.add_argument('--energy-gradient-diagnostics',action='store_true')
    p.add_argument('--energy-parents',type=int,default=1)
    p.add_argument('--geometry-softening',type=float)
    p.add_argument('--position-parameterization',choices=['endpoint','displacement'])
    p.add_argument('--position-backbone',choices=['flowmol','radial_reference'])
    p.add_argument('--energy-shard',type=Path,default=Path('/n/holylabs/woo_lab/Lab/yulili/bgfm/processed_data/omol25_4m_processed/perturbation_train_n30000_s0.pt'))
    args=p.parse_args()
    if args.steps<1 or args.batch_size<1 or args.energy_every<1:
        raise ValueError('Steps and batch settings must be positive')
    if args.energy_parents<1 or (args.checkpoint_energy and args.discrete_adjoint_energy):
        raise ValueError('Require positive energy parent count and one gradient mode')
    if not 0<args.terminal_time<=1 or not math.isfinite(args.lr) or args.lr<=0:
        raise ValueError('Require T in (0,1] and a positive finite learning rate')
    if not math.isfinite(args.lambda_energy) or args.lambda_energy<0:
        raise ValueError('Energy weight must be finite and nonnegative')
    torch.manual_seed(args.seed)
    cfg=read_config_file(args.config)
    cfg['mol_fm'].pop('bgfm',None)  # preserve constructor ordering
    if cfg['dataset']['max_atoms']!=200 or cfg['mol_fm']['total_loss_weights']['e']!=0:
        raise ValueError('This protocol requires bond-free OMol25 with max_atoms=200')
    model=model_from_config(cfg)
    warm=torch.load(args.warm_checkpoint,map_location='cpu',weights_only=False)
    warm_protocol=warm.get('research_protocol',{})
    from cfm_mol.radial_reference import prepare_research_backbone,patch_radial_reference
    warm_backbone=warm_protocol.get('position_backbone','flowmol')
    args.position_backbone=warm_backbone if args.position_backbone is None else args.position_backbone
    if warm_backbone=='radial_reference' and args.position_backbone!='radial_reference':
        raise ValueError('Cannot load a radial checkpoint into the original backbone')
    prepare_research_backbone(model,warm_protocol)
    default_head='displacement' if args.position_backbone=='radial_reference' else warm_protocol.get('position_parameterization','endpoint')
    args.position_parameterization=default_head if args.position_parameterization is None else args.position_parameterization
    if args.position_parameterization=='endpoint' and args.terminal_time==1:
        raise ValueError('This endpoint training protocol requires T<1')
    if warm_protocol and warm_protocol['data_endpoint_time']!=args.terminal_time:
        raise ValueError('Warm position checkpoint has a different endpoint time')
    model.load_state_dict(warm['state_dict'],strict=True)
    if args.position_backbone=='radial_reference':
        if args.position_parameterization!='displacement':
            raise ValueError('Radial reference requires the displacement head')
        patch_radial_reference(model)
    from cfm_mol.smooth_geometry import patch_smooth_geometry
    args.geometry_softening=warm_protocol.get('geometry_softening',0.) if args.geometry_softening is None else args.geometry_softening
    if args.position_backbone=='radial_reference' and args.geometry_softening!=0:
        raise ValueError('The radial reference has its own smooth kernels; geometry softening does not apply')
    patch_smooth_geometry(model,args.geometry_softening)
    model.to(args.device).float().train()
    ds_cfg=dict(cfg['dataset'],fake_atom_p=0.,fake_atom_std=1.,
                explicit_aromaticity=cfg['mol_fm'].get('explicit_aromaticity',False))
    # Dataset-generated prior features are unused by clamped_fm_loss; avoiding
    # alignment also prevents wasting work on an incompatible interpolation.
    prior_cfg=cfg['mol_fm']['prior_config']
    prior_cfg['x']['align']=False
    dataset=MoleculeDataset('train',ds_cfg,prior_config=prior_cfg)
    order=torch.randperm(len(dataset),generator=torch.Generator().manual_seed(args.seed+1))
    if args.steps*args.batch_size>len(order):
        raise ValueError('This bounded pilot must not silently recycle its data order')
    energy_loader=None
    if args.lambda_energy:
        energy_loader=PerturbationLoader([args.energy_shard],n_atom_types=model.n_atom_types,
            b_parents=args.energy_parents,device=args.device,max_atoms_per_parent=12,seed=args.seed+2,
            perturbation_indices=args.perturbation_indices)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-12)
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'metrics.jsonl').exists():
        raise ValueError('Refusing to overwrite an existing experiment')
    protocol={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    protocol.update(format='clamped_position_v1',
        composition_prior_checkpoint=warm_protocol.get('composition_prior_checkpoint',str(args.warm_checkpoint)),
        warm_sha256=hashlib.sha256(args.warm_checkpoint.read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        data_order_sha256=hashlib.sha256(order.numpy().tobytes()).hexdigest(),
        data_endpoint_time=args.terminal_time,alignment=False,history_self_conditioning=False,
        steric_retractions=False,energy_temperature_eV=1.,
        purpose='development baseline; not a finished molecular result')
    if energy_loader is not None:
        protocol['energy_shard_sha256']=hashlib.sha256(args.energy_shard.read_bytes()).hexdigest()
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    started=time.monotonic();energy_applied=0;energy_skipped=0
    metrics=args.out/'metrics.jsonl'
    for step in range(args.steps):
        tick=time.monotonic()
        indices=order[step*args.batch_size:(step+1)*args.batch_size].tolist()
        g=dgl.batch([dataset[index] for index in indices]).to(args.device)
        nbi,_=get_batch_idxs(g);uem=get_upper_edge_mask(g)
        optimizer.zero_grad(set_to_none=True)
        fm_generator=torch.Generator(device=args.device).manual_seed(args.seed+1000003*step)
        fm=clamped_fm_loss(model,g,nbi,uem,terminal_time=args.terminal_time,generator=fm_generator,
            parameterization=args.position_parameterization)
        if not torch.isfinite(fm):raise FloatingPointError('Non-finite clamped FM objective')
        fm.backward()
        record={'step':step+1,'data_indices':indices,'fm_loss':float(fm.detach()),
                'n_atoms':g.batch_num_nodes().cpu().tolist(),'energy_applied':False}
        if energy_loader is not None and step%args.energy_every==0:
            gp,energies,pid,pnbi,puem=energy_loader.next_batch()
            record['energy_parent_indices']=energy_loader._last_parent_indices
            if args.energy_control=='zero':
                energies=torch.where(torch.isfinite(energies),torch.zeros_like(energies),energies)
            elif args.energy_control=='shuffle':
                energies=energies.clone()
                shuffle_generator=torch.Generator(device=args.device).manual_seed(args.seed+704729*step)
                for parent in pid.unique():
                    selected=torch.where((pid==parent)&torch.isfinite(energies))[0]
                    order_e=torch.randperm(len(selected),device=args.device,generator=shuffle_generator)
                    energies[selected]=energies[selected[order_e]]
            fm_grads=[None if p.grad is None else p.grad.detach().clone() for p in model.parameters()] if args.energy_gradient_diagnostics else None
            try:
                energy,diagnostics=energy_consistency_loss_per_mol(model,gp,pnbi,puem,
                    energies,pid,kT=1.,n_ode_steps=args.energy_steps,n_hutchinson=0 if args.energy_exact_trace else 1,
                    density_options={'mode':'clamped_cnf','terminal_time':args.terminal_time,
                        'solver':args.energy_solver,
                        'parameterization':args.position_parameterization,
                        'n_trace_replicates':2,'residual_estimator':args.energy_estimator,
                        'trace_distribution':args.trace_distribution,
                        'common_trace_within_parent':args.common_probes,
                        'checkpoint_steps':args.checkpoint_energy,
                        'discrete_adjoint':args.discrete_adjoint_energy,
                        'trace_seed':args.seed+1729+1000033*step})
                finite=bool(torch.isfinite(energy))
                allowed=finite and (args.energy_cap<=0 or abs(float(energy.detach()))<=args.energy_cap)
                record.update(energy_loss=float(energy.detach()) if finite else None,
                              energy_diagnostics=finite_json(diagnostics))
                if allowed:
                    (args.lambda_energy*energy).backward()
                    energy_applied+=1;record['energy_applied']=True
                    if fm_grads is not None:
                        fm_squared=[];aux_squared=[];dots=[]
                        for parameter,original in zip(model.parameters(),fm_grads):
                            if parameter.grad is None:continue
                            auxiliary=parameter.grad if original is None else parameter.grad-original
                            aux_squared.append(auxiliary.square().sum())
                            if original is not None:
                                fm_squared.append(original.square().sum());dots.append((original*auxiliary).sum())
                        fm_norm=torch.stack(fm_squared).sum().sqrt()
                        aux_norm=torch.stack(aux_squared).sum().sqrt()
                        record.update(fm_gradient_norm=float(fm_norm),weighted_energy_gradient_norm=float(aux_norm),
                            fm_energy_gradient_cosine=float(torch.stack(dots).sum()/(fm_norm*aux_norm).clamp_min(1e-30)))
                else:energy_skipped+=1
            except FloatingPointError as exc:
                energy_skipped+=1;record['energy_error']=str(exc)
            del fm_grads
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        record.update(gradient_norm=float(norm),seconds=time.monotonic()-tick)
        with metrics.open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        if (step+1)%25==0 or step==0:
            print(json.dumps(record,allow_nan=False),flush=True)
        if (step+1)%100==0 or step+1==args.steps:
            if any(not torch.isfinite(v).all() for v in model.state_dict().values() if v.is_floating_point()):
                raise FloatingPointError('Non-finite weights after optimizer update')
            temporary=args.out/'last.ckpt.tmp'
            torch.save({'state_dict':model.state_dict(),'optimizer_states':[optimizer.state_dict()],
                'global_step':step+1,'research_protocol':protocol},temporary)
            temporary.replace(args.out/'last.ckpt')
    result={'complete':True,'global_step':args.steps,'all_weights_finite':True,
        'energy_applied':energy_applied,'energy_skipped':energy_skipped,
        'seconds':time.monotonic()-started,'terminal_time':args.terminal_time,
        'peak_gpu_bytes':torch.cuda.max_memory_allocated() if str(args.device).startswith('cuda') else None,
        'scope':'position-factor development training; composition prior stays frozen'}
    (args.out/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
