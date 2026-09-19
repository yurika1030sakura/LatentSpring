#!/usr/bin/env python3
"""Generate and SAVE all raw outputs from an adapted real LatentSpring model.

No energy/force calls, rejection, geometry relaxation, or terminal noise. The
condition file is a JSON list (or {'rows': [...]}) of atomic_numbers, charge,
and spin_multiplicity. Graph/basin/reference fields, if present, are not read.
The checkpoint records a single-temperature target; this is not a temperature
sweep or a claim that the finite network has exact Boltzmann frequencies.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from cfm_mol.weighted_endpoints import EndpointDraw, file_sha256
from cfm_mol.weighted_endpoint_fm import dgl_batch, checkpoint_source
from cfm_mol.source_checkpoint import prior_from_checkpoint
from cfm_mol.clamped_density import sample_clamped_flow


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','config','conditions','out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--samples',type=int,default=32)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--midpoint-steps',type=int,default=32)
    p.add_argument('--seed',type=int,default=260948)
    p.add_argument('--device',default='cuda')
    p.add_argument('--allow-legacy-pickle',action='store_true')
    a=p.parse_args()
    if min(a.samples,a.batch_size,a.midpoint_steps)<1:raise ValueError('Positive counts required')
    from flowmol.model_utils.load import read_config_file,model_from_config
    from cfm_mol.radial_reference import prepare_research_backbone
    from cfm_mol.smooth_geometry import patch_smooth_geometry
    cfg=read_config_file(a.config);cfg['mol_fm'].pop('bgfm',None)
    if cfg['dataset']['max_atoms']!=200 or cfg['mol_fm']['total_loss_weights']['e']!=0:
        raise ValueError('Expected the original bond-free configuration')
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=not a.allow_legacy_pickle)
    recipe=state['research_protocol'];target=recipe.get('direct_endpoint_training') or recipe['mass_preserving_minima']
    if recipe.get('position_parameterization')!='displacement':raise ValueError('Wrong head semantics')
    if target['terminal_noise_std_A']!=0:raise ValueError('Unexpected terminal noise declaration')
    model=model_from_config(cfg);prepare_research_backbone(model,recipe)
    model.load_state_dict(state['state_dict'],strict=True)
    patch_smooth_geometry(model,recipe.get('geometry_softening',0.))
    device=torch.device(a.device);model=model.to(device).float().eval().requires_grad_(False)
    prior=prior_from_checkpoint(state)
    source=json.loads(a.conditions.read_text());conditions=source['rows'] if isinstance(source,dict) else source
    a.out.mkdir(parents=True,exist_ok=False);rows=[];torch.set_num_threads(2)
    for i,c in enumerate(conditions):
        # The whitelist is intentional: never use optional reference coordinates.
        z=c['atomic_numbers'];condition=dict(atomic_numbers=z,numbers=z,n_atoms=len(z),
            charge=c['charge'],spin_multiplicity=c['spin_multiplicity'])
        positions=[];initial=[];start=time.perf_counter()
        for k in range(0,a.samples,a.batch_size):
            batch=min(a.batch_size,a.samples-k)
            draws=[EndpointDraw('generation','none','none',condition,np.zeros((len(z),3))) for _ in range(batch)]
            g,nbi,uem=dgl_batch(draws,cfg,device,model_kT_eV=target['model_kT_eV'])
            g.ndata['has_reference_geometry'].zero_()
            seed=a.seed*1000003+i*100003+k
            x0=checkpoint_source(prior,draws,seed)
            with torch.no_grad():
                y=sample_clamped_flow(model,g,nbi,uem,x0=x0.to(device).float(),
                    n_ode_steps=a.midpoint_steps,terminal_time=1.,parameterization='displacement')
            positions.append(y.reshape(batch,len(z),3).cpu().numpy())
            initial.append(x0.reshape(batch,len(z),3).numpy())
        if device.type=='cuda':torch.cuda.synchronize()
        path=a.out/f'raw_condition_{i:04d}.npz'
        np.savez_compressed(path,raw_positions=np.concatenate(positions),source_positions=np.concatenate(initial),
            atomic_numbers=np.asarray(z,dtype=np.int64))
        rows.append(dict(condition=condition,attempted=a.samples,seconds=time.perf_counter()-start,
                         file=path.name,sha256=file_sha256(path)))
    result=dict(complete=True,rows=rows,checkpoint_sha256=file_sha256(a.checkpoint),
        config_sha256=file_sha256(a.config),condition_sha256=file_sha256(a.conditions),
        teacher_temperature_K=target['teacher_temperature_K'],model_kT_feature_eV=target['model_kT_eV'],
        geometry_refinement=False,energy_filter=False,terminal_noise_A=0.,physical_queries=0,
        primitive_network_calls_per_attempt=a.midpoint_steps*2*(2 if recipe.get('geometry_self_conditioning') else 1),
        scientific_metrics_evaluated=False)
    (a.out/'generation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(str(a.out/'generation.json'))


if __name__=='__main__':main()
