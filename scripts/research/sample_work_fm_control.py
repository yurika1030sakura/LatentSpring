#!/usr/bin/env python3
"""Sample the original FM checkpoint on exactly a work calibration's condition.

Same intrinsic Gaussian prior seeds and electronic state. Midpoint 16/64 costs
32/128 field calls per sample. No FM density or importance weights are claimed.
This diagnoses whether the stochastic bridge degraded the starting FM geometry;
it is not an equal-training-cost Boltzmann comparison.
"""
import argparse
import json
from pathlib import Path
import time

import dgl
import torch
from flowmol.model_utils.load import read_config_file,model_from_config
from flowmol.data_processing.utils import get_batch_idxs,get_upper_edge_mask

from cfm_mol.clamped_density import sample_clamped_flow
from cfm_mol.condition_systems import graph_from_condition
from cfm_mol.nonequilibrium import centered_orthonormal_basis
from cfm_mol.radial_reference import prepare_research_backbone
from cfm_mol.smooth_geometry import patch_smooth_geometry
from molecular_tempered_pilot import sha,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--particles',type=int,help='Optional fixed baseline sample count; defaults to source evaluation count')
    p.add_argument('--device',default='cuda');args=p.parse_args()
    source=args.work_run/'results.json';old=json.loads(source.read_text())
    if not old['complete']:raise ValueError('Require a completed source calibration')
    cfgpath=Path(old['configuration']['config']);ckpt=Path(old['configuration']['checkpoint'])
    if sha(ckpt)!=old['checkpoint_sha256']:raise ValueError('Initialization checkpoint has changed')
    cfg=read_config_file(cfgpath);cfg['mol_fm'].pop('bgfm',None);cfg['mol_fm']['prior_config']['x']['align']=False
    if cfg['dataset']['max_atoms']!=200 or cfg['mol_fm']['total_loss_weights']['e']!=0:raise ValueError('Require bond-free max_atoms=200')
    state=torch.load(str(ckpt),map_location='cpu',weights_only=False);protocol=state['research_protocol']
    if protocol['position_parameterization']!='displacement' or protocol['data_endpoint_time']!=1.:
        raise ValueError('Require explicit T=1 displacement FM checkpoint')
    model=model_from_config(cfg);prepare_research_backbone(model,protocol)
    model.load_state_dict(state['state_dict'],strict=True);patch_smooth_geometry(model,protocol.get('geometry_softening',0.))
    model=model.to(args.device).float().eval();del state
    condition=old['condition'];n=len(condition['numbers']);dimension=3*(n-1)
    settings=old['configuration'];batch=settings['batch'];count=settings['eval_particles'] if args.particles is None else args.particles;seed=settings['seed']
    if count<1 or count%batch:raise ValueError('Positive evaluation count divisible by source batch size required')
    model_kT=protocol.get('requested_kT',1.)
    base=graph_from_condition({'atomic_numbers':condition['numbers'],'charge':condition['charge'],
        'spin_multiplicity':condition['spin_multiplicity'],'requested_kT_eV':model_kT},cfg['dataset']['atom_map'],
        n_bond_classes=5 if cfg['mol_fm'].get('explicit_aromaticity',False) else 4)
    graph=dgl.batch([base]*batch).to(args.device);nbi,_=get_batch_idxs(graph);uem=get_upper_edge_mask(graph)
    basis=centered_orthonormal_basis(n,device=args.device)
    args.out.mkdir(parents=True,exist_ok=True);output=args.out/'results.json'
    if output.exists():raise FileExistsError(output)
    report={'complete':False,'scope':__doc__,'condition':condition,'source_results':str(source.resolve()),
        'source_sha256':sha(source),'checkpoint_sha256':sha(ckpt),'script_sha256':sha(Path(__file__)),
        'reference_geometry_loaded':False,'prior_seed_rule':'900000001 + source_seed * 1009 + batch_index',
        'model_input_kT_eV':model_kT,'source_target_kT_eV':settings['kT'],
        'source_seed':seed,'particles':count,'prior_std':old['prior_std'],'solver':'midpoint',
        'steps':[16,64],'neural_field_calls_per_sample':[32,128],'samples':[]}
    write_json(output,report);outputs={steps:[] for steps in report['steps']};start=time.perf_counter()
    for index in range(count//batch):
        generator=torch.Generator(device=args.device).manual_seed(900000001+seed*1009+index)
        z=torch.randn((batch,dimension),device=args.device,dtype=torch.float64,generator=generator)*old['prior_std']
        x0=torch.einsum('nk,bkd->bnd',basis,z.reshape(batch,n-1,3)).reshape(batch*n,3).float()
        for steps in report['steps']:
            x=sample_clamped_flow(model,graph,nbi,uem,x0=x0,n_ode_steps=steps,terminal_time=1.,parameterization='displacement')
            outputs[steps].append(x.cpu().reshape(batch,n,3))
        if (index+1)%8==0:print(json.dumps({'batches':index+1,'seconds':time.perf_counter()-start}),flush=True)
    for steps in report['steps']:
        outputs[steps]=torch.cat(outputs[steps]);name=f'fm_midpoint_{steps}_samples.pt'
        torch.save({'positions':outputs[steps],'condition':condition,'density_scope':'not evaluated'},args.out/name)
        report['samples'].append({'file':name,'sha256':sha(args.out/name)})
    drift=(outputs[64]-outputs[16]).double().square().sum(-1).mean(-1).sqrt()
    report.update(complete=True,seconds=time.perf_counter()-start,coordinate_rms_16_64_A=drift.tolist(),
        peak_gpu_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None)
    write_json(output,report)
    print(json.dumps({'complete':True,'max_coordinate_rms_16_64_A':float(drift.max()),'seconds':report['seconds']}),flush=True)


if __name__=='__main__':main()
