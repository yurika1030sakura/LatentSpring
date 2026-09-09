#!/usr/bin/env python3
"""Frozen-FM geometry baseline on an outcome-independent condition manifest.

Only elemental identities and electronic state are loaded; reference coordinates
are not an input. Every selected condition is attempted and failures are retained.
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
from cfm_mol.condition_systems import load_condition,graph_from_condition
from cfm_mol.nonequilibrium import centered_orthonormal_basis
from cfm_mol.radial_reference import prepare_research_backbone
from cfm_mol.smooth_geometry import patch_smooth_geometry
from molecular_tempered_pilot import sha,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--samples',type=int,default=32);p.add_argument('--batch',type=int,default=16)
    p.add_argument('--seed',type=int,default=9063);p.add_argument('--device',default='cuda');args=p.parse_args()
    if min(args.samples,args.batch)<1 or args.samples%args.batch:raise ValueError('Positive divisible sample and batch counts required')
    manifest=json.loads(args.manifest.read_text())
    if not manifest['complete'] or manifest['role']!='new_development':raise ValueError('Only completed development manifests are allowed')
    output=args.out/'panel.json'
    if output.exists():raise FileExistsError(output)
    cfg=read_config_file(args.config);cfg['mol_fm'].pop('bgfm',None);cfg['mol_fm']['prior_config']['x']['align']=False
    if cfg['dataset']['max_atoms']!=200 or cfg['mol_fm']['total_loss_weights']['e']!=0:raise ValueError('Require bond-free max_atoms=200')
    state=torch.load(str(args.checkpoint),map_location='cpu',weights_only=False);protocol=state['research_protocol']
    if not protocol.get('electronic_conditioning') or protocol['position_parameterization']!='displacement' or protocol['data_endpoint_time']!=1.:
        raise ValueError('Require metadata-aware T=1 displacement checkpoint')
    model=model_from_config(cfg);prepare_research_backbone(model,protocol);model.load_state_dict(state['state_dict'],strict=True)
    patch_smooth_geometry(model,protocol.get('geometry_softening',0.));model=model.to(args.device).float().eval();del state
    args.out.mkdir(parents=True,exist_ok=True)
    report={'complete':False,'scope':__doc__,'manifest_sha256':sha(args.manifest),'checkpoint_sha256':sha(args.checkpoint),
        'config_sha256':sha(args.config),'script_sha256':sha(Path(__file__)),'samples_per_condition':args.samples,
        'batch':args.batch,'seed':args.seed,'reference_geometry_loaded':False,'rows':[],
        'model_input_kT':protocol['requested_kT'],'importance_weights':'not evaluated'}
    write_json(output,report);start=time.perf_counter()
    for index in range(len(manifest['rows'])):
        condition=load_condition(args.manifest,index);condition['requested_kT_eV']=protocol['requested_kT']
        condition_report={**condition,'numbers':condition['atomic_numbers']}
        row={'panel_index':index,'condition':condition_report,'success':False};directory=args.out/f'condition_{index:02d}'
        directory.mkdir(exist_ok=False)
        try:
            base=graph_from_condition(condition,cfg['dataset']['atom_map'],n_bond_classes=5 if cfg['mol_fm'].get('explicit_aromaticity',False) else 4)
            n=base.num_nodes();graph=dgl.batch([base]*args.batch).to(args.device);nbi,_=get_batch_idxs(graph);uem=get_upper_edge_mask(graph)
            basis=centered_orthonormal_basis(n,device=args.device);outputs={steps:[] for steps in [16,64]}
            for batch_index in range(args.samples//args.batch):
                generator=torch.Generator(device=args.device).manual_seed(args.seed+100003*condition['candidate_index']+batch_index)
                z=torch.randn((args.batch,3*(n-1)),device=args.device,dtype=torch.float64,generator=generator)*protocol['prior_std']
                x0=torch.einsum('nk,bkd->bnd',basis,z.reshape(args.batch,n-1,3)).reshape(args.batch*n,3).float()
                for steps in outputs:
                    x=sample_clamped_flow(model,graph,nbi,uem,x0=x0,n_ode_steps=steps,terminal_time=1.,parameterization='displacement')
                    outputs[steps].append(x.cpu().reshape(args.batch,n,3))
            files=[]
            for steps in outputs:
                outputs[steps]=torch.cat(outputs[steps]);path=directory/f'fm_midpoint_{steps}_samples.pt'
                torch.save({'positions':outputs[steps],'condition':condition_report,'density_scope':'not evaluated'},path)
                files.append({'file':path.name,'sha256':sha(path)})
            drift=(outputs[64]-outputs[16]).double().square().sum(-1).mean(-1).sqrt()
            detail={'complete':True,'condition':condition_report,'samples':files,'particles':args.samples,
                'neural_field_calls_per_sample':[32,128],'coordinate_rms_16_64_A':drift.tolist(),
                'manifest_sha256':report['manifest_sha256'],'checkpoint_sha256':report['checkpoint_sha256'],
                'reference_geometry_loaded':False,'importance_weights':'not evaluated'}
            write_json(directory/'results.json',detail)
            row.update(success=True,results=str((directory/'results.json').resolve()),
                results_sha256=sha(directory/'results.json'),max_coordinate_rms_16_64_A=float(drift.max()))
        except Exception as exc:
            row['error']=f'{type(exc).__name__}: {exc}'
            write_json(directory/'failure.json',row)
        report['rows'].append(row);write_json(output,report);print(json.dumps(row),flush=True)
    report.update(complete=True,seconds=time.perf_counter()-start,successful_conditions=sum(row['success'] for row in report['rows']))
    write_json(output,report)


if __name__=='__main__':main()
