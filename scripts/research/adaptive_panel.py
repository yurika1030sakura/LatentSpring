#!/usr/bin/env python3
"""Recheck the easy and two most unstable development-panel geometries."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from flowmol.model_utils.load import read_config_file,model_from_config
from cfm_mol.perturbation_loader import PerturbationLoader
from cfm_mol.clamped_density import log_density_clamped_flow
from cfm_mol.clamped_reference import log_density_clamped_reference


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    args=p.parse_args();source=json.loads(args.panel.read_text())
    if not source['complete']:raise ValueError('Source panel is incomplete')
    checkpoint=Path(source['checkpoint'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=source['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    cfg=read_config_file('configs/sweep/a1_fm_only_s2.yaml');cfg['mol_fm'].pop('bgfm',None)
    model=model_from_config(cfg)
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['state_dict'],strict=True)
    model.to(args.device).double().eval()
    loader=PerturbationLoader([source['shard']],n_atom_types=model.n_atom_types,
        b_parents=1,device=args.device,max_atoms_per_parent=12)
    rows=source['rows']
    hardest=sorted(range(len(rows)),key=lambda i:rows[i]['resolutions'][-1]['max_mean_centered_change_from_previous'],reverse=True)[:2]
    selected=list(dict.fromkeys([0]+hardest))
    report={'selection':'first parent plus two largest 64-to-128 mean centered changes; development diagnosis',
        'source_panel':str(args.panel),'source_panel_sha256':hashlib.sha256(args.panel.read_bytes()).hexdigest(),
        'dtype':'float64','rows':[],'complete':False}
    args.out.mkdir(parents=True,exist_ok=True)
    for index in selected:
        original=rows[index];parent=original['parent_id']
        loader._order=torch.tensor([parent]);loader._ptr=0
        g,e,pid,nbi,uem=loader.next_batch()
        fingerprint=hashlib.sha256(g.ndata['x_1_true'].cpu().numpy().tobytes()).hexdigest()
        if fingerprint!=original['geometry_sha256']:raise ValueError('Geometry changed')
        for key,value in list(g.ndata.items()):
            if value.is_floating_point():g.ndata[key]=value.double()
        for key,value in list(g.edata.items()):
            if value.is_floating_point():g.edata[key]=value.double()
        row={'parent_id':parent,'geometry_sha256':fingerprint,'probe_seed':9100+index,'references':[]}
        gen=torch.Generator().manual_seed(9100+index)
        x=g.ndata['x_1_true']
        probes=[(2*torch.randint(0,2,x.shape,generator=gen)-1).to(x) for _ in range(8)]
        q=log_density_clamped_flow(model,g,nbi,uem,n_ode_steps=128,n_hutchinson=1,
            n_trace_replicates=8,xi_fn=lambda step,k,x:probes[k])
        row['midpoint_128_float64']=q.cpu().tolist()
        for rtol in [1e-5,1e-7]:
            result=log_density_clamped_reference(model,g,nbi,uem,rtol=rtol,atol=rtol/100,
                quadrature_orders=(2,4),n_replicates=8,seed=9100+index,max_nfe=5000)
            row['references'].append(result)
            print(json.dumps({'parent':parent,'rtol':rtol,'nfe':result['nfe'],
                'intervals':result['accepted_intervals'],'seconds':result['total_seconds']}),flush=True)
        report['rows'].append(row)
        (args.out/'reference.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    report['complete']=True
    (args.out/'reference.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
