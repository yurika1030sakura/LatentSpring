#!/usr/bin/env python3
"""Matched-prior conditional samples on composition-disjoint development data.

No reference coordinates initialise sampling. The stored reference supplies
only composition/charge to the graph; independent Gaussian priors supply X0.
64/128 coordinate comparisons diagnose integration separately from quality.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import dgl
import torch
from flowmol.model_utils.load import read_config_file,model_from_config
from flowmol.data_processing.dataset import MoleculeDataset
from flowmol.data_processing.utils import get_batch_idxs,get_upper_edge_mask
from cfm_mol.clamped_density import sample_clamped_flow
from cfm_mol.clamped_fm import clamped_fm_loss
from checkpoint_panel import write_json


def geometry_summary(x):
    distances=torch.pdist(x)
    return {'min_pair_distance_A':float(distances.min()) if len(distances) else None,
            'radius_of_gyration_A':float((x-x.mean(0)).square().sum(-1).mean().sqrt())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--checkpoint',nargs=2,action='append',metavar=('ARM','PATH'),required=True)
    p.add_argument('--split',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--parents',type=int,default=8)
    p.add_argument('--samples',type=int,default=4)
    p.add_argument('--steps',type=int,nargs='+',default=[64,128])
    p.add_argument('--seed',type=int,default=9005)
    p.add_argument('--terminal-time',type=float,default=.8)
    p.add_argument('--device',default='cuda')
    p.add_argument('--solver',choices=['midpoint','rk4'],default='midpoint')
    args=p.parse_args()
    if args.parents<1 or args.samples<1 or len(set(arm for arm,_ in args.checkpoint))!=len(args.checkpoint):
        raise ValueError('Require positive counts and unique arm names')
    if not args.steps or any(s<1 for s in args.steps) or args.steps!=sorted(set(args.steps)):
        raise ValueError('Sampling resolutions must be positive, distinct and increasing')
    split=json.loads(args.split.read_text())
    cfg=read_config_file(args.config);cfg['mol_fm'].pop('bgfm',None)
    cfg['mol_fm']['prior_config']['x']['align']=False
    ds_cfg=dict(cfg['dataset'],fake_atom_p=0.,fake_atom_std=1.,
                explicit_aromaticity=cfg['mol_fm'].get('explicit_aromaticity',False))
    dataset=MoleculeDataset('val',ds_cfg,prior_config=cfg['mol_fm']['prior_config'])
    eligible=split['validation_indices_by_max_atoms']['12']
    # Boundary charges may have been clipped in the old preprocessing. Exclude
    # them before looking at any generated outputs, never infer original spin.
    eligible=[i for i in eligible if all(-2<int(c)<3 for c in
        dataset.atom_charges[slice(*map(int,dataset.node_idx_array[i]))])]
    if len(eligible)<args.parents:raise ValueError('Insufficient eligible held compositions')
    order=torch.randperm(len(eligible),generator=torch.Generator().manual_seed(args.seed))
    indices=[eligible[int(i)] for i in order[:args.parents]]
    graphs=[dataset[i] for i in indices]
    report={'claim':'conditional geometry development comparison; not de novo or Boltzmann validation',
        'split_sha256':hashlib.sha256(args.split.read_bytes()).hexdigest(),
        'config_sha256':hashlib.sha256(args.config.read_bytes()).hexdigest(),
        'terminal_time':args.terminal_time,'seed':args.seed,'validation_indices':indices,
        'sampling_steps':args.steps,
        'solver':args.solver,
        'selection':'seeded random <=12-atom composition-disjoint validation, excluding clipped charge boundaries',
        'reference_coordinate_use':'composition only for generation; independent normal prior; used as FM target for separate loss diagnostic',
        'spin_metadata_available':False,'arms':[],'references':[],'complete':False}
    args.out.mkdir(parents=True,exist_ok=True)
    for index,g in zip(indices,graphs):
        charges=g.ndata['c_1_true'].argmax(-1)-2
        symbols=[cfg['dataset']['atom_map'][int(i)] for i in g.ndata['a_1_true'].argmax(-1)]
        report['references'].append({'validation_index':index,'symbols':symbols,
            'charge_recorded':int(charges.sum()),'spin':None,
            'positions':g.ndata['x_1_true'].tolist(),**geometry_summary(g.ndata['x_1_true'])})
    write_json(args.out/'samples.json',report)
    for name,path in args.checkpoint:
        checkpoint=Path(path);state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        protocol=state.get('research_protocol')
        parameterization=(protocol or {}).get('position_parameterization','endpoint')
        if protocol is not None and protocol['data_endpoint_time']!=args.terminal_time:
            raise ValueError('Evaluation T differs from position training endpoint')
        model=model_from_config(cfg)
        from cfm_mol.radial_reference import prepare_research_backbone
        prepare_research_backbone(model,protocol or {})
        model.load_state_dict(state['state_dict'],strict=True);model.to(args.device).float().eval()
        from cfm_mol.smooth_geometry import patch_smooth_geometry
        patch_smooth_geometry(model,(protocol or {}).get('geometry_softening',0.))
        metadata=None
        metadata_path=(protocol or {}).get('electronic_metadata')
        if metadata_path:
            from ase.data import atomic_numbers
            from cfm_mol.electronic_metadata import ElectronicMetadata
            metadata=ElectronicMetadata(metadata_path,'val',[atomic_numbers[s] for s in cfg['dataset']['atom_map']])
            metadata.verify_processed_file(dataset.processed_data_dir/'val_data_processed.pt')
            report['spin_metadata_available']=True
            report['electronic_metadata_progress_sha256']=metadata.progress_sha256
            for reference in report['references']:
                source_index=int(metadata.indices[reference['validation_index']])
                reference['spin']=int(metadata.values['spin_multiplicity'][source_index])
                reference['charge_recorded']=int(metadata.values['total_charge'][source_index])
                reference['raw_source_index']=int(metadata.values['raw_indices'][source_index])
        prior_std=(protocol or {}).get('prior_std',1.)
        arm={'name':name,'checkpoint':str(checkpoint),
            'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'position_training_steps':state['global_step'] if protocol is not None else 0,
            'geometry_softening':(protocol or {}).get('geometry_softening',0.),
            'position_parameterization':parameterization,
            'position_backbone':(protocol or {}).get('position_backbone','flowmol'),
            'prior_std':prior_std,
            'electronic_conditioning':(protocol or {}).get('electronic_conditioning',False),
            'samples':[],'held_fm_losses':[]}
        report['arms'].append(arm)
        for row,(index,base) in enumerate(zip(indices,graphs)):
            g=dgl.batch([base]*args.samples).to(args.device)
            if metadata is not None:metadata.attach(g,[index]*args.samples,(protocol or {}).get('requested_kT',1.))
            nbi,_=get_batch_idxs(g);uem=get_upper_edge_mask(g)
            generator=torch.Generator().manual_seed(args.seed+100003*index)
            x0=torch.randn(g.ndata['x_1_true'].shape,generator=generator).to(args.device)*prior_std
            outputs=[];start=time.monotonic()
            for steps in args.steps:
                outputs.append(sample_clamped_flow(model,g,nbi,uem,x0=x0,
                    n_ode_steps=steps,terminal_time=args.terminal_time,parameterization=parameterization,solver=args.solver).cpu())
            n=base.num_nodes()
            for sample in range(args.samples):
                sl=slice(sample*n,(sample+1)*n)
                x=outputs[-1][sl]
                item={'validation_index':index,'sample_id':sample,'positions':x.tolist(),
                      **geometry_summary(x),'seconds_per_parent':time.monotonic()-start}
                if len(outputs)>=2:
                    drift=float((outputs[-1][sl]-outputs[-2][sl]).square().sum(-1).mean().sqrt())
                    item['coordinate_convergence']={'coarse_steps':args.steps[-2],
                        'fine_steps':args.steps[-1],'rms_A':drift}
                    if args.steps[-2:]==[64,128]:item['coordinate_rms_64_128_A']=drift
                arm['samples'].append(item)
            # Matched FM draws across arms; loss is a diagnostic, not likelihood.
            with torch.no_grad():
                generator=torch.Generator(device=args.device).manual_seed(args.seed+index)
                fm=clamped_fm_loss(model,g,nbi,uem,terminal_time=args.terminal_time,generator=generator,
                    parameterization=parameterization,prior_std=prior_std)
            arm['held_fm_losses'].append({'validation_index':index,'loss':float(fm)})
            write_json(args.out/'samples.json',report)
            print(json.dumps({'arm':name,'parent':index,'seconds':time.monotonic()-start}),flush=True)
        del model,state
        if str(args.device).startswith('cuda'):torch.cuda.empty_cache()
    report['complete']=True
    write_json(args.out/'samples.json',report)


if __name__=='__main__':main()
