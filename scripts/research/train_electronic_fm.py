#!/usr/bin/env python3
"""Matched FM continuation with verified global charge/spin or legacy features.

This is an empirical-data FM baseline, not Boltzmann training. Requested kT is
a model condition for future teacher training, not a temperature label inferred
for the OMol corpus. Warm starts reset AdamW; optimizer state is saved explicitly.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import dgl
import torch
from ase.data import atomic_numbers
from flowmol.model_utils.load import read_config_file,model_from_config
from flowmol.data_processing.dataset import MoleculeDataset
from flowmol.data_processing.utils import get_batch_idxs,get_upper_edge_mask

from cfm_mol.clamped_fm import clamped_fm_loss
from cfm_mol.electronic_conditioning import patch_electronic_conditioning
from cfm_mol.electronic_metadata import ElectronicMetadata
from cfm_mol.radial_reference import prepare_research_backbone
from cfm_mol.smooth_geometry import patch_smooth_geometry


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--warm-checkpoint',type=Path,required=True)
    p.add_argument('--metadata',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=10000);p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--seed',type=int,default=9041);p.add_argument('--global-state',type=int,choices=[0,1],default=1)
    p.add_argument('--prior-std',type=float,default=1.);p.add_argument('--requested-kT',type=float,default=1.)
    p.add_argument('--lr',type=float,default=2e-5);p.add_argument('--device',default='cuda')
    p.add_argument('--pairing',choices=['none','independent','rotation','steric'],default='none')
    p.add_argument('--pairing-protocol',type=Path)
    args=p.parse_args()
    if args.pairing_protocol:
        frozen=json.loads(args.pairing_protocol.read_text())
        assert frozen['frozen'] and args.pairing in frozen['methods']
        assert args.seed==frozen['training_seed'] and args.steps==frozen['training_steps']
        assert args.lr==frozen['learning_rate'] and args.batch_size==frozen['batch_size']
        assert args.global_state==1 and args.prior_std==1. and args.requested_kT==1.
        assert sha(args.warm_checkpoint)==frozen['warm_checkpoint_sha256']
        assert sha(args.config)==frozen['config_sha256']
    if min(args.steps,args.batch_size)<1 or min(args.prior_std,args.requested_kT,args.lr)<=0:raise ValueError('Invalid training configuration')
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'metrics.jsonl').exists():raise FileExistsError('Refusing to overwrite training')
    torch.manual_seed(args.seed)
    cfg=read_config_file(args.config);cfg['mol_fm'].pop('bgfm',None)
    if cfg['dataset']['max_atoms']!=200 or cfg['mol_fm']['total_loss_weights']['e']!=0:raise ValueError('Require bond-free max_atoms=200 protocol')
    cfg['mol_fm']['prior_config']['x']['align']=False
    metadata=ElectronicMetadata(args.metadata,'train',[atomic_numbers[s] for s in cfg['dataset']['atom_map']])
    from cfm_mol.chemical_moves import covalent_radii
    radii_by_type=covalent_radii(metadata.atomic_numbers.tolist(),dtype=torch.float32,device=args.device)
    state=torch.load(str(args.warm_checkpoint),map_location='cpu',weights_only=False)
    warm=state.get('research_protocol',{})
    if warm.get('position_backbone','flowmol')!='flowmol':raise ValueError('Electronic adapter requires the FlowMol backbone')
    model=model_from_config(cfg);prepare_research_backbone(model,warm)
    if warm.get('electronic_conditioning',False):
        if not args.global_state:raise ValueError('Cannot silently remove a trained electronic adapter')
        patch_electronic_conditioning(model)
    model.load_state_dict(state['state_dict'],strict=True)
    if args.global_state:patch_electronic_conditioning(model)
    patch_smooth_geometry(model,warm.get('geometry_softening',0.));model.to(args.device).float().train()
    terminal=warm.get('data_endpoint_time',1.);head=warm.get('position_parameterization','displacement')
    dataset=MoleculeDataset('train',dict(cfg['dataset'],fake_atom_p=0.,fake_atom_std=1.,
        explicit_aromaticity=cfg['mol_fm'].get('explicit_aromaticity',False)),prior_config=cfg['mol_fm']['prior_config'])
    metadata.verify_processed_file(dataset.processed_data_dir/'train_data_processed.pt')
    if len(dataset)!=len(metadata.indices):raise ValueError('Metadata does not cover the training dataset')
    order=torch.randperm(len(dataset),generator=torch.Generator().manual_seed(args.seed+1))
    if args.steps*args.batch_size>len(order):raise ValueError('This bounded run must not recycle the data order')
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-12)
    protocol={**warm,**{k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'format':'electronic_geometry_fm_v1','electronic_conditioning':bool(args.global_state),
        'electronic_metadata':str(args.metadata.resolve()),'metadata_progress_sha256':metadata.progress_sha256,
        'prior_std':args.prior_std,'position_parameterization':head,'data_endpoint_time':terminal,
        'warm_sha256':sha(args.warm_checkpoint),'config_sha256':sha(args.config),'script_sha256':sha(__file__),
        'data_order_sha256':hashlib.sha256(order.numpy().tobytes()).hexdigest(),
        'purpose':__doc__,'composition_prior_checkpoint':warm.get('composition_prior_checkpoint',str(args.warm_checkpoint.resolve()))}
    protocol.update(alignment=args.pairing in ['rotation','steric'],
        paired_Haar_augmentation=args.pairing!='none',
        pairing_protocol_sha256=None if args.pairing_protocol is None else sha(args.pairing_protocol),
        score_proxy_warning='For correlated FM pairings, the independent-Gaussian velocity-to-score formula is invalid. This pilot uses FM only.')
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    # Extra adapter initialization must not change dataset-side randomness
    # between the legacy and global-state arms. FM noise has its own generator.
    torch.manual_seed(args.seed+17)
    start=time.perf_counter()
    for step in range(args.steps):
        indices=order[step*args.batch_size:(step+1)*args.batch_size].tolist()
        graph=dgl.batch([dataset[index] for index in indices]).to(args.device)
        metadata.attach(graph,indices,args.requested_kT)
        nbi,_=get_batch_idxs(graph);uem=get_upper_edge_mask(graph)
        pairing_records=[]
        loss=clamped_fm_loss(model,graph,nbi,uem,terminal_time=terminal,parameterization=head,
            prior_std=args.prior_std,generator=torch.Generator(device=args.device).manual_seed(args.seed*1000003+step),
            pairing=None if args.pairing=='none' else args.pairing,
            pairing_radii=radii_by_type[graph.ndata['a_1_true'].argmax(-1)],
            pairing_generator=torch.Generator(device=args.device).manual_seed(args.seed*2000003+step),
            pairing_diagnostics=pairing_records)
        if not torch.isfinite(loss):raise FloatingPointError('Non-finite FM loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        row={'step':step+1,'fm_loss':float(loss.detach()),'gradient_norm':float(norm),
            'n_atoms':graph.num_nodes(),'seconds':time.perf_counter()-start,
            'processed_indices':indices,'pairing':pairing_records}
        with (args.out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if (step+1)%100==0 or step==0:print(json.dumps(row),flush=True)
        if (step+1)%1000==0 or step+1==args.steps:
            if not all(torch.isfinite(value).all() for value in model.state_dict().values() if value.is_floating_point()):
                raise FloatingPointError('Non-finite model parameters')
            temporary=args.out/'last.ckpt.tmp'
            torch.save({'state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
                        'global_step':step+1,'research_protocol':protocol},temporary)
            temporary.replace(args.out/'last.ckpt')
    summary={'complete':True,'global_step':args.steps,'all_weights_finite':True,'seconds':time.perf_counter()-start,
        'global_state':bool(args.global_state),'prior_std':args.prior_std,'scope':'metadata-aware FM baseline; no energy-advantage claim'}
    (args.out/'validation.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
