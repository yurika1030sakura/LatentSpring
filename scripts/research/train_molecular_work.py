#!/usr/bin/env python3
"""Train a stochastic proposal by mean finite-path work, initialized from FM.

This is an existing path-space KL/SNF teacher objective. The eventual position
student remains a separately evaluated flow-matching model. Work is shifted by
a fixed source energy for logging; no normalizer is assumed known.
"""
import argparse
import json
import math
from pathlib import Path
import time

import dgl
import numpy as np
import torch
from ase.data import atomic_numbers
from flowmol.model_utils.load import read_config_file,model_from_config
from flowmol.data_processing.dataset import MoleculeDataset
from flowmol.data_processing.utils import get_batch_idxs,get_upper_edge_mask

from cfm_mol.clamped_density import deterministic_field,position_velocity
from cfm_mol.electronic_metadata import ElectronicMetadata
from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.nonequilibrium import centered_orthonormal_basis,normalized_weights,WeightedPaths
from cfm_mol.path_work import gaussian_training_path,external_energy
from cfm_mol.radial_reference import prepare_research_backbone
from cfm_mol.smooth_geometry import patch_smooth_geometry
from molecular_tempered_pilot import sha,write_json,geometry_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--metadata',type=Path,required=True);p.add_argument('--oracle',type=Path,required=True)
    p.add_argument('--oracle-python',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source-row',type=int,default=5846);p.add_argument('--steps',type=int,default=100)
    p.add_argument('--path-steps',type=int,default=16);p.add_argument('--batch',type=int,default=2)
    p.add_argument('--eval-particles',type=int,default=64);p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--noise',type=float,default=.2);p.add_argument('--kT',type=float,default=1.)
    p.add_argument('--restraint',type=float,default=.1);p.add_argument('--seed',type=int,default=9051)
    p.add_argument('--mode',choices=['joint','backward_only'],default='joint')
    p.add_argument('--checkpoint-steps',action='store_true');p.add_argument('--device',default='cuda')
    args=p.parse_args()
    if min(args.steps,args.path_steps,args.batch,args.eval_particles)<1 or args.eval_particles%args.batch:
        raise ValueError('Positive counts and evaluation divisible by batch are required')
    if any(not math.isfinite(v) or v<=0 for v in [args.lr,args.noise,args.kT,args.restraint]):raise ValueError('Invalid scale')
    args.out.mkdir(parents=True,exist_ok=True);output=args.out/'results.json'
    if output.exists():raise FileExistsError(output)
    torch.manual_seed(args.seed)
    cfg=read_config_file(args.config);cfg['mol_fm'].pop('bgfm',None);cfg['mol_fm']['prior_config']['x']['align']=False
    if cfg['dataset']['max_atoms']!=200 or cfg['mol_fm']['total_loss_weights']['e']!=0:raise ValueError('Require bond-free max_atoms=200')
    metadata=ElectronicMetadata(args.metadata,'val',[atomic_numbers[s] for s in cfg['dataset']['atom_map']])
    dataset=MoleculeDataset('val',dict(cfg['dataset'],fake_atom_p=0.,fake_atom_std=1.,
        explicit_aromaticity=cfg['mol_fm'].get('explicit_aromaticity',False)),prior_config=cfg['mol_fm']['prior_config'])
    metadata.verify_processed_file(dataset.processed_data_dir/'val_data_processed.pt')
    base=dataset[args.source_row];n=base.num_nodes();dimension=3*(n-1)
    if not 2<=n<=200:raise ValueError('Invalid atom count')
    accepted=int(metadata.indices[args.source_row]);charge=int(metadata.values['total_charge'][accepted]);spin=int(metadata.values['spin_multiplicity'][accepted])
    energy_zero=float(metadata.values['energy_float64'][accepted])
    numbers=np.array([atomic_numbers[cfg['dataset']['atom_map'][int(i)]] for i in base.ndata['a_1_true'].argmax(-1)])
    state=torch.load(str(args.checkpoint),map_location='cpu',weights_only=False);protocol=state.get('research_protocol',{})
    if protocol.get('position_parameterization')!='displacement' or protocol.get('data_endpoint_time')!=1.:
        raise ValueError('This proposal trainer requires the explicit T=1 displacement checkpoint')
    def load_model():
        value=model_from_config(cfg);prepare_research_backbone(value,protocol)
        value.load_state_dict(state['state_dict'],strict=True);patch_smooth_geometry(value,protocol.get('geometry_softening',0.))
        return value.to(args.device).float().train()
    forward=load_model();backward=load_model();del state
    if args.mode=='backward_only':
        for parameter in forward.parameters():parameter.requires_grad_(False)
    parameters=[v for model in [forward,backward] for v in model.parameters() if v.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=args.lr,weight_decay=0.)
    graph=dgl.batch([base]*args.batch).to(args.device);metadata.attach(graph,[args.source_row]*args.batch,args.kT)
    nbi,_=get_batch_idxs(graph);uem=get_upper_edge_mask(graph)
    basis=centered_orthonormal_basis(n,device=args.device);prior_std=math.sqrt(args.kT/args.restraint)
    def drift(model,z,t,sign):
        x=torch.einsum('nk,bkd->bnd',basis,z.reshape(args.batch,n-1,3)).reshape(args.batch*n,3).float()
        # Re-enter deterministic mode in checkpoint recomputation as well.
        with graph.local_scope(),deterministic_field(model.vector_field),torch.autocast(device_type=x.device.type,enabled=False):
            for key in ['a','c']:graph.ndata[key+'_t']=graph.ndata[key+'_1_true']
            graph.edata['e_t']=graph.edata['e_1_true']
            v=position_velocity(model,graph,x,x.new_full((args.batch,),t),nbi,uem,parameterization='displacement')
        return sign*torch.einsum('nk,bnd->bkd',basis,v.double().reshape(args.batch,n,3)).reshape_as(z)
    root=Path(__file__).resolve().parents[2]
    report={'complete':False,'scope':__doc__,'configuration':{k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'checkpoint_sha256':sha(args.checkpoint),'oracle_sha256':sha(args.oracle),'metadata_progress_sha256':metadata.progress_sha256,
        'source_sha256':{str(path):sha(path) for path in [Path(__file__).resolve(),root/'cfm_mol/path_work.py',root/'cfm_mol/energy_oracle.py']},
        'condition':{'source_row':args.source_row,'raw_index':int(metadata.values['raw_indices'][accepted]),
            'numbers':numbers.tolist(),'charge':charge,'spin_multiplicity':spin},
        'energy_zero_eV':energy_zero,'prior_std':prior_std,'evaluations':{},'training':[]}
    write_json(output,report);start=time.perf_counter()
    with EnergyOracle(args.oracle_python,root/'scripts/research/oracle_worker.py',args.oracle,numbers=numbers,charge=charge,spin_multiplicity=spin) as oracle:
        def draw(seed,training):
            g=torch.Generator(device=args.device).manual_seed(seed)
            x0=torch.randn((args.batch,dimension),device=args.device,dtype=torch.float64,generator=g)*prior_std
            path=gaussian_training_path(x0,lambda z,t:drift(forward,z,t,1.),lambda z,t:drift(backward,z,t,-1.),
                torch.linspace(0,1,args.path_steps+1,dtype=torch.float64),args.noise,g,prior_std=prior_std,
                checkpoint_steps=args.checkpoint_steps and training)
            positions=torch.einsum('nk,bkd->bnd',basis,path.terminal.reshape(args.batch,n-1,3))
            energy,force=oracle.evaluate(positions)
            linked=external_energy(positions,energy,force) if training else energy.to(positions)
            reduced=(linked-energy_zero+args.restraint/2*positions.square().sum((1,2)))/args.kT
            return path.work(reduced),positions,energy
        def evaluate(label):
            before=oracle.evaluated;works=[];positions=[];energies=[]
            with torch.no_grad():
                for batch in range(args.eval_particles//args.batch):
                    w,x,e=draw(900000001+args.seed*1009+batch,False)
                    works.append(w.cpu());positions.append(x.cpu());energies.append(e)
            work=torch.cat(works);x=torch.cat(positions);energy=torch.cat(energies)
            weights=normalized_weights(-work)
            result={'mean_shifted_work':float(work.mean()),'work_std':float(work.std()),
                'mean_energy_eV':float(energy.mean()),'weights':WeightedPaths(x,-work,{}).summary(),
                'geometry':geometry_metrics(x,torch.full((len(x),),1/len(x),dtype=torch.float64),numbers),
                'weighted_geometry':geometry_metrics(x,weights,numbers),'oracle_evaluations':oracle.evaluated-before}
            torch.save({'positions':x,'energy_eV':energy,'work':work,'condition':report['condition']},args.out/f'{label}_samples.pt')
            report['evaluations'][label]=result;write_json(output,report);print(json.dumps({'evaluation':label,**result}),flush=True)
        evaluate('initial')
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            work,_,energy=draw(500000003+args.seed*1009+step,True);loss=work.mean();loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);optimizer.step()
            row={'step':step+1,'mean_shifted_work':float(loss.detach()),'mean_energy_eV':float(energy.mean()),
                 'gradient_norm':float(norm),'seconds':time.perf_counter()-start}
            report['training'].append(row)
            if (step+1)%10==0 or step==0:write_json(output,report);print(json.dumps(row),flush=True)
        if not all(torch.isfinite(p).all() for p in parameters):raise FloatingPointError('Non-finite proposal parameters')
        evaluate('final')
        torch.save({'forward_state_dict':forward.state_dict(),'backward_state_dict':backward.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),'research_protocol':protocol,'proposal_protocol':report['configuration'],
            'global_step':args.steps,'prior_std':prior_std},args.out/'last.ckpt')
        report.update(complete=True,oracle_evaluations=oracle.evaluated,seconds=time.perf_counter()-start,
            peak_gpu_bytes=torch.cuda.max_memory_allocated() if str(args.device).startswith('cuda') else None)
        write_json(output,report)


if __name__=='__main__':main()
