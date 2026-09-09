#!/usr/bin/env python3
"""Independent evaluation of a frozen trained stochastic proposal, without updates."""
import argparse
import json
import math
from pathlib import Path
import time

import dgl
import torch
from flowmol.model_utils.load import read_config_file,model_from_config
from flowmol.data_processing.utils import get_batch_idxs,get_upper_edge_mask

from cfm_mol.clamped_density import deterministic_field,position_velocity
from cfm_mol.condition_systems import graph_from_condition
from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.nonequilibrium import centered_orthonormal_basis,WeightedPaths,normalized_weights
from cfm_mol.path_work import gaussian_training_path
from cfm_mol.radial_reference import prepare_research_backbone
from cfm_mol.smooth_geometry import patch_smooth_geometry
from molecular_tempered_pilot import sha,write_json,geometry_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--particles',type=int,default=4096);p.add_argument('--batch',type=int,default=64)
    p.add_argument('--source-evaluation-stream',action='store_true',help='Reproduce a prefix of the saved evaluation stream for loader validation')
    p.add_argument('--seed',type=int,default=9065);p.add_argument('--device',default='cuda');args=p.parse_args()
    if min(args.particles,args.batch)<1 or args.particles%args.batch:raise ValueError('Positive divisible evaluation counts required')
    source=args.run/'results.json';old=json.loads(source.read_text())
    if not old['complete']:raise ValueError('Require completed training')
    checkpoint=args.run/'last.ckpt';state=torch.load(str(checkpoint),map_location='cpu',weights_only=False)
    recipe=state['proposal_protocol'];protocol=state['research_protocol']
    if recipe!=old['configuration']:raise ValueError('Checkpoint protocol differs from completed run')
    if args.source_evaluation_stream and (args.batch!=recipe['batch'] or args.particles>recipe['eval_particles']):
        raise ValueError('Source-stream reproduction requires its original batch and a prefix sample count')
    cfg=read_config_file(Path(recipe['config']));cfg['mol_fm'].pop('bgfm',None);cfg['mol_fm']['prior_config']['x']['align']=False
    if cfg['dataset']['max_atoms']!=200 or cfg['mol_fm']['total_loss_weights']['e']!=0:raise ValueError('Require bond-free max_atoms=200')
    if protocol['position_parameterization']!='displacement' or protocol['data_endpoint_time']!=1.:
        raise ValueError('Require the explicit T=1 displacement architecture')
    def load_model(key):
        model=model_from_config(cfg);prepare_research_backbone(model,protocol)
        model.load_state_dict(state[key],strict=True);patch_smooth_geometry(model,protocol.get('geometry_softening',0.))
        return model.to(args.device).float().eval()
    forward=load_model('forward_state_dict');backward=load_model('backward_state_dict');del state
    # The temperature reset, if used, is already baked into these trained
    # weights. Applying it a second time would silently change the proposal.
    condition=old['condition'];numbers=condition['numbers'];n=len(numbers);dimension=3*(n-1)
    base=graph_from_condition({'atomic_numbers':numbers,'charge':condition['charge'],'spin_multiplicity':condition['spin_multiplicity'],
        'requested_kT_eV':recipe['kT']},cfg['dataset']['atom_map'],n_bond_classes=5 if cfg['mol_fm'].get('explicit_aromaticity',False) else 4)
    graph=dgl.batch([base]*args.batch).to(args.device);nbi,_=get_batch_idxs(graph);uem=get_upper_edge_mask(graph)
    basis=centered_orthonormal_basis(n,device=args.device)
    def drift(model,z,t,sign):
        x=torch.einsum('nk,bkd->bnd',basis,z.reshape(args.batch,n-1,3)).reshape(args.batch*n,3).float()
        with graph.local_scope(),deterministic_field(model.vector_field),torch.autocast(device_type=x.device.type,enabled=False):
            for key in ['a','c']:graph.ndata[key+'_t']=graph.ndata[key+'_1_true']
            graph.edata['e_t']=graph.edata['e_1_true']
            value=position_velocity(model,graph,x,x.new_full((args.batch,),t),nbi,uem,parameterization='displacement')
        return sign*torch.einsum('nk,bnd->bkd',basis,value.double().reshape(args.batch,n,3)).reshape_as(z)
    args.out.mkdir(parents=True,exist_ok=True);output=args.out/'results.json'
    if output.exists():raise FileExistsError(output)
    root=Path(__file__).resolve().parents[2];oracle_path=Path(recipe['oracle'])
    if sha(oracle_path)!=old['oracle_sha256']:raise ValueError('Oracle checkpoint changed')
    report={'complete':False,'scope':__doc__,'condition':condition,'configuration':{**recipe,'steps':0,
        'batch':args.batch,'eval_particles':args.particles,'seed':recipe['seed'] if args.source_evaluation_stream else args.seed,'device':args.device},
        'source_training_steps':recipe['steps'],'source_training_result':str(source.resolve()),
        'source_training_result_sha256':sha(source),'trained_checkpoint_sha256':sha(checkpoint),
        'oracle_sha256':old['oracle_sha256'],'script_sha256':sha(Path(__file__)),
        'energy_zero_eV':old['energy_zero_eV'],'reference_geometry_loaded':False,
        'temperature_reset_reapplied':False,'evaluation_seed_rule':'900000001 + source_seed*1009 + batch_index' if args.source_evaluation_stream else 'seed + 100003*batch_index',
        'source_evaluation_stream':args.source_evaluation_stream,
        'evaluations':{},'training':[]}
    write_json(output,report);positions=[];energies=[];works=[];start=time.perf_counter()
    terminal_std=math.sqrt(recipe['kT']/recipe['restraint']) if recipe['reference_kernel']=='gaussian' else None
    with EnergyOracle(Path(recipe['oracle_python']),root/'scripts/research/oracle_worker.py',oracle_path,
        numbers=numbers,charge=condition['charge'],spin_multiplicity=condition['spin_multiplicity'],batch_size=16) as oracle,torch.no_grad():
        for index in range(args.particles//args.batch):
            draw_seed=900000001+recipe['seed']*1009+index if args.source_evaluation_stream else args.seed+100003*index
            generator=torch.Generator(device=args.device).manual_seed(draw_seed)
            x0=torch.randn((args.batch,dimension),device=args.device,dtype=torch.float64,generator=generator)*old['prior_std']
            path=gaussian_training_path(x0,lambda z,t:drift(forward,z,t,1.),lambda z,t:drift(backward,z,t,-1.),
                torch.linspace(0,1,recipe['path_steps']+1,dtype=torch.float64),recipe['noise'],generator,
                prior_std=old['prior_std'],terminal_std=terminal_std,max_drift_norm=recipe['max_drift_per_sqrt_dimension']*math.sqrt(dimension),
                mean_parameterization=recipe.get('mean_parameterization','reference'),noise_annealing_power=recipe.get('noise_annealing_power',0.))
            x=torch.einsum('nk,bkd->bnd',basis,path.terminal.reshape(args.batch,n-1,3));energy,_=oracle.evaluate(x)
            reduced=(energy.to(x)-old['energy_zero_eV']+recipe['restraint']/2*x.square().sum((1,2)))/recipe['kT']
            positions.append(x.cpu());energies.append(energy);works.append(path.work(reduced).cpu())
            if (index+1)%8==0:print(json.dumps({'particles':(index+1)*args.batch,'seconds':time.perf_counter()-start}),flush=True)
        x=torch.cat(positions);energy=torch.cat(energies);work=torch.cat(works);weights=normalized_weights(-work)
        result={'mean_shifted_work':float(work.mean()),'work_std':float(work.std()),'mean_energy_eV':float(energy.mean()),
            'weights':WeightedPaths(x,-work,{}).summary(),'geometry':geometry_metrics(x,torch.full((len(x),),1/len(x),dtype=torch.float64),numbers),
            'weighted_geometry':geometry_metrics(x,weights,numbers),'oracle_evaluations':oracle.evaluated}
        torch.save({'positions':x,'energy_eV':energy,'work':work,'condition':condition},args.out/'final_samples.pt')
        if args.source_evaluation_stream:
            original=torch.load(str(args.run/'final_samples.pt'),map_location='cpu',weights_only=False)
            comparison={'max_position_error_A':float((x-original['positions'][:len(x)]).abs().max()),
                'max_work_error':float((work-original['work'][:len(x)]).abs().max()),
                'max_energy_error_eV':float((energy-original['energy_eV'][:len(x)]).abs().max())}
            report['source_stream_comparison']=comparison;write_json(output,report)
            if comparison['max_position_error_A']>1e-4 or comparison['max_work_error']>.02:
                raise ValueError('Frozen loader failed source-stream numerical agreement check')
        report['evaluations']['final']=result;report.update(complete=True,oracle_evaluations=oracle.evaluated,seconds=time.perf_counter()-start)
        write_json(output,report);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
