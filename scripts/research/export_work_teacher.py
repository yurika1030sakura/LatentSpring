#!/usr/bin/env python3
"""Independent work-weighted endpoint pool for a controlled forward-FM diagnostic."""
import argparse
import json
import math
from pathlib import Path
import time

import torch
from flowmol.model_utils.load import read_config_file,model_from_config

from cfm_mol.condition_systems import graph_from_condition
from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.importance_curriculum import weight_controls
from cfm_mol.molecular_path_drift import MolecularPathDrift
from cfm_mol.nonequilibrium import centered_orthonormal_basis
from cfm_mol.path_balance import fixed_backward_residuals,backward_log_probability,fixed_path_log_factors
from cfm_mol.path_work import gaussian_training_path
from cfm_mol.radial_reference import prepare_research_backbone
from cfm_mol.smooth_geometry import patch_smooth_geometry
from molecular_tempered_pilot import sha,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--backward-refit',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--particles',type=int,default=4096);p.add_argument('--batch',type=int,default=64)
    p.add_argument('--seed',type=int,default=9084);p.add_argument('--device',default='cuda');args=p.parse_args()
    if min(args.particles,args.batch)<1 or args.particles%args.batch:raise ValueError('Positive divisible counts required')
    source=args.source_run/'results.json';old=json.loads(source.read_text());refit_path=args.backward_refit/'results.json';refit=json.loads(refit_path.read_text())
    checkpoint=args.source_run/'last.ckpt';refit_checkpoint=args.backward_refit/'backward_refit.ckpt'
    if not old['complete'] or not refit['complete'] or not refit['forward_state_unchanged']:raise ValueError('Require complete checked source/refit')
    if refit['source_checkpoint_sha256']!=sha(checkpoint) or refit['backward_checkpoint_sha256']!=sha(refit_checkpoint):raise ValueError('Source checkpoint mismatch')
    state=torch.load(str(checkpoint),map_location='cpu',weights_only=False);recipe=state['proposal_protocol'];protocol=state['research_protocol']
    if recipe!=old['configuration'] or recipe.get('precision_kind','none')!='none' or recipe.get('reference_kernel','gaussian')!='gaussian':raise ValueError('Require scalar Gaussian source')
    cfg=read_config_file(Path(recipe['config']));cfg['mol_fm'].pop('bgfm',None);cfg['mol_fm']['prior_config']['x']['align']=False
    if cfg['dataset']['max_atoms']!=200 or cfg['mol_fm']['total_loss_weights']['e']!=0:raise ValueError('Require bond-free max_atoms200')
    if protocol['position_parameterization']!='displacement' or protocol['data_endpoint_time']!=1.:raise ValueError('Require T1 displacement')
    output=args.out/'results.json'
    if output.exists():raise FileExistsError(output)
    args.out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    def load(weights):
        model=model_from_config(cfg);prepare_research_backbone(model,protocol);model.load_state_dict(weights,strict=True)
        patch_smooth_geometry(model,protocol.get('geometry_softening',0.));return model.to(args.device).float().eval()
    forward=load(state['forward_state_dict']);backward=load(torch.load(str(refit_checkpoint),map_location='cpu',weights_only=False)['backward_state_dict']);del state
    condition=old['condition'];numbers=condition['numbers'];n=len(numbers);dimension=3*(n-1)
    base=graph_from_condition({'atomic_numbers':numbers,'charge':condition['charge'],'spin_multiplicity':condition['spin_multiplicity'],
        'requested_kT_eV':recipe['kT']},cfg['dataset']['atom_map'],n_bond_classes=5 if cfg['mol_fm'].get('explicit_aromaticity',False) else 4).to(args.device)
    field_f=MolecularPathDrift(forward,base);field_b=MolecularPathDrift(backward,base,sign=-1.)
    basis=centered_orthonormal_basis(n,device=args.device);times=torch.linspace(0,1,recipe['path_steps']+1,dtype=torch.float64)
    settings={'prior_std':old['prior_std'],'terminal_std':math.sqrt(recipe['kT']/recipe['restraint']),
        'max_drift_norm':recipe['max_drift_per_sqrt_dimension']*math.sqrt(dimension),
        'mean_parameterization':recipe['mean_parameterization'],'noise_annealing_power':recipe['noise_annealing_power']}
    ratios=torch.tensor(refit['evaluations']['after_fitted_width']['variance_ratios'],dtype=torch.float64,device=args.device)
    if ratios.shape!=(recipe['path_steps'],) or (ratios<=0).any() or (ratios>=2).any():raise ValueError('Invalid backward widths')
    root=Path(__file__).resolve().parents[2];oracle_path=Path(recipe['oracle'])
    if sha(oracle_path)!=old['oracle_sha256']:raise ValueError('Oracle changed')
    report={'complete':False,'scope':__doc__,'condition':condition,'configuration':{k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'source_recipe':recipe,'research_protocol':protocol,'source_checkpoint_sha256':sha(checkpoint),'source_results_sha256':sha(source),
        'backward_refit_sha256':sha(refit_path),'backward_checkpoint_sha256':sha(refit_checkpoint),'oracle_sha256':sha(oracle_path),
        'energy_zero_eV':old['energy_zero_eV'],'backward_variance_ratios':ratios.tolist(),'temperature_reset_reapplied':False,
        'limitations':['Work weights are stochastic path weights, not an endpoint likelihood.',
            'Stabilized training weights are intermediate objectives, not a qualified final Boltzmann ensemble.',
            'No forward student has been trained or evaluated by this export.']}
    write_json(output,report);states=[];energies=[];logqs=[];logbs=[]
    with EnergyOracle(Path(recipe['oracle_python']),root/'scripts/research/oracle_worker.py',oracle_path,numbers=numbers,
        charge=condition['charge'],spin_multiplicity=condition['spin_multiplicity'],batch_size=16) as oracle,torch.no_grad():
        for batch in range(args.particles//args.batch):
            generator=torch.Generator(device=args.device).manual_seed(args.seed+100003*batch)
            x0=torch.randn((args.batch,dimension),dtype=torch.float64,device=args.device,generator=generator)*old['prior_std']
            zero=lambda z,t:torch.zeros_like(z)
            path=gaussian_training_path(x0,field_f,zero,times,recipe['noise'],generator,retain_states=True,**settings)
            residual,norm=fixed_backward_residuals(path.states,zero,times,recipe['noise'],**settings)
            logq=path.log_initial+path.log_forward_minus_backward+backward_log_probability(residual,norm,dimension)
            parts=[]
            for begin in range(0,args.batch,16):
                r,c=fixed_backward_residuals(path.states[begin:begin+16],field_b,times,recipe['noise'],**settings)
                parts.append(backward_log_probability(r,c,dimension,ratios))
            logb=torch.cat(parts)
            if batch==0:
                initial,lf,_=fixed_path_log_factors(path.states[:16],field_f,zero,times,recipe['noise'],**settings)
                error=float((initial+lf-logq[:16]).abs().max());report['forward_density_check_max_error']=error
                if error>.05:raise ValueError('Forward cache density mismatch')
            x=torch.einsum('nk,bkd->bnd',basis,path.terminal.reshape(args.batch,n-1,3));energy,_=oracle.evaluate(x)
            states.append(path.states.cpu());energies.append(energy);logqs.append(logq.cpu());logbs.append(logb.cpu())
            if (batch+1)%8==0:print(json.dumps({'particles':(batch+1)*args.batch,'seconds':time.perf_counter()-start}),flush=True)
        states=torch.cat(states);energy=torch.cat(energies);logq=torch.cat(logqs);logb=torch.cat(logbs)
        positions=torch.einsum('nk,bkd->bnd',basis.cpu(),states[:,-1].reshape(args.particles,n-1,3))
        work=(energy-old['energy_zero_eV']+recipe['restraint']/2*positions.square().sum((1,2)))/recipe['kT']+logq-logb
        weights,controls=weight_controls(-work,.5)
        torch.save({'positions':positions,'energy_eV':energy,'work':work,'states':states,'log_forward_path':logq,
            'log_backward_path':logb,'training_weights':weights,'condition':condition},args.out/'teacher.pt')
        report.update(complete=True,oracle_evaluations=oracle.evaluated,controls=controls,mean_energy_eV=float(energy.mean()),
            mean_work=float(work.mean()),work_std=float(work.std()),teacher_sha256=sha(args.out/'teacher.pt'),seconds=time.perf_counter()-start)
        write_json(output,report);print(json.dumps(controls),flush=True)


if __name__=='__main__':main()
