#!/usr/bin/env python3
"""Refit only an auxiliary reverse model on frozen forward paths; endpoints never change."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from flowmol.model_utils.load import read_config_file,model_from_config

from cfm_mol.condition_systems import graph_from_condition
from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.molecular_path_drift import MolecularPathDrift
from cfm_mol.nonequilibrium import centered_orthonormal_basis,WeightedPaths
from cfm_mol.path_work import gaussian_training_path
from cfm_mol.path_balance import fixed_backward_residuals,backward_log_probability,fixed_path_log_factors
from cfm_mol.radial_reference import prepare_research_backbone
from cfm_mol.smooth_geometry import patch_smooth_geometry
from molecular_tempered_pilot import sha,write_json


def tensor_hash(value):return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=500)
    p.add_argument('--train-paths',type=int,default=2048);p.add_argument('--eval-paths',type=int,default=256)
    p.add_argument('--batch',type=int,default=16);p.add_argument('--generation-batch',type=int,default=64)
    p.add_argument('--lr',type=float,default=1e-5);p.add_argument('--seed',type=int,default=9081)
    p.add_argument('--device',default='cuda');args=p.parse_args()
    if min(args.steps,args.train_paths,args.eval_paths,args.batch,args.generation_batch)<1:raise ValueError('Positive counts required')
    if any(n%args.generation_batch for n in [args.train_paths,args.eval_paths]):raise ValueError('Cache counts must divide generation batch')
    if args.train_paths%args.batch or args.eval_paths%args.batch:raise ValueError('Cache counts must divide refit batch')
    if not math.isfinite(args.lr) or args.lr<=0:raise ValueError('Positive learning rate required')
    source=args.source_run/'results.json';old=json.loads(source.read_text())
    if not old['complete']:raise ValueError('Source training must be completed')
    checkpoint=args.source_run/'last.ckpt';state=torch.load(str(checkpoint),map_location='cpu',weights_only=False)
    recipe=state['proposal_protocol'];protocol=state['research_protocol']
    if recipe!=old['configuration'] or recipe.get('precision_kind','none')!='none':raise ValueError('This diagnostic requires a verified scalar-noise source')
    if protocol['position_parameterization']!='displacement' or protocol['data_endpoint_time']!=1.:raise ValueError('Require T=1 displacement model')
    output=args.out/'results.json'
    if output.exists():raise FileExistsError(output)
    args.out.mkdir(parents=True,exist_ok=True);start=time.perf_counter();torch.manual_seed(args.seed)
    root=Path(__file__).resolve().parents[2]
    cfg=read_config_file(Path(recipe['config']));cfg['mol_fm'].pop('bgfm',None);cfg['mol_fm']['prior_config']['x']['align']=False
    if cfg['dataset']['max_atoms']!=200 or cfg['mol_fm']['total_loss_weights']['e']!=0:raise ValueError('Require bond-free max_atoms200')
    def load(key):
        model=model_from_config(cfg);prepare_research_backbone(model,protocol);model.load_state_dict(state[key],strict=True)
        patch_smooth_geometry(model,protocol.get('geometry_softening',0.));return model.to(args.device).float().eval()
    forward=load('forward_state_dict');backward=load('backward_state_dict');del state
    for parameter in forward.parameters():parameter.requires_grad_(False)
    frozen={key:value.detach().cpu().clone() for key,value in forward.state_dict().items()}
    condition=old['condition'];numbers=condition['numbers'];n=len(numbers);dimension=3*(n-1)
    base=graph_from_condition({'atomic_numbers':numbers,'charge':condition['charge'],'spin_multiplicity':condition['spin_multiplicity'],
        'requested_kT_eV':recipe['kT']},cfg['dataset']['atom_map'],n_bond_classes=5 if cfg['mol_fm'].get('explicit_aromaticity',False) else 4).to(args.device)
    field_f=MolecularPathDrift(forward,base);field_b=MolecularPathDrift(backward,base,sign=-1.)
    basis=centered_orthonormal_basis(n,device=args.device);times=torch.linspace(0,1,recipe['path_steps']+1,dtype=torch.float64)
    settings={'prior_std':old['prior_std'],'terminal_std':math.sqrt(recipe['kT']/recipe['restraint']) if recipe['reference_kernel']=='gaussian' else None,
        'max_drift_norm':recipe['max_drift_per_sqrt_dimension']*math.sqrt(dimension),
        'mean_parameterization':recipe.get('mean_parameterization','reference'),'noise_annealing_power':recipe.get('noise_annealing_power',0.)}
    report={'complete':False,'format':'frozen_forward_backward_refit_v1','scope':__doc__,
        'configuration':{k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'condition':condition,'source_results_sha256':sha(source),'source_checkpoint_sha256':sha(checkpoint),
        'source_protocol':recipe,'code_sha256':{str(p):sha(p) for p in [Path(__file__),root/'cfm_mol/path_balance.py',root/'cfm_mol/path_work.py',root/'cfm_mol/molecular_path_drift.py']},'forward_parameters_frozen':True,'temperature_reset_reapplied':False,
        'variance_ratio_interval':[.25,1.9],'train_seed':args.seed,'eval_seed':args.seed+1,'minibatch_seed':args.seed+2,
        'history':[],'evaluations':{},'limitations':['One fixed forward proposal; this cannot improve unweighted generated geometry.',
            'Refitting the same backward mean family does not test every possible reverse conditional.',
            'Optional scalar width fitting uses training paths only and stays below twice reference variance.',
            'Weight ESS and backward likelihood do not certify global target coverage.']}
    write_json(output,report)
    def generate(count,seed):
        states=[];logs=[]
        with torch.no_grad():
            for index in range(count//args.generation_batch):
                g=torch.Generator(device=args.device).manual_seed(seed+100003*index)
                x0=torch.randn((args.generation_batch,dimension),dtype=torch.float64,device=args.device,generator=g)*old['prior_std']
                zero=lambda z,t:torch.zeros_like(z)
                path=gaussian_training_path(x0,field_f,zero,times,recipe['noise'],g,retain_states=True,**settings)
                residual,norm=fixed_backward_residuals(path.states,zero,times,recipe['noise'],**settings)
                logq=path.log_initial+path.log_forward_minus_backward+backward_log_probability(residual,norm,dimension)
                states.append(path.states.cpu());logs.append(logq.cpu())
        return {'states':torch.cat(states),'log_forward_path':torch.cat(logs)}
    training=generate(args.train_paths,args.seed);heldout=generate(args.eval_paths,args.seed+1)
    check_count=min(args.batch,args.train_paths)
    with torch.no_grad():
        initial,logf,_=fixed_path_log_factors(training['states'][:check_count].to(args.device),field_f,lambda z,t:torch.zeros_like(z),times,recipe['noise'],**settings)
        discrepancy=float((initial.cpu()+logf.cpu()-training['log_forward_path'][:check_count]).abs().max())
    if discrepancy>.05:raise RuntimeError('Cached forward path density disagrees with independent rescoring')
    report['forward_density_cache_check']={'paths':check_count,'max_difference':discrepancy,'extra_field_evaluations':check_count*recipe['path_steps']}
    for name,data in [('training',training),('heldout',heldout)]:
        torch.save(data,args.out/f'{name}_paths.pt');report[name+'_paths_sha256']=sha(args.out/f'{name}_paths.pt')
    positions=torch.einsum('nk,bkd->bnd',basis.cpu(),heldout['states'][:,-1].reshape(args.eval_paths,n-1,3))
    endpoint_hash=tensor_hash(positions);oracle_path=Path(recipe['oracle']);root=Path(__file__).resolve().parents[2]
    if sha(oracle_path)!=old['oracle_sha256']:raise ValueError('Oracle changed')
    with EnergyOracle(Path(recipe['oracle_python']),root/'scripts/research/oracle_worker.py',oracle_path,
        numbers=numbers,charge=condition['charge'],spin_multiplicity=condition['spin_multiplicity'],batch_size=16) as oracle:
        energy,_=oracle.evaluate(positions);report['oracle_evaluations']=oracle.evaluated
    reduced=(energy-old['energy_zero_eV']+recipe['restraint']/2*positions.square().sum((1,2)))/recipe['kT']
    saved_evaluations={}
    def residuals(data):
        chunks=[];normalizations=[]
        with torch.no_grad():
            for begin in range(0,len(data['states']),args.batch):
                residual,norm=fixed_backward_residuals(data['states'][begin:begin+args.batch].to(args.device),field_b,times,recipe['noise'],**settings)
                chunks.append(residual.cpu());normalizations.append(norm.cpu())
        return torch.cat(chunks),torch.cat(normalizations)
    def evaluate(stage):
        tr,tn=residuals(training);er,en=residuals(heldout)
        raw_ratios=tr.mean(0)/dimension;ratios=raw_ratios.clamp(.25,1.9)
        for suffix,scales in [('original_width',None),('fitted_width',ratios)]:
            logb=backward_log_probability(er,en,dimension,scales)
            work=reduced+heldout['log_forward_path']-logb
            row={'mean_work':float(work.mean()),'work_std':float(work.std()),'mean_backward_nll':float(-logb.mean()),
                'weights':WeightedPaths(positions,-work,{}).summary(),
                'variance_ratios':None if scales is None else scales.tolist(),'unclipped_training_variance_ratios':raw_ratios.tolist(),
                'endpoint_tensor_sha256':endpoint_hash}
            label=stage+'_'+suffix;report['evaluations'][label]=row;saved_evaluations[label]=work
            print(json.dumps({'evaluation':label,**row}),flush=True)
        write_json(output,report)
    evaluate('before')
    params=[p for p in backward.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(params,lr=args.lr,weight_decay=0.)
    selection=torch.Generator().manual_seed(args.seed+2)
    for step in range(args.steps):
        indices=torch.randint(args.train_paths,(args.batch,),generator=selection)
        optimizer.zero_grad(set_to_none=True)
        residual,norm=fixed_backward_residuals(training['states'][indices].to(args.device),field_b,times,recipe['noise'],**settings)
        loss=-backward_log_probability(residual,norm,dimension).mean();loss.backward()
        gradient=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step()
        row={'step':step+1,'backward_nll':float(loss.detach()),'gradient_norm':float(gradient),'seconds':time.perf_counter()-start}
        report['history'].append(row)
        if (step+1)%25==0 or step==0:write_json(output,report);print(json.dumps(row),flush=True)
    evaluate('after')
    if not all(torch.equal(value.cpu(),frozen[key]) for key,value in forward.state_dict().items()):raise RuntimeError('Frozen forward state changed')
    if tensor_hash(positions)!=endpoint_hash:raise RuntimeError('Heldout coordinates changed')
    torch.save({'positions':positions,'energy_eV':energy,'works':saved_evaluations,'condition':condition},args.out/'heldout_comparison.pt')
    torch.save({'backward_state_dict':backward.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
        'source_checkpoint_sha256':sha(checkpoint),'refit_steps':args.steps},args.out/'backward_refit.ckpt')
    report.update(complete=True,forward_state_unchanged=True,heldout_positions_unchanged=True,seconds=time.perf_counter()-start,
        comparison_sha256=sha(args.out/'heldout_comparison.pt'),backward_checkpoint_sha256=sha(args.out/'backward_refit.ckpt'),
        forward_generation_field_evaluations=(args.train_paths+args.eval_paths)*recipe['path_steps'])
    write_json(output,report)


if __name__=='__main__':main()
