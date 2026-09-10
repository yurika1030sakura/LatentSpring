#!/usr/bin/env python3
"""Train an FM-initialized stochastic proposal with existing path-space objectives.

Mean work uses pathwise KL/SNF gradients; log variance scores detached fresh
paths and holds their observation measure fixed within each update. Both are
prior-art teacher objectives. The eventual position
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
from cfm_mol.condition_systems import load_condition,graph_from_condition
from cfm_mol.electronic_metadata import ElectronicMetadata
from cfm_mol.electronic_conditioning import neutralize_constant_temperature_input
from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.nonequilibrium import centered_orthonormal_basis,normalized_weights,WeightedPaths
from cfm_mol.path_work import gaussian_training_path,external_energy
from cfm_mol.path_balance import fixed_path_log_factors,log_variance_balance
from cfm_mol.molecular_path_drift import MolecularPathDrift
from cfm_mol.work_resume import validate_resume_recipe,restore_work_optimizer
from cfm_mol.pair_precision import LearnedPairPrecision
from cfm_mol.radial_reference import prepare_research_backbone
from cfm_mol.smooth_geometry import patch_smooth_geometry
from molecular_tempered_pilot import sha,write_json,geometry_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--resume-work-run',type=Path,help='Completed run to continue; --steps specifies the new total update count')
    p.add_argument('--metadata',type=Path);p.add_argument('--oracle',type=Path,required=True)
    p.add_argument('--condition-manifest',type=Path);p.add_argument('--condition-index',type=int)
    p.add_argument('--oracle-python',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source-row',type=int,default=5846);p.add_argument('--steps',type=int,default=100)
    p.add_argument('--path-steps',type=int,default=16);p.add_argument('--batch',type=int,default=2)
    p.add_argument('--eval-particles',type=int,default=64);p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--noise',type=float,default=.2);p.add_argument('--kT',type=float,default=1.)
    p.add_argument('--restraint',type=float,default=.1);p.add_argument('--seed',type=int,default=9051)
    p.add_argument('--precision-kind',choices=['none','learned','fixed','trace'],default='none')
    p.add_argument('--precision-strength',type=float,default=32.)
    p.add_argument('--precision-lr',type=float,default=1e-3)
    p.add_argument('--gradient-diagnostics',type=int,default=0,help='Freeze weights and compare paired gradient estimators for this many independent batches')
    p.add_argument('--objective',choices=['mean_work','log_variance'],default='mean_work')
    p.add_argument('--mode',choices=['joint','backward_only','forward_energy_only'],default='joint')
    p.add_argument('--oracle-batch-size',type=int,default=1)
    p.add_argument('--reference-kernel',choices=['euler','gaussian'],default='gaussian')
    p.add_argument('--mean-parameterization',choices=['reference','native'],default='reference')
    p.add_argument('--neutralize-temperature-input',action='store_true')
    p.add_argument('--noise-annealing-power',type=float,default=0.)
    p.add_argument('--prior-std',type=float,default=1.)
    p.add_argument('--max-drift-per-sqrt-dimension',type=float,default=20.)
    p.add_argument('--checkpoint-steps',action='store_true');p.add_argument('--device',default='cuda')
    args=p.parse_args()
    if min(args.steps,args.path_steps,args.batch,args.eval_particles)<1 or args.eval_particles%args.batch:
        raise ValueError('Positive counts and evaluation divisible by batch are required')
    if any(not math.isfinite(v) or v<=0 for v in [args.lr,args.noise,args.kT,args.restraint,args.prior_std,args.max_drift_per_sqrt_dimension]):raise ValueError('Invalid scale')
    if (args.condition_manifest is None)!=(args.condition_index is None):raise ValueError('Condition manifest and index must be specified together')
    if args.condition_manifest is None and args.metadata is None:raise ValueError('Legacy rows require verified electronic metadata')
    if args.gradient_diagnostics<0 or (args.gradient_diagnostics and (args.mode!='joint' or args.batch<2)):
        raise ValueError('Gradient diagnostics require joint mode and batch >= 2')
    if args.objective=='log_variance' and (args.mode!='joint' or args.batch<2):
        raise ValueError('Fixed-path log variance requires joint training and batch >= 2')
    if args.precision_kind!='none' and (args.mode!='joint' or args.objective!='mean_work' or args.gradient_diagnostics):
        raise ValueError('Pair precision currently requires joint mean-work training')
    if not math.isfinite(args.precision_strength) or args.precision_strength<0 or not math.isfinite(args.precision_lr) or args.precision_lr<=0:
        raise ValueError('Invalid precision hyperparameters')
    args.out.mkdir(parents=True,exist_ok=True);output=args.out/'results.json'
    if output.exists():raise FileExistsError(output)
    torch.manual_seed(args.seed)
    cfg=read_config_file(args.config);cfg['mol_fm'].pop('bgfm',None);cfg['mol_fm']['prior_config']['x']['align']=False
    if cfg['dataset']['max_atoms']!=200 or cfg['mol_fm']['total_loss_weights']['e']!=0:raise ValueError('Require bond-free max_atoms=200')
    metadata=None
    if args.condition_manifest is not None:
        # No reference-coordinate dataset is opened in this branch. This loader
        # refuses reserved conditions: training is development, even after freeze.
        condition=load_condition(args.condition_manifest,args.condition_index)
        condition['requested_kT_eV']=args.kT
        base=graph_from_condition(condition,cfg['dataset']['atom_map'],
            n_bond_classes=5 if cfg['mol_fm'].get('explicit_aromaticity',False) else 4)
        numbers=np.asarray(condition['atomic_numbers']);charge=condition['charge'];spin=condition['spin_multiplicity']
        energy_zero=float(condition['energy_eV'])
        source_condition={k:condition[k] for k in ['raw_index','manifest_index','manifest_sha256','manifest_role','source']}
        source_condition['reference_geometry_loaded']=False
    else:
        metadata=ElectronicMetadata(args.metadata,'val',[atomic_numbers[s] for s in cfg['dataset']['atom_map']])
        dataset=MoleculeDataset('val',dict(cfg['dataset'],fake_atom_p=0.,fake_atom_std=1.,
            explicit_aromaticity=cfg['mol_fm'].get('explicit_aromaticity',False)),prior_config=cfg['mol_fm']['prior_config'])
        metadata.verify_processed_file(dataset.processed_data_dir/'val_data_processed.pt')
        base=dataset[args.source_row]
        accepted=int(metadata.indices[args.source_row]);charge=int(metadata.values['total_charge'][accepted]);spin=int(metadata.values['spin_multiplicity'][accepted])
        energy_zero=float(metadata.values['energy_float64'][accepted])
        numbers=np.array([atomic_numbers[cfg['dataset']['atom_map'][int(i)]] for i in base.ndata['a_1_true'].argmax(-1)])
        source_condition={'source_row':args.source_row,'raw_index':int(metadata.values['raw_indices'][accepted]),
            'reference_geometry_loaded':True,'reference_geometry_used_to_initialize':False}
    if not math.isfinite(energy_zero):raise ValueError('Finite logging energy offset required')
    n=base.num_nodes();dimension=3*(n-1)
    if not 2<=n<=200:raise ValueError('Invalid atom count')
    state=torch.load(str(args.checkpoint),map_location='cpu',weights_only=False);protocol=state.get('research_protocol',{})
    if protocol.get('position_parameterization')!='displacement' or protocol.get('data_endpoint_time')!=1.:
        raise ValueError('This proposal trainer requires the explicit T=1 displacement checkpoint')
    if args.neutralize_temperature_input and (protocol.get('format')!='electronic_geometry_fm_v1' or not protocol.get('electronic_conditioning') or 'requested_kT' not in protocol):
        raise ValueError('Temperature initialization requires verified constant-input electronic FM training')
    resumed=None;resume_state=None;start_step=0
    if args.resume_work_run is not None:
        resumed=json.loads((args.resume_work_run/'results.json').read_text())
        resume_state=torch.load(str(args.resume_work_run/'last.ckpt'),map_location='cpu',weights_only=False)
        start_step=validate_resume_recipe(resumed,resume_state,vars(args))
        if resume_state['research_protocol']!=protocol:raise ValueError('Resume changes backbone architecture protocol')
        if resumed['checkpoint_sha256']!=sha(args.checkpoint) or resumed['oracle_sha256']!=sha(args.oracle):
            raise ValueError('Resume changes source FM checkpoint or potential')
        if sha(Path(resumed['configuration']['config']))!=sha(args.config):raise ValueError('Resume changes architecture config')
        expected={'numbers':numbers.tolist(),'charge':charge,'spin_multiplicity':spin,'raw_index':source_condition['raw_index']}
        if any(resumed['condition'].get(k)!=v for k,v in expected.items()):raise ValueError('Resume changes molecular condition')
        if resumed['energy_zero_eV']!=energy_zero:raise ValueError('Resume changes logging energy offset')
        if resumed.get('metadata_progress_sha256')!=(metadata.progress_sha256 if metadata else None):raise ValueError('Resume changes verified metadata')
    def load_model(branch):
        value=model_from_config(cfg);prepare_research_backbone(value,protocol)
        value.load_state_dict(state['state_dict'] if resume_state is None else resume_state[branch+'_state_dict'],strict=True);patch_smooth_geometry(value,protocol.get('geometry_softening',0.))
        if args.neutralize_temperature_input and resume_state is None:neutralize_constant_temperature_input(value,float(protocol['requested_kT']))
        return value.to(args.device).float().train()
    forward=load_model('forward');backward=load_model('backward');del state
    if args.mode=='backward_only':
        for parameter in forward.parameters():parameter.requires_grad_(False)
    precision_model=None
    if args.precision_kind!='none':
        precision_model=LearnedPairPrecision(mode=args.precision_kind,strength=args.precision_strength).double().to(args.device)
        if resume_state is not None:precision_model.load_state_dict(resume_state['precision_state_dict'],strict=True)
    field_parameters=[v for model in [forward,backward] for v in model.parameters() if v.requires_grad]
    precision_parameters=[] if precision_model is None else [v for v in precision_model.parameters() if v.requires_grad]
    parameters=field_parameters+precision_parameters
    groups=[{'params':field_parameters,'lr':args.lr}]
    if precision_parameters:groups.append({'params':precision_parameters,'lr':args.precision_lr})
    optimizer=torch.optim.AdamW(groups,lr=args.lr,weight_decay=0.)
    if resume_state is not None:restore_work_optimizer(optimizer,resume_state['optimizer_state_dict'],start_step)
    del resume_state
    graph=dgl.batch([base]*args.batch).to(args.device)
    if metadata is not None:metadata.attach(graph,[args.source_row]*args.batch,args.kT)
    nbi,_=get_batch_idxs(graph);uem=get_upper_edge_mask(graph)
    if args.objective=='log_variance' or args.gradient_diagnostics:
        conditioned=dgl.unbatch(graph)[0]
        observed_forward=MolecularPathDrift(forward,conditioned)
        observed_backward=MolecularPathDrift(backward,conditioned,sign=-1.)
    basis=centered_orthonormal_basis(n,device=args.device);prior_std=args.prior_std
    terminal_std=math.sqrt(args.kT/args.restraint) if args.reference_kernel=='gaussian' else None
    def drift(model,z,t,sign):
        x=torch.einsum('nk,bkd->bnd',basis,z.reshape(args.batch,n-1,3)).reshape(args.batch*n,3).float()
        # Re-enter deterministic mode in checkpoint recomputation as well.
        with graph.local_scope(),deterministic_field(model.vector_field),torch.autocast(device_type=x.device.type,enabled=False):
            for key in ['a','c']:graph.ndata[key+'_t']=graph.ndata[key+'_1_true']
            graph.edata['e_t']=graph.edata['e_1_true']
            v=position_velocity(model,graph,x,x.new_full((args.batch,),t),nbi,uem,parameterization='displacement')
        return sign*torch.einsum('nk,bnd->bkd',basis,v.double().reshape(args.batch,n,3)).reshape_as(z)
    def precision(z,t):
        x=torch.einsum('nk,bkd->bnd',basis,z.reshape(len(z),n-1,3))
        return precision_model(x,t,numbers,charge,spin,args.kT)
    precision_options={} if precision_model is None else {'forward_precision':precision,'backward_precision':precision}
    root=Path(__file__).resolve().parents[2]
    report={'complete':False,'scope':__doc__,'configuration':{k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'checkpoint_sha256':sha(args.checkpoint),'oracle_sha256':sha(args.oracle),'metadata_progress_sha256':metadata.progress_sha256 if metadata else None,
        'source_sha256':{str(path):sha(path) for path in [Path(__file__).resolve(),root/'cfm_mol/path_work.py',root/'cfm_mol/energy_oracle.py',root/'scripts/research/oracle_worker.py',root/'cfm_mol/condition_systems.py',root/'cfm_mol/electronic_conditioning.py',root/'cfm_mol/path_balance.py',root/'cfm_mol/molecular_path_drift.py',root/'cfm_mol/gradient_diagnostics.py',root/'cfm_mol/work_resume.py',root/'cfm_mol/pair_precision.py',root/'cfm_mol/matrix_gaussian.py']},
        'condition':{**source_condition,
            'numbers':numbers.tolist(),'charge':charge,'spin_multiplicity':spin},
        'energy_zero_eV':energy_zero,'prior_std':prior_std,'evaluations':{},'training':[]}
    report['temperature_input_initialization']={'neutralized':args.neutralize_temperature_input,
        'checkpoint_requested_kT':protocol.get('requested_kT'),'new_input_kT_eV':args.kT,
        'applied_in_this_run':args.neutralize_temperature_input and resumed is None}
    report['training_start_step']=start_step
    report['precision_protocol']={'kind':args.precision_kind,'strength':args.precision_strength,
        'softening_A':.1,'length_scale_A':2.,'trainable_parameters':sum(v.numel() for v in precision_parameters),
        'scope':'bounded pair precision; exact discrete Gaussian factors; molecular benefit unproven'}
    if resumed is not None:
        report['resume']={'run':str(args.resume_work_run.resolve()),'source_results_sha256':sha(args.resume_work_run/'results.json'),
            'source_checkpoint_sha256':sha(args.resume_work_run/'last.ckpt'),'optimizer_restored':True,
            'source_cumulative_oracle_evaluations':resumed.get('cumulative_oracle_evaluations',resumed['oracle_evaluations'])}
    report['objective_protocol']={'name':args.objective,'observation_measure':
        'fresh current-forward paths, detached and fixed within each gradient update' if args.objective=='log_variance' else 'reparameterized current-forward paths',
        'replay':False,'per_condition':True,'novel_objective_claim':False}
    write_json(output,report);start=time.perf_counter()
    with EnergyOracle(args.oracle_python,root/'scripts/research/oracle_worker.py',args.oracle,numbers=numbers,charge=charge,spin_multiplicity=spin,
                      batch_size=args.oracle_batch_size) as oracle:
        def draw(seed,training,retain_states=False):
            g=torch.Generator(device=args.device).manual_seed(seed)
            x0=torch.randn((args.batch,dimension),device=args.device,dtype=torch.float64,generator=g)*prior_std
            path=gaussian_training_path(x0,lambda z,t:drift(forward,z,t,1.),lambda z,t:drift(backward,z,t,-1.),
                torch.linspace(0,1,args.path_steps+1,dtype=torch.float64),args.noise,g,prior_std=prior_std,
                checkpoint_steps=args.checkpoint_steps and training,terminal_std=terminal_std,
                max_drift_norm=args.max_drift_per_sqrt_dimension*math.sqrt(dimension),
                forward_energy_only=args.mode=='forward_energy_only' and training,
                mean_parameterization=args.mean_parameterization,noise_annealing_power=args.noise_annealing_power,retain_states=retain_states,**precision_options)
            positions=torch.einsum('nk,bkd->bnd',basis,path.terminal.reshape(args.batch,n-1,3))
            energy,force=oracle.evaluate(positions)
            linked=external_energy(positions,energy,force) if training else energy.to(positions)
            reduced=(linked-energy_zero+args.restraint/2*positions.square().sum((1,2)))/args.kT
            return path.work(reduced),positions,energy,path.states,reduced
        def evaluate(label):
            before=oracle.evaluated;works=[];positions=[];energies=[]
            with torch.no_grad():
                for batch in range(args.eval_particles//args.batch):
                    w,x,e,_,_=draw(900000001+args.seed*1009+batch,False)
                    works.append(w.cpu());positions.append(x.cpu());energies.append(e)
            work=torch.cat(works);x=torch.cat(positions);energy=torch.cat(energies)
            weights=normalized_weights(-work)
            result={'mean_shifted_work':float(work.mean()),'work_std':float(work.std()),
                'mean_energy_eV':float(energy.mean()),'weights':WeightedPaths(x,-work,{}).summary(),
                'geometry':geometry_metrics(x,torch.full((len(x),),1/len(x),dtype=torch.float64),numbers),
                'weighted_geometry':geometry_metrics(x,weights,numbers),'oracle_evaluations':oracle.evaluated-before}
            torch.save({'positions':x,'energy_eV':energy,'work':work,'condition':report['condition']},args.out/f'{label}_samples.pt')
            if label=='initial' and resumed is not None:
                previous=torch.load(str(args.resume_work_run/'final_samples.pt'),map_location='cpu',weights_only=False)
                count=min(len(x),len(previous['positions']))
                comparison={'particles':count,'max_position_error_A':float((x[:count]-previous['positions'][:count]).abs().max()),
                    'max_work_error':float((work[:count]-previous['work'][:count]).abs().max())}
                report['resume_source_stream_comparison']=comparison;write_json(output,report)
                if comparison['max_position_error_A']>1e-4 or comparison['max_work_error']>.02:
                    raise ValueError('Resumed trained proposal failed source-stream reproduction')
            if label=='final' and args.mode=='backward_only':
                initial=torch.load(str(args.out/'initial_samples.pt'),map_location='cpu',weights_only=False)
                result['frozen_forward_positions_unchanged']=bool(torch.equal(x,initial['positions']))
                if not result['frozen_forward_positions_unchanged']:
                    raise RuntimeError('Backward-only control changed the fixed forward sampling law')
            report['evaluations'][label]=result;write_json(output,report);print(json.dumps({'evaluation':label,**result}),flush=True)
        if args.gradient_diagnostics:
            from cfm_mol.gradient_diagnostics import summarize_paired_gradients
            collections={key:[] for key in ['pathwise_forward','fixed_score_forward','pathwise_backward','fixed_score_backward']}
            records=[];forward_parameters=[p for p in forward.parameters() if p.requires_grad]
            split=len(forward_parameters)
            def flatten(values,params):
                return torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1).float().cpu() for g,p in zip(values,params)])
            for batch in range(args.gradient_diagnostics):
                work,_,energy,states,reduced=draw(700000007+args.seed*1009+batch,True,True)
                mean_gradient=torch.autograd.grad(work.mean(),parameters,allow_unused=True)
                initial,factor_f,factor_b=fixed_path_log_factors(states,observed_forward,observed_backward,
                    torch.linspace(0,1,args.path_steps+1,dtype=torch.float64),args.noise,
                    prior_std=prior_std,terminal_std=terminal_std,
                    max_drift_norm=args.max_drift_per_sqrt_dimension*math.sqrt(dimension),
                    mean_parameterization=args.mean_parameterization,noise_annealing_power=args.noise_annealing_power)
                observed=reduced.detach()+initial+factor_f-factor_b
                discrepancy=float((observed.detach()-work.detach()).abs().max())
                if discrepancy>.05:raise RuntimeError('Paired observed-path work mismatch')
                score_gradient=torch.autograd.grad(log_variance_balance(observed),parameters,allow_unused=True)
                for prefix,gradients in [('pathwise',mean_gradient),('fixed_score',score_gradient)]:
                    collections[prefix+'_forward'].append(flatten(gradients[:split],parameters[:split]))
                    collections[prefix+'_backward'].append(flatten(gradients[split:],parameters[split:]))
                row={'batch':batch,'mean_work':float(work.detach().mean()),'fixed_path_work_max_difference':discrepancy,
                    'seconds':time.perf_counter()-start}
                records.append(row);print(json.dumps(row),flush=True)
            report['gradient_diagnostics']=summarize_paired_gradients(collections,args.batch)
            report['gradient_diagnostics']['batches']=records
            report['gradient_diagnostics']['weights_updated']=False
            report.update(complete=True,oracle_evaluations=oracle.evaluated,seconds=time.perf_counter()-start)
            write_json(output,report)
            return
        evaluate('initial')
        for step in range(start_step,args.steps):
            optimizer.zero_grad(set_to_none=True)
            if args.objective=='log_variance':
                with torch.no_grad():
                    sampled_work,_,energy,states,reduced=draw(500000003+args.seed*1009+step,False,True)
                initial,factor_f,factor_b=fixed_path_log_factors(states,observed_forward,observed_backward,
                    torch.linspace(0,1,args.path_steps+1,dtype=torch.float64),args.noise,
                    prior_std=prior_std,terminal_std=terminal_std,
                    max_drift_norm=args.max_drift_per_sqrt_dimension*math.sqrt(dimension),
                    mean_parameterization=args.mean_parameterization,noise_annealing_power=args.noise_annealing_power)
                work=reduced.detach()+initial+factor_f-factor_b
                discrepancy=float((work.detach()-sampled_work).abs().max())
                if discrepancy>.05:raise RuntimeError(f'Observed path factors disagree with sampler: {discrepancy}')
                loss=log_variance_balance(work)
            else:
                work,_,energy,_,_=draw(500000003+args.seed*1009+step,True);loss=work.mean()
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);optimizer.step()
            row={'step':step+1,'mean_shifted_work':float(work.detach().mean()),'training_loss':float(loss.detach()),'mean_energy_eV':float(energy.mean()),
                 'gradient_norm':float(norm),'seconds':time.perf_counter()-start}
            if args.objective=='log_variance':row['fixed_path_work_max_difference']=discrepancy
            report['training'].append(row)
            if (step+1)%10==0 or step==0:write_json(output,report);print(json.dumps(row),flush=True)
        if not all(torch.isfinite(p).all() for p in parameters):raise FloatingPointError('Non-finite proposal parameters')
        evaluate('final')
        torch.save({'forward_state_dict':forward.state_dict(),'backward_state_dict':backward.state_dict(),
            'precision_state_dict':None if precision_model is None else precision_model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),'research_protocol':protocol,'proposal_protocol':report['configuration'],
            'global_step':args.steps,'prior_std':prior_std},args.out/'last.ckpt')
        report.update(complete=True,oracle_evaluations=oracle.evaluated,
            cumulative_oracle_evaluations=oracle.evaluated+(resumed.get('cumulative_oracle_evaluations',resumed['oracle_evaluations']) if resumed else 0),
            seconds=time.perf_counter()-start,
            peak_gpu_bytes=torch.cuda.max_memory_allocated() if str(args.device).startswith('cuda') else None)
        write_json(output,report)


if __name__=='__main__':main()
