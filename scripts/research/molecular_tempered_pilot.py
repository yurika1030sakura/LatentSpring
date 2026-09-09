#!/usr/bin/env python3
"""Matched-oracle tempered sampling from Gaussian or FM-mixture proposals.

Development comparison with fixed schedules. A separate FM pilot fixes a finite
Gaussian mixture with exact density; this is not the FM model's likelihood.
Raw electronic-state provenance is unavailable, so the computed spin is declared.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import dgl
import numpy as np
import torch
from ase.data import atomic_numbers,covalent_radii
from flowmol.data_processing.dataset import MoleculeDataset
from flowmol.data_processing.utils import get_batch_idxs,get_upper_edge_mask
from flowmol.model_utils.load import read_config_file,model_from_config

from cfm_mol.clamped_density import sample_clamped_flow
from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.nonequilibrium import centered_orthonormal_basis,normalized_weights
from cfm_mol.radial_reference import prepare_research_backbone
from cfm_mol.smooth_geometry import patch_smooth_geometry
from cfm_mol.tempered_smc import DensityValue,IsotropicGaussianMixture,tempered_smc
from cfm_mol.rotation_mixture import RotatedGaussianMixture,symmetry_templates
from cfm_mol.defensive_proposal import DefensiveProposal


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write_json(path,data):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def geometry_metrics(x,weights,numbers):
    profiles=torch.stack([torch.pdist(row) for row in x])
    mean=weights@profiles
    radii=torch.tensor(covalent_radii[numbers],dtype=x.dtype)
    pairs=torch.triu_indices(len(numbers),len(numbers),offset=1)
    overlap=(profiles<.6*(radii[pairs[0]]+radii[pairs[1]])).any(-1)
    return {'weighted_pair_distance_variance_A2':float(weights@((profiles-mean).square().mean(-1))),
            'weighted_overlap_fraction':float(weights@overlap.double()),
            'unweighted_overlap_fraction':float(overlap.double().mean()),
            'weighted_radius_gyration_A':float(weights@x.square().sum(-1).mean(-1).sqrt()),
            'scope':'geometry heuristics; not chemical validity or measured Boltzmann mode coverage'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--oracle',type=Path,required=True);p.add_argument('--oracle-python',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--source-rows',type=int,nargs='+',default=[5846])
    p.add_argument('--seeds',type=int,nargs='+',default=[9031]);p.add_argument('--particles',type=int,default=32)
    p.add_argument('--centers',type=int,default=32);p.add_argument('--stages',type=int,default=32)
    p.add_argument('--fm-steps',type=int,default=128);p.add_argument('--mixture-std',type=float,default=.3)
    p.add_argument('--proposal-std',type=float,default=.08);p.add_argument('--moves',type=int,default=1)
    p.add_argument('--resample-threshold',type=float,default=.5);p.add_argument('--beta-power',type=float,default=2.)
    p.add_argument('--max-score-norm',type=float,default=100.)
    p.add_argument('--max-permutations',type=int,default=64)
    p.add_argument('--rotation-tolerance',type=float,default=1e-10)
    p.add_argument('--defensive-fraction',type=float,default=.2)
    p.add_argument('--kT',type=float,default=1.);p.add_argument('--restraint-strength',type=float,default=.1)
    p.add_argument('--arms',nargs='+',choices=['prior_rwm','prior_mala','mixture_rwm','mixture_mala',
        'rotation_rwm','rotation_mala','symmetry_rwm','symmetry_mala','confinement_mala',
        'defensive_mixture_mala','defensive_symmetry_mala','confinement_hybrid','defensive_symmetry_hybrid'],
                   default=['prior_rwm','prior_mala','mixture_rwm','mixture_mala'])
    p.add_argument('--device',default='cpu');p.add_argument('--oracle-device',default='cpu')
    args=p.parse_args()
    if min(args.particles,args.centers,args.stages,args.fm_steps,args.moves)<1:raise ValueError('Invalid experiment size')
    if not all(np.isfinite(v) and v>0 for v in [args.kT,args.restraint_strength,args.mixture_std,args.proposal_std,args.beta_power]):
        raise ValueError('Invalid physical/proposal scale')
    args.out.mkdir(parents=True,exist_ok=True);output=args.out/'results.json'
    if output.exists():raise FileExistsError(output)
    root=Path(__file__).resolve().parents[2]
    source_files=[Path(__file__).resolve(),root/'cfm_mol/tempered_smc.py',root/'cfm_mol/energy_oracle.py',
        root/'cfm_mol/rotation_mixture.py',root/'cfm_mol/defensive_proposal.py',root/'scripts/research/oracle_worker.py']
    cfg=read_config_file(args.config);cfg['mol_fm'].pop('bgfm',None);cfg['mol_fm']['prior_config']['x']['align']=False
    dataset=MoleculeDataset('val',dict(cfg['dataset'],fake_atom_p=0.,fake_atom_std=1.,
        explicit_aromaticity=cfg['mol_fm'].get('explicit_aromaticity',False)),prior_config=cfg['mol_fm']['prior_config'])
    state=torch.load(str(args.checkpoint),map_location='cpu',weights_only=False)
    protocol=state.get('research_protocol')
    if protocol is None:raise ValueError('Require a declared conditional FM checkpoint')
    model=model_from_config(cfg);prepare_research_backbone(model,protocol);model.load_state_dict(state['state_dict'],strict=True)
    model.to(args.device).float().eval();patch_smooth_geometry(model,protocol.get('geometry_softening',0.))
    report={'complete':False,'scope':__doc__,'configuration':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'checkpoint_sha256':sha(args.checkpoint),'oracle_sha256':sha(args.oracle),'config_sha256':sha(args.config),
        'source_sha256':{str(f):sha(f) for f in source_files},'research_protocol':protocol,
        'target':'exp(-(eSEN(x)+restraint_strength/2*||x||^2)/kT) on intrinsic COM-free H',
        'normalization_assumption':'eSEN bounded below, positive harmonic confinement',
        'conditions':[],'rows':[]}
    write_json(output,report)
    for source_row in args.source_rows:
        base=dataset[source_row];n=base.num_nodes()
        if not 2<=n<=200:raise ValueError('Unsupported atom count')
        charges=base.ndata['c_1_true'].argmax(-1)-2
        if ((charges<=-2)|(charges>=3)).any():raise ValueError('Clipping-boundary charge excluded')
        symbols=[cfg['dataset']['atom_map'][int(i)] for i in base.ndata['a_1_true'].argmax(-1)]
        numbers=np.array([atomic_numbers[s] for s in symbols]);charge=int(charges.sum());spin=1+(int(numbers.sum())-charge)%2
        condition={'source_row':source_row,'symbols':symbols,'charge':charge,'spin_multiplicity':spin,
            'spin_provenance':'declared minimum electron-parity multiplicity; original OMol state unknown',
            'source_scope':'legacy validation development condition, not blind test'}
        basis=centered_orthonormal_basis(n);dimension=3*(n-1)
        # The pilot centers are independent of every production-particle seed.
        graph=dgl.batch([base]*args.centers).to(args.device);nbi,_=get_batch_idxs(graph);uem=get_upper_edge_mask(graph)
        start=time.perf_counter()
        with torch.no_grad():
            positions=sample_clamped_flow(model,graph,nbi,uem,n_ode_steps=args.fm_steps,
                terminal_time=protocol['data_endpoint_time'],parameterization=protocol['position_parameterization'],
                generator=torch.Generator(device=args.device).manual_seed(29031+source_row)).cpu().double().reshape(args.centers,n,3)
        centers=torch.einsum('nk,bnd->bkd',basis,positions).reshape(args.centers,dimension)
        center_file=args.out/f'centers_{source_row}.pt';torch.save({'positions':positions,'centers':centers,'condition':condition},center_file)
        condition.update(fm_center_seconds=time.perf_counter()-start,fm_center_field_evaluations=2*args.fm_steps*args.centers,
                         center_file=str(center_file.resolve()),center_sha256=sha(center_file))
        report['conditions'].append(condition);write_json(output,report)
        mixtures={'prior':IsotropicGaussianMixture(torch.zeros(1,dimension,dtype=torch.float64),1.),
                  'mixture':IsotropicGaussianMixture(centers,args.mixture_std),
                  'confinement':IsotropicGaussianMixture(torch.zeros(1,dimension,dtype=torch.float64),np.sqrt(args.kT/args.restraint_strength))}
        if any(arm.startswith('rotation_') for arm in args.arms):
            mixtures['rotation']=RotatedGaussianMixture(centers,args.mixture_std,tolerance=args.rotation_tolerance)
        if any(arm.startswith('symmetry_') or arm.startswith('defensive_symmetry_') for arm in args.arms):
            augmented,symmetries=symmetry_templates(centers,basis,numbers,
                max_permutations=args.max_permutations,reflect=True,seed=49031+source_row)
            mixtures['symmetry']=RotatedGaussianMixture(augmented,args.mixture_std,tolerance=args.rotation_tolerance)
            condition['symmetry_templates']=symmetries
            write_json(output,report)
        for family in ['mixture','symmetry']:
            if any(arm.startswith('defensive_'+family+'_') for arm in args.arms):
                mixtures['defensive_'+family]=DefensiveProposal(mixtures[family],dimension,
                    np.sqrt(args.kT/args.restraint_strength),args.defensive_fraction)
        with EnergyOracle(args.oracle_python,root/'scripts/research/oracle_worker.py',args.oracle,
                numbers=numbers,charge=charge,spin_multiplicity=spin,device=args.oracle_device) as oracle:
            def target(z):
                cartesian=torch.einsum('nk,bkd->bnd',basis,z.reshape(len(z),n-1,3))
                energy,force=oracle.evaluate(cartesian)
                restraint=args.restraint_strength/2*z.square().sum(-1)
                score=(torch.einsum('nk,bnd->bkd',basis,force).reshape_as(z)-args.restraint_strength*z)/args.kT
                return DensityValue(-(energy+restraint)/args.kT,score)
            for seed in args.seeds:
                for arm in args.arms:
                    family,kernel=arm.rsplit('_',1);initial=mixtures[family]
                    g=torch.Generator().manual_seed(seed+source_row*100003)
                    x0,components=initial.sample(args.particles,g)
                    row={'source_row':source_row,'seed':seed,'arm':arm,'complete':False,'history':[]}
                    report['rows'].append(row);before=oracle.evaluated;requested_before=oracle.requested_evaluations;start=time.perf_counter()
                    def callback(item):
                        row['history'].append(item);write_json(output,report)
                        if item['stage']%8==0:print(json.dumps({'source_row':source_row,'seed':seed,'arm':arm,**item}),flush=True)
                    try:
                        result=tempered_smc(x0,initial,target,torch.linspace(0,1,args.stages+1,dtype=torch.float64)**args.beta_power,
                            proposal_std=args.proposal_std,generator=g,kernel=kernel,moves_per_stage=args.moves,
                            resample_threshold=args.resample_threshold,max_score_norm=args.max_score_norm,callback=callback)
                        weights=normalized_weights(result.log_weights)
                        x=torch.einsum('nk,bkd->bnd',basis,result.positions.reshape(len(x0),n-1,3))
                        energy=-args.kT*result.target_log_values-args.restraint_strength/2*result.positions.square().sum(-1)
                        particle_file=args.out/f'particles_{source_row}_{seed}_{arm}.pt'
                        torch.save({'positions':x,'log_weights':result.log_weights,'ancestors':result.ancestors,
                            'initial_components':components,'energies_eV':energy,'condition':condition,
                            'last_global_proposal':result.last_global_proposal},particle_file)
                        row.update(success=True,summary=result.summary(),geometry=geometry_metrics(x,weights,numbers),
                            weighted_energy_eV=float(weights@energy),unweighted_mean_energy_eV=float(energy.mean()),
                            particle_file=str(particle_file.resolve()),particle_sha256=sha(particle_file))
                        density_component=initial.local if isinstance(initial,DefensiveProposal) else initial
                        if isinstance(initial,DefensiveProposal):
                            row['defensive_component']={'fraction':initial.wide_fraction,
                                'standard_deviation_A':initial.confinement_std,
                                'initial_wide_particles':int((components==-1).sum()),
                                'final_weight_from_wide_ancestors':float(weights@(components[result.ancestors]==-1).double())}
                            if kernel=='hybrid':
                                row['defensive_component']['ancestry_scope']='initial chain roots; later global proposals may come from either mixture component'
                        if isinstance(density_component,RotatedGaussianMixture):
                            row['density_numerics']={'cumulative_max_order':density_component.max_observed_order,
                                'cumulative_max_final_log_refinement_change':density_component.max_observed_refinement_change,
                                'certified_error_bound':False,
                                'scope':'deterministic quadrature convergence screen; sampling identities concern the exact integral'}
                    except Exception as exc:
                        row.update(success=False,error=f'{type(exc).__name__}: {str(exc)[:1200]}')
                    row.update(complete=True,oracle_evaluations=oracle.evaluated-before,
                        requested_oracle_evaluations=oracle.requested_evaluations-requested_before,
                        seconds=time.perf_counter()-start)
                    write_json(output,report);print(json.dumps({k:v for k,v in row.items() if k!='history'}),flush=True)
    report['complete']=True;write_json(output,report)


if __name__=='__main__':main()
