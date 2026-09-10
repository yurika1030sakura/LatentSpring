#!/usr/bin/env python3
"""Independent Gaussian-start HMC-SMC pilot; a reference candidate, not a certificate."""
import argparse
import json
import math
from pathlib import Path
import time

import torch

from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.nonequilibrium import centered_orthonormal_basis,normalized_weights
from cfm_mol.tempered_smc import DensityValue,IsotropicGaussianMixture,tempered_smc
from cfm_mol.triatomic_reference import invariant_observables
from molecular_tempered_pilot import sha,write_json,geometry_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--condition-run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reference',type=Path);p.add_argument('--particles',type=int,default=32)
    p.add_argument('--stages',type=int,default=128);p.add_argument('--leapfrog',type=int,default=8)
    p.add_argument('--seed',type=int,default=9069);args=p.parse_args()
    source=args.condition_run/'results.json';old=json.loads(source.read_text());recipe=old['configuration']
    if not old['complete']:raise ValueError('Require a completed source for audited condition metadata')
    condition={k:v for k,v in old['condition'].items() if not k.startswith('reference_geometry_')};numbers=condition['numbers'];n=len(numbers);dimension=3*(n-1)
    kT=recipe['kT'];restraint=recipe['restraint'];energy_zero=old['energy_zero_eV']
    if min(args.particles,args.stages,args.leapfrog)<1:raise ValueError('Positive counts required')
    oracle_path=Path(recipe['oracle'])
    if sha(oracle_path)!=old['oracle_sha256']:raise ValueError('Oracle checkpoint changed')
    output=args.out/'results.json'
    if output.exists():raise FileExistsError(output)
    args.out.mkdir(parents=True,exist_ok=True);root=Path(__file__).resolve().parents[2]
    initial_std=math.sqrt(1./restraint);basis=centered_orthonormal_basis(n)
    initial=IsotropicGaussianMixture(torch.zeros(1,dimension,dtype=torch.float64),initial_std)
    generator=torch.Generator().manual_seed(args.seed);z0,_=initial.sample(args.particles,generator)
    report={'complete':False,'scope':__doc__,'condition':condition,
        'configuration':{'kT':kT,'restraint':restraint,'particles':args.particles,'stages':args.stages,
            'leapfrog_steps':args.leapfrog,'seed':args.seed,'initial_std':initial_std,'beta_power':2.,
            'hmc_step_rule':'.05 * sqrt(kT / max(beta, kT))','max_score_norm':100./kT},
        'energy_zero_eV':energy_zero,'oracle_sha256':old['oracle_sha256'],'condition_source_sha256':sha(source),
        'source_sha256':{str(path):sha(path) for path in [Path(__file__),root/'cfm_mol/tempered_smc.py',root/'cfm_mol/fixed_target_mcmc.py']},
        'reference_geometry_loaded':False,'fm_weights_or_samples_loaded':False,'history':[],
        'limitations':['One SMC population is not a converged reference.',
            'Resampling can increase endpoint ESS while losing initial lineages.',
            'No iid endpoint or confidence interval is inferred from resampled particles.']}
    write_json(output,report);start=time.perf_counter()
    with EnergyOracle(Path(recipe['oracle_python']),root/'scripts/research/oracle_worker.py',oracle_path,
        numbers=numbers,charge=condition['charge'],spin_multiplicity=condition['spin_multiplicity'],batch_size=16) as oracle:
        def target(z):
            x=torch.einsum('nk,bkd->bnd',basis,z.reshape(len(z),n-1,3));energy,force=oracle.evaluate(x)
            return DensityValue(-(energy-energy_zero+restraint/2*x.square().sum((1,2)))/kT,
                torch.einsum('nk,bnd->bkd',basis,(force-restraint*x)/kT).reshape_as(z))
        def record(row):
            report['history'].append(row)
            if row['stage']%8==0:
                write_json(output,report);print(json.dumps({**row,'seconds':time.perf_counter()-start}),flush=True)
        result=tempered_smc(z0,initial,target,torch.linspace(0,1,args.stages+1,dtype=torch.float64).square(),
            kernel='hmc',proposal_std=.05,generator=generator,hmc_leapfrog_steps=args.leapfrog,
            hmc_step_size_schedule=lambda beta:.05*math.sqrt(kT/max(beta,kT)),
            max_score_norm=100./kT,resample_threshold=.5,callback=record)
        expected=args.particles*(1+args.stages*args.leapfrog)
        if oracle.evaluated!=result.target_evaluations or oracle.evaluated!=expected:raise RuntimeError('HMC-SMC oracle budget mismatch')
        checked=target(result.positions)
        error=float((checked.log_value-result.target_log_values).abs().max())
        if error>1e-3:raise RuntimeError(f'Cached accepted target value mismatch: {error}')
        x=torch.einsum('nk,bkd->bnd',basis,result.positions.reshape(len(z0),n-1,3));weights=normalized_weights(result.log_weights)
        energy=energy_zero-kT*checked.log_value-restraint/2*x.square().sum((1,2))
        torch.save({'positions':x,'energy_eV':energy,'log_weights':result.log_weights,
            'sample_cluster_ids':result.ancestors,'condition':condition,'density_scope':'weighted SMC population, not iid paths'},args.out/'final_samples.pt')
        report.update(summary=result.summary(),absolute_log_normalizer_estimate=result.log_normalizer_estimate-energy_zero/kT,
            mean_energy_eV=float(weights@energy),geometry=geometry_metrics(x,weights,numbers),
            terminal_cache_max_error=error,terminal_cache_tolerance=1e-3,terminal_cache_check_queries=args.particles,oracle_evaluations=oracle.evaluated,
            seconds=time.perf_counter()-start)
        if args.reference is not None:
            ref=json.loads((args.reference/'reference.json').read_text())
            if not ref['complete'] or any(ref['condition'][key]!=condition[key] for key in ['numbers','charge','spin_multiplicity']):raise ValueError('Reference condition mismatch')
            if ref['oracle_sha256']!=old['oracle_sha256'] or ref['restraint_eV_A2']!=restraint:raise ValueError('Reference potential mismatch')
            reference=max([row for row in ref['aggregate'] if abs(row['kT_eV']-kT)<1e-12],key=lambda row:row['particles_per_scramble'])
            moments={key:{'smc_estimate':float(weights@torch.as_tensor(value)),
                'reference_mean':reference['moments'][key]['mean']} for key,value in invariant_observables(x.numpy()).items()}
            report['reference_comparison']={'log_normalizer_difference':report['absolute_log_normalizer_estimate']-reference['log_of_mean_normalizer_estimate'],
                'reference_relative_se':reference['relative_se_across_scrambles'],'moments':moments,'source_sha256':sha(args.reference/'reference.json')}
        report['complete']=True;write_json(output,report);print(json.dumps(report['summary']),flush=True)


if __name__=='__main__':main()
