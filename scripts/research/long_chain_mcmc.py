#!/usr/bin/env python3
"""Equal-query longer-chain MALA/HMC baseline with chain-clustered assessment."""
import argparse
import json
import math
from pathlib import Path
import time

import torch

from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.fixed_target_mcmc import mala_population,hmc_population
from cfm_mol.nonequilibrium import centered_orthonormal_basis
from cfm_mol.tempered_smc import DensityValue
from molecular_tempered_pilot import sha,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--oracle',type=Path,required=True);p.add_argument('--oracle-python',type=Path,required=True)
    p.add_argument('--kernel',choices=['mala','hmc'],required=True)
    p.add_argument('--kT',type=float,default=.025851999786435);p.add_argument('--restraint',type=float,default=.1)
    args=p.parse_args();source=args.samples.parent/'results.json';parent=json.loads(source.read_text())
    if not parent['complete']:raise ValueError('Require completed FM source')
    entry=next(row for row in parent['samples'] if row['file']==args.samples.name)
    if sha(args.samples)!=entry['sha256']:raise ValueError('FM sample hash changed')
    if not all(math.isfinite(v) and v>0 for v in [args.kT,args.restraint]):raise ValueError('Invalid target scales')
    data=torch.load(str(args.samples),map_location='cpu',weights_only=False);condition=data['condition'];numbers=condition['numbers'];n=len(numbers)
    if len(data['positions'])<8:raise ValueError('Need eight fixed independent starts')
    indices=torch.randperm(len(data['positions']),generator=torch.Generator().manual_seed(9062))[:8]
    positions=data['positions'][indices].double();basis=centered_orthonormal_basis(n)
    initial=torch.einsum('nk,bnd->bkd',basis,positions).reshape(8,-1)
    output=args.out/'results.json'
    if output.exists():raise FileExistsError(output)
    args.out.mkdir(parents=True,exist_ok=True);root=Path(__file__).resolve().parents[2]
    # Both methods collect at precisely these per-chain oracle-update counts.
    collect=set(range(615,1064,64));samples=[];energy_samples=[];history=[]
    report={'complete':False,'scope':__doc__,'condition':condition,
        'configuration':{'kernel':args.kernel,'kT':args.kT,'restraint':args.restraint,'chains':8,'seed':9061},
        'source_samples_sha256':sha(args.samples),'oracle_sha256':sha(args.oracle),'selected_source_indices':indices.tolist(),
        'selection_seed':9062,'collection_force_updates':sorted(collect),'samples_per_chain':8,
        'hmc_leapfrog_counts':[7]+[32]*33 if args.kernel=='hmc' else None,
        'source_field_calls_per_initial_sample':32,'history':history,
        'limitations':['Finite chains are not declared equilibrated.','Eight retained draws per chain are correlated.',
            'No endpoint density, importance ESS or normalizer is claimed.','All initially selected chains remain in the output.']}
    write_json(output,report);start=time.perf_counter()
    with EnergyOracle(args.oracle_python,root/'scripts/research/oracle_worker.py',args.oracle,numbers=numbers,
        charge=condition['charge'],spin_multiplicity=condition['spin_multiplicity'],batch_size=16) as oracle:
        def target(z):
            x=torch.einsum('nk,bkd->bnd',basis,z.reshape(len(z),n-1,3));energy,force=oracle.evaluate(x)
            return DensityValue(-(energy+args.restraint/2*x.square().sum((1,2)))/args.kT,
                torch.einsum('nk,bnd->bkd',basis,(force-args.restraint*x)/args.kT).reshape_as(z))
        def record(iteration,z,value,accepted,evaluations):
            updates=evaluations//8-1
            if updates==0 or updates in collect or iteration%32==0:
                x=torch.einsum('nk,bkd->bnd',basis,z.reshape(8,n-1,3))
                energy=-args.kT*value.log_value-args.restraint/2*x.square().sum((1,2))
                row={'iteration':iteration,'force_updates_per_chain':updates,'queries':evaluations,
                    'mean_energy_eV':float(energy.mean()),'acceptance':float(accepted.sum())/(8*iteration) if iteration else None,
                    'seconds':time.perf_counter()-start}
                history.append(row);write_json(output,report);print(json.dumps(row),flush=True)
                if updates in collect:samples.append(x.clone());energy_samples.append(energy.clone())
        generator=torch.Generator().manual_seed(9061)
        if args.kernel=='mala':
            _,_,stats=mala_population(initial,target,steps=1063,proposal_std=.1*math.sqrt(args.kT),
                max_score_norm=100./args.kT,generator=generator,callback=record)
        else:
            _,_,stats=hmc_population(initial,target,leapfrog_counts=[7]+[32]*33,step_size=.05*math.sqrt(args.kT),
                max_score_norm=100./args.kT,generator=generator,callback=record)
        if len(samples)!=8 or oracle.evaluated!=8512 or stats['target_evaluations']!=8512:raise RuntimeError('Matched budget/collection count changed')
        x=torch.stack(samples,1).reshape(64,n,3);energy=torch.stack(energy_samples,1).reshape(64)
        torch.save({'positions':x,'energy_eV':energy,'condition':condition,
            'sample_cluster_ids':torch.arange(8).repeat_interleave(8),'density_scope':'unknown finite-time MCMC law'},args.out/'final_samples.pt')
        torch.save({'positions':positions,'condition':condition,'source_indices':indices},args.out/'initial_selected.pt')
        report.update(complete=True,oracle_evaluations=oracle.evaluated,summary=stats,seconds=time.perf_counter()-start)
        write_json(output,report)


if __name__=='__main__':main()
