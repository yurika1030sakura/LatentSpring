#!/usr/bin/env python3
"""Equal-oracle direct-IS control for confinement and defensive symmetry proposals.

Uses a completed independent FM-center pilot. No MCMC or resampling obscures
the importance-weight comparison. Every seed, tail component and failure is
retained; numerical rotational integration diagnostics remain explicit.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from ase.data import atomic_numbers

from cfm_mol.defensive_proposal import DefensiveProposal
from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.nonequilibrium import centered_orthonormal_basis,normalized_weights,WeightedPaths
from cfm_mol.rotation_mixture import RotatedGaussianMixture,symmetry_templates,random_rotations
from cfm_mol.tempered_smc import IsotropicGaussianMixture
from cfm_mol.triatomic_reference import invariant_observables
from molecular_tempered_pilot import geometry_metrics


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot',type=Path,required=True);p.add_argument('--source-row',type=int,required=True)
    p.add_argument('--oracle',type=Path,required=True);p.add_argument('--oracle-python',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--particles',type=int,default=1024)
    p.add_argument('--seeds',type=int,nargs='+',default=[9031,9032,9033]);p.add_argument('--batch',type=int,default=64)
    p.add_argument('--kT',type=float,default=1.);p.add_argument('--restraint',type=float,default=.1)
    p.add_argument('--mixture-std',type=float,default=.3);p.add_argument('--defensive-fraction',type=float,default=.2)
    args=p.parse_args()
    if min(args.particles,args.batch)<1 or args.kT<=0 or args.restraint<=0:raise ValueError('Invalid integration controls')
    args.out.mkdir(parents=True,exist_ok=True);output=args.out/'results.json'
    if output.exists():raise FileExistsError(output)
    source_file=args.pilot/'results.json';pilot=json.loads(source_file.read_text())
    if not pilot['complete']:raise ValueError('Require a complete center pilot')
    condition=next(c for c in pilot['conditions'] if c['source_row']==args.source_row)
    center_file=Path(condition['center_file'])
    if sha(center_file)!=condition['center_sha256']:raise ValueError('Pilot centers changed')
    data=torch.load(str(center_file),map_location='cpu',weights_only=False)
    centers=data['centers'].double();numbers=np.array([atomic_numbers[s] for s in condition['symbols']]);n=len(numbers)
    basis=centered_orthonormal_basis(n);dimension=3*(n-1)
    augmented,symmetries=symmetry_templates(centers,basis,numbers,reflect=True,seed=49031+args.source_row)
    sigma=math.sqrt(args.kT/args.restraint)
    proposals={'confinement':IsotropicGaussianMixture(torch.zeros(1,dimension,dtype=torch.float64),sigma),
        'defensive_mixture':DefensiveProposal(IsotropicGaussianMixture(centers,args.mixture_std),dimension,sigma,args.defensive_fraction),
        'defensive_symmetry':DefensiveProposal(RotatedGaussianMixture(augmented,args.mixture_std),dimension,sigma,args.defensive_fraction)}
    root=Path(__file__).resolve().parents[2]
    files=[Path(__file__).resolve(),root/'cfm_mol/defensive_proposal.py',root/'cfm_mol/rotation_mixture.py',
           root/'cfm_mol/tempered_smc.py',root/'cfm_mol/energy_oracle.py',root/'scripts/research/oracle_worker.py']
    report={'complete':False,'scope':__doc__,'condition':condition,'symmetries':symmetries,
        'configuration':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'pilot_source':str(source_file.resolve()),'pilot_sha256':sha(source_file),'oracle_sha256':sha(args.oracle),
        'source_sha256':{str(f):sha(f) for f in files},'rows':[]}
    write(output,report)
    with EnergyOracle(args.oracle_python,root/'scripts/research/oracle_worker.py',args.oracle,
        numbers=numbers,charge=condition['charge'],spin_multiplicity=condition['spin_multiplicity']) as oracle:
        original=data['positions'][:8].double();e0,_=oracle.evaluate(original)
        r=random_rotations(len(original),torch.Generator().manual_seed(9029))
        permutations=list(range(n))
        for element in sorted(set(numbers)):
            group=np.flatnonzero(numbers==element)
            if len(group)>1:permutations[group[0]],permutations[group[1]]=permutations[group[1]],permutations[group[0]]
        errors={}
        for name,x in [('rotation',original@r),('reflection',-original),('identical_atom_permutation',original[:,permutations])]:
            value,_=oracle.evaluate(x);errors[name]=float((value-e0).abs().max())
        report['symmetry_energy_check_max_eV']=errors;report['symmetry_check_oracle_calls']=oracle.evaluated
        write(output,report)
        for seed in args.seeds:
            for name,proposal in proposals.items():
                start=time.perf_counter();before=oracle.evaluated
                row={'seed':seed,'arm':name,'complete':False};report['rows'].append(row);write(output,report)
                try:
                    z,labels=proposal.sample(args.particles,torch.Generator().manual_seed(seed+args.source_row*100003))
                    positions=torch.einsum('nk,bkd->bnd',basis,z.reshape(len(z),n-1,3))
                    logq=[];energies=[]
                    for begin in range(0,len(z),args.batch):
                        end=begin+args.batch
                        logq.append(proposal(z[begin:end]).log_value)
                        e,_=oracle.evaluate(positions[begin:end]);energies.append(e)
                    e=torch.cat(energies);q=torch.cat(logq)
                    logw=-(e+args.restraint/2*z.square().sum(-1))/args.kT-q
                    weights=normalized_weights(logw)
                    result=WeightedPaths(positions,logw,{})
                    row.update(success=True,summary=result.summary(),geometry=geometry_metrics(positions,weights,numbers),
                        weighted_energy_eV=float(weights@e))
                    if n==3:row['moments']={key:float(weights.numpy()@value) for key,value in invariant_observables(positions.numpy()).items()}
                    if isinstance(proposal,DefensiveProposal):
                        row['wide_particle_count']=int((labels==-1).sum());row['weight_from_wide_component']=float(weights@(labels==-1).double())
                    local=proposal.local if isinstance(proposal,DefensiveProposal) else proposal
                    if isinstance(local,RotatedGaussianMixture):
                        row['density_numerics']={'cumulative_max_order':local.max_observed_order,
                            'cumulative_max_final_refinement_change':local.max_observed_refinement_change,'certified_error_bound':False}
                    sample_file=args.out/f'{name}_{seed}.pt'
                    torch.save({'positions':positions,'log_weights':logw,'energy_eV':e,'condition':condition},sample_file)
                    row.update(sample_file=str(sample_file.resolve()),sample_sha256=sha(sample_file))
                except Exception as exc:row.update(success=False,error=f'{type(exc).__name__}: {str(exc)[:1000]}')
                row.update(complete=True,oracle_evaluations=oracle.evaluated-before,seconds=time.perf_counter()-start)
                write(output,report);print(json.dumps(row),flush=True)
    report['complete']=True;write(output,report)


if __name__=='__main__':main()
