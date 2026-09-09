#!/usr/bin/env python3
"""Matched-oracle MALA baseline from frozen FM geometries; no equilibrium claim."""
import argparse
import json
import math
from pathlib import Path
import time

import torch

from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.fixed_target_mcmc import mala_population
from cfm_mol.nonequilibrium import centered_orthonormal_basis
from cfm_mol.tempered_smc import DensityValue
from molecular_tempered_pilot import sha,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--oracle-python',type=Path,required=True);p.add_argument('--oracle',type=Path,required=True)
    p.add_argument('--steps',type=int,default=132);p.add_argument('--proposal-std',type=float,default=.1)
    p.add_argument('--seed',type=int,default=9061);p.add_argument('--kT',type=float,default=1.)
    p.add_argument('--restraint',type=float,default=.1);p.add_argument('--oracle-batch-size',type=int,default=16)
    p.add_argument('--scale-proposal-with-temperature',action='store_true',
        help='Scale proposal std by sqrt(kT/1eV) and score cap by 1eV/kT, preserving the capped physical-force drift')
    args=p.parse_args()
    if args.steps<1 or any(not math.isfinite(v) or v<=0 for v in [args.kT,args.restraint,args.proposal_std]):raise ValueError('Positive counts/scales required')
    parent=args.samples.parent/'results.json';source=json.loads(parent.read_text())
    if not source['complete']:raise ValueError('Require completed FM sampling')
    entries=[(i,row) for i,row in enumerate(source['samples']) if row['file']==args.samples.name]
    if len(entries)!=1 or sha(args.samples)!=entries[0][1]['sha256']:raise ValueError('FM source sample hash mismatch')
    source_field_calls=source['neural_field_calls_per_sample'][entries[0][0]]
    proposal_std=args.proposal_std*math.sqrt(args.kT) if args.scale_proposal_with_temperature else args.proposal_std
    score_cap=100./args.kT if args.scale_proposal_with_temperature else 100.
    data=torch.load(str(args.samples),map_location='cpu',weights_only=False)
    condition=data['condition'];numbers=condition['numbers'];positions=data['positions'].double()
    n=len(numbers);basis=centered_orthonormal_basis(n)
    initial=torch.einsum('nk,bnd->bkd',basis,positions).reshape(len(positions),-1)
    args.out.mkdir(parents=True,exist_ok=True);output=args.out/'results.json'
    if output.exists():raise FileExistsError(output)
    root=Path(__file__).resolve().parents[2]
    report={'complete':False,'scope':__doc__,'condition':condition,
        'configuration':{k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'source_samples_sha256':sha(args.samples),'source_results_sha256':sha(parent),'oracle_sha256':sha(args.oracle),
        'script_sha256':sha(Path(__file__)),'kernel_sha256':sha(root/'cfm_mol/fixed_target_mcmc.py'),
        'initialization_field_calls_per_sample':source_field_calls,'particles':len(positions),'history':[],
        'effective_proposal_std_A':proposal_std,'effective_score_norm_cap_per_A':score_cap,
        'limitations':['FM initialization is not equilibrium.','No endpoint density, normalizer or importance ESS is claimed.',
            'Acceptance is not mixing or mode coverage.','One seed/condition; this is a budgeted baseline.']}
    write_json(output,report);start=time.perf_counter();snapshots=[]
    with EnergyOracle(args.oracle_python,root/'scripts/research/oracle_worker.py',args.oracle,numbers=numbers,
        charge=condition['charge'],spin_multiplicity=condition['spin_multiplicity'],batch_size=args.oracle_batch_size) as oracle:
        def target(z):
            x=torch.einsum('nk,bkd->bnd',basis,z.reshape(len(z),n-1,3))
            energy,force=oracle.evaluate(x)
            log_value=-(energy+args.restraint/2*x.square().sum((1,2)))/args.kT
            force=(force-args.restraint*x)/args.kT
            score=torch.einsum('nk,bnd->bkd',basis,force).reshape_as(z)
            return DensityValue(log_value,score)
        def record(step,z,value,accepted,evaluations):
            if step==0 or step%10==0 or step==args.steps:
                x=torch.einsum('nk,bkd->bnd',basis,z.reshape(len(z),n-1,3))
                energy=-args.kT*value.log_value-args.restraint/2*x.square().sum((1,2))
                row={'step':step,'mean_energy_eV':float(energy.mean()),'mean_restrained_energy_eV':float(-args.kT*value.log_value.mean()),
                    'acceptance_fraction':float(accepted.sum())/(len(z)*step) if step else None,
                    'target_evaluations':evaluations,'seconds':time.perf_counter()-start}
                report['history'].append(row);snapshots.append({'step':step,'positions':x.clone(),'energy_eV':energy.clone()})
                write_json(output,report);print(json.dumps(row),flush=True)
                if step in [0,args.steps]:
                    torch.save({'positions':x,'energy_eV':energy,'condition':condition,
                        'density_scope':'unknown finite-time MCMC endpoint density'},args.out/('initial_samples.pt' if step==0 else 'final_samples.pt'))
        final,_,stats=mala_population(initial,target,steps=args.steps,proposal_std=proposal_std,max_score_norm=score_cap,
            generator=torch.Generator().manual_seed(args.seed),callback=record)
        if stats['target_evaluations']!=oracle.evaluated:raise RuntimeError('Oracle query accounting mismatch')
        torch.save({'condition':condition,'snapshots':snapshots},args.out/'trajectory_snapshots.pt')
        report.update(complete=True,summary=stats,oracle_evaluations=oracle.evaluated,seconds=time.perf_counter()-start)
        write_json(output,report)


if __name__=='__main__':main()
