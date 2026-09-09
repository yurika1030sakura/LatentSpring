#!/usr/bin/env python3
"""Independent randomized-QMC reference for a three-atom restrained eSEN target.

No learned model or generated templates enter the integration proposal. This
estimates thermodynamic integrals; finite-sample uncertainty and resolution
changes are reported, never silently labeled an exact reference.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp
import torch
import yaml
from ase.data import atomic_numbers

from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.rotation_mixture import random_rotations
from cfm_mol.triatomic_reference import (gaussian_triatomic_shapes,
    centered_gaussian_mixture_log_density,invariant_observables)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--processed',type=Path,required=True)
    p.add_argument('--source-row',type=int,default=1137);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--oracle',type=Path,required=True);p.add_argument('--oracle-python',type=Path,required=True)
    p.add_argument('--power',type=int,default=10);p.add_argument('--scrambles',type=int,default=4)
    p.add_argument('--batch',type=int,default=64);p.add_argument('--kT',type=float,default=1.)
    p.add_argument('--restraint',type=float,default=.1);p.add_argument('--oracle-device',default='cpu')
    args=p.parse_args()
    if args.kT<=0 or args.restraint<=0 or args.batch<1 or args.power<3 or args.scrambles<2:raise ValueError('Invalid reference protocol')
    args.out.mkdir(parents=True,exist_ok=True);output=args.out/'reference.json'
    if output.exists():raise FileExistsError(output)
    atom_map=yaml.safe_load(args.config.read_text())['dataset']['atom_map']
    data=torch.load(str(args.processed),map_location='cpu',mmap=True,weights_only=False)
    lo,hi=map(int,data['node_idx_array'][args.source_row])
    if hi-lo!=3:raise ValueError('This reference applies only to three atoms')
    symbols=[atom_map[int(i)] for i in data['atom_types'][lo:hi]]
    numbers=[atomic_numbers[s] for s in symbols];charges=data['atom_charges'][lo:hi]
    if ((charges<=-2)|(charges>=3)).any():raise ValueError('Clipped-boundary charge excluded')
    charge=int(charges.sum());spin=1+(sum(numbers)-charge)%2
    scales=[.5,1.,math.sqrt(args.kT/args.restraint)]
    root=Path(__file__).resolve().parents[2]
    report={'complete':False,'scope':__doc__,'source_row':args.source_row,'symbols':symbols,'charge':charge,
        'spin_multiplicity':spin,'electronic_state':'declared minimum electron-parity state, matching the current pilot',
        'configuration':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'proposal_scales_A':scales,'intrinsic_dimension':6,'oracle_sha256':sha(args.oracle),
        'source_sha256':{str(path):sha(path) for path in [Path(__file__).resolve(),root/'cfm_mol/triatomic_reference.py',root/'cfm_mol/energy_oracle.py',root/'scripts/research/oracle_worker.py']},
        'target':'eSEN + restraint/2*sum_i||x_i||^2 on orthonormal COM-free H',
        'normalization_assumption':'eSEN bounded below and harmonic confinement positive',
        'rotation_reduction':'uses architectural rotation invariance; finite-precision variation checked separately',
        'rows':[],'convergence_certified':False}
    write(output,report);start=time.perf_counter()
    with EnergyOracle(args.oracle_python,root/'scripts/research/oracle_worker.py',args.oracle,
        numbers=numbers,charge=charge,spin_multiplicity=spin,device=args.oracle_device) as oracle:
        check=gaussian_triatomic_shapes(3,1.,9027)
        rotation=random_rotations(len(check),torch.Generator().manual_seed(9087)).numpy()
        e0,_=oracle.evaluate(check);er,_=oracle.evaluate(check@rotation)
        rotation_error=float((e0-er).abs().max())
        report['rotation_energy_check_max_eV']=rotation_error
        if rotation_error>1e-3:raise ValueError('Energy rotation check failed; refusing reduced integration')
        for repeat in range(args.scrambles):
            geometries=[];energies=[]
            for component,scale in enumerate(scales):
                x=gaussian_triatomic_shapes(args.power,scale,19027+repeat*1009+component*7001)
                e=[]
                for begin in range(0,len(x),args.batch):
                    value,_=oracle.evaluate(x[begin:begin+args.batch]);e.extend(value.tolist())
                geometries.append(x);energies.append(np.asarray(e))
                print(json.dumps({'repeat':repeat,'component':component,'evaluated':oracle.evaluated,'seconds':time.perf_counter()-start}),flush=True)
            row={'scramble':repeat,'resolutions':[]}
            for power in range(max(3,args.power-3),args.power+1):
                n=2**power;x=np.concatenate([a[:n] for a in geometries]);e=np.concatenate([a[:n] for a in energies])
                logw=-(e+args.restraint/2*(x*x).sum((1,2)))/args.kT-centered_gaussian_mixture_log_density(x,scales)
                weights=np.exp(logw-logsumexp(logw));moments={key:float(weights@value) for key,value in invariant_observables(x).items()}
                row['resolutions'].append({'particles':len(x),'log_normalizer_estimate':float(logsumexp(logw)-math.log(len(x))),
                    'weight_ess':float(1/(weights@weights)),'maximum_weight':float(weights.max()),'moments':moments})
            np.savez_compressed(args.out/f'scramble_{repeat}.npz',positions=np.stack(geometries),energy_eV=np.stack(energies))
            report['rows'].append(row);report['oracle_evaluations']=oracle.evaluated;write(output,report)
        report['aggregate']=[]
        for j in range(len(report['rows'][0]['resolutions'])):
            resolutions=[row['resolutions'][j] for row in report['rows']]
            logs=np.array([r['log_normalizer_estimate'] for r in resolutions]);relative=np.exp(logs-logs.max())
            report['aggregate'].append({'particles_per_scramble':resolutions[0]['particles'],
                'log_of_mean_normalizer_estimate':float(logsumexp(logs)-math.log(len(logs))),
                'relative_standard_error_across_scrambles':float(relative.std(ddof=1)/math.sqrt(len(relative))/relative.mean()),
                'log_estimate_range':float(logs.max()-logs.min()),
                'mean_weight_ess':float(np.mean([r['weight_ess'] for r in resolutions]))})
        report.update(complete=True,seconds=time.perf_counter()-start,oracle_evaluations=oracle.evaluated)
        write(output,report);print(json.dumps(report['aggregate']),flush=True)


if __name__=='__main__':main()
