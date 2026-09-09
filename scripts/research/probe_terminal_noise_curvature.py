#!/usr/bin/env python3
"""Probe local COM-free potential curvature using finite oracle-force differences.

Fixed leading sample indices are used, not selected eigenvalues or energies.
Two difference scales and antisymmetry diagnostics do not constitute a rigorous
global Hessian bound. No equilibrium assumption is made about the input samples.
"""
import argparse
import json
import math
from pathlib import Path

import torch

from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.nonequilibrium import centered_orthonormal_basis
from molecular_tempered_pilot import sha,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--oracle-python',type=Path,required=True);p.add_argument('--oracle',type=Path,required=True)
    p.add_argument('--count',type=int,default=4);p.add_argument('--h',type=float,nargs='+',default=[.01,.003])
    p.add_argument('--restraint',type=float,default=.1)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    if args.count<1 or args.restraint<=0 or any(not math.isfinite(h) or h<=0 for h in args.h):raise ValueError('Invalid diagnostic settings')
    if not json.loads((args.samples.parent/'results.json').read_text())['complete']:raise ValueError('Require completed source run')
    data=torch.load(str(args.samples),map_location='cpu',weights_only=False)
    if args.count>len(data['positions']):raise ValueError('Not enough fixed source samples')
    condition=data['condition'];numbers=condition['numbers'];n=len(numbers);basis=centered_orthonormal_basis(n)
    directions=torch.einsum('nk,ad->kand',basis,torch.eye(3,dtype=torch.float64)).reshape(3*(n-1),n,3)
    flattened=directions.flatten(1);identity=torch.eye(len(directions),dtype=torch.float64)
    torch.testing.assert_close(flattened@flattened.T,identity,rtol=0,atol=1e-12)
    root=Path(__file__).resolve().parents[2]
    report={'complete':False,'scope':__doc__,'source_samples':str(args.samples.resolve()),
        'source_samples_sha256':sha(args.samples),'oracle_checkpoint_sha256':sha(args.oracle),
        'script_sha256':sha(Path(__file__)),'condition':condition,'restraint_eV_A2':args.restraint,'rows':[]}
    args.out.parent.mkdir(parents=True,exist_ok=True);write_json(args.out,report)
    with EnergyOracle(args.oracle_python,root/'scripts/research/oracle_worker.py',args.oracle,
        numbers=numbers,charge=condition['charge'],spin_multiplicity=condition['spin_multiplicity'],batch_size=16) as oracle:
        for index,x in enumerate(data['positions'][:args.count].double()):
            estimates=[]
            for h in args.h:
                probes=torch.cat([x[None]+h*directions,x[None]-h*directions]);_,force=oracle.evaluate(probes)
                differential=-(force[:len(directions)]-force[len(directions):])/(2*h)
                matrix=differential.flatten(1)@flattened.T+args.restraint*identity
                values=torch.linalg.eigvalsh((matrix+matrix.T)/2)
                estimates.append({'h_A':h,'max_eigenvalue_eV_A2':float(values.max()),
                    'eigenvalues_eV_A2':values.tolist(),'antisymmetric_norm':float((matrix-matrix.T).norm())})
            report['rows'].append({'source_sample_id':index,'estimates':estimates});write_json(args.out,report)
        report.update(complete=True,oracle_queries=oracle.evaluated)
        write_json(args.out,report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
