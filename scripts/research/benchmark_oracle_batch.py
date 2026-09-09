#!/usr/bin/env python3
"""Check batched fairchem results against serial ASE on fixed saved geometries."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from cfm_mol.energy_oracle import EnergyOracle


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples',type=Path,required=True)
    p.add_argument('--oracle-python',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--particles',type=int,default=32)
    p.add_argument('--batch-size',type=int,default=16)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    if min(args.particles,args.batch_size)<1:raise ValueError('Positive counts required')
    source=torch.load(str(args.samples),map_location='cpu',weights_only=False)
    positions=source['positions'][:args.particles];condition=source['condition']
    root=Path(__file__).resolve().parents[2]
    results={};timing={}
    for batch in [1,args.batch_size]:
        with EnergyOracle(args.oracle_python,root/'scripts/research/oracle_worker.py',args.checkpoint,
                numbers=condition['numbers'],charge=condition['charge'],spin_multiplicity=condition['spin_multiplicity'],
                batch_size=batch,timeout_seconds=120.) as oracle:
            oracle.evaluate(positions[:min(batch,len(positions))])
            start=time.perf_counter();energy,force=oracle.evaluate(positions)
            timing[str(batch)]=time.perf_counter()-start
            results[batch]=(energy,force)
    errors=[float((a-b).abs().max()) for a,b in zip(results[1],results[args.batch_size])]
    passed=errors[0]<2e-4 and errors[1]<2e-3
    report={'complete':True,'passed':passed,'scope':'Numerical consistency and warm throughput; not physical accuracy.',
        'condition':condition,'particles':len(positions),'batch_size':args.batch_size,'seconds':timing,
        'speedup':timing['1']/timing[str(args.batch_size)],
        'max_energy_error_eV':errors[0],'max_force_error_eV_A':errors[1],
        'tolerances':{'energy_eV':2e-4,'force_eV_A':2e-3},
        'sha256':{str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [args.samples,args.checkpoint,Path(__file__),root/'scripts/research/oracle_worker.py']}}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    if not passed:raise RuntimeError('Batched oracle failed serial consistency screen')


if __name__=='__main__':main()
