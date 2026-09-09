#!/usr/bin/env python3
"""Independent GFN2-xTB strain check, with every attempted sample retained.

Archived OMol spin labels are unavailable. This development evaluator declares
minimum electron-parity spin (UHF 0/1) for both single point and relaxation;
it does not claim the original DFT electronic state. Strain alone does not
check identity, connectivity, diversity or Boltzmann populations.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

from checkpoint_panel import write_json

HARTREE_EV=27.211386245988
BOHR_A=.529177210903
ENERGY=re.compile(r'TOTAL ENERGY\s+(-?\d+(?:\.\d+)?(?:[EeDd][+-]?\d+)?)')


def gradient_norm(text,n_atoms):
    """Turbomole gradient: the last N triples are gradients, not coordinates."""
    triples=[]
    for line in text.splitlines():
        parts=line.replace('D','E').replace('d','e').split()
        if len(parts)!=3:continue
        try:triples.append([float(p) for p in parts])
        except ValueError:continue
    if len(triples)<n_atoms:raise ValueError('Incomplete gradient block')
    norms=[math.sqrt(sum(v*v for v in row))*HARTREE_EV/BOHR_A for row in triples[-n_atoms:]]
    if not all(math.isfinite(v) for v in norms):raise ValueError('Non-finite gradient')
    return max(norms)


def invoke(command,work,timeout,phase):
    try:
        result=subprocess.run(command,cwd=work,capture_output=True,text=True,timeout=timeout)
    except subprocess.TimeoutExpired:
        return {'ok':False,'failure':phase+'_timeout'}
    (work/(phase+'.stdout')).write_text(result.stdout)
    (work/(phase+'.stderr')).write_text(result.stderr)
    matches=ENERGY.findall(result.stdout)
    energy=float(matches[-1].replace('D','E')) if matches else None
    if result.returncode or energy is None or not math.isfinite(energy):
        return {'ok':False,'failure':phase+'_exit_'+str(result.returncode),'energy_present':energy is not None}
    return {'ok':True,'energy_eV':energy*HARTREE_EV,'stdout':result.stdout}


def evaluate(task,binary,out,max_cycles):
    # rdkit is already a FlowMol dependency; it supplies only a periodic table.
    from rdkit import Chem
    periodic=Chem.GetPeriodicTable()
    symbols=task['symbols'];charge=task['charge_recorded']
    electrons=sum(periodic.GetAtomicNumber(symbol) for symbol in symbols)-charge
    unpaired=electrons%2
    key=f"{task['arm']}_{task['validation_index']}_{task['sample_id']}"
    work=out/'details'/key;work.mkdir(parents=True,exist_ok=False)
    xyz=work/'input.xyz'
    xyz.write_text(str(len(symbols))+'\nminimum parity spin development evaluation\n'+''.join(
        f'{symbol} {x:.12g} {y:.12g} {z:.12g}\n' for symbol,(x,y,z) in zip(symbols,task['positions'])))
    base={k:v for k,v in task.items() if k not in ['positions','symbols']}
    base.update(n_atoms=len(symbols),uhf_assumed=unpaired,spin_metadata_available=False,success=False)
    start=time.monotonic()
    common=[binary,str(xyz),'--gfn','2','--chrg',str(charge),'--uhf',str(unpaired)]
    sp=invoke(common+['--grad'],work,90,'single_point')
    if not sp['ok']:return dict(base,**sp,seconds=time.monotonic()-start)
    base['initial_energy_eV']=sp['energy_eV']
    gradient=work/'gradient'
    if gradient.exists():
        try:base['initial_max_force_norm_eV_A']=gradient_norm(gradient.read_text(),len(symbols))
        except ValueError as exc:base['gradient_error']=str(exc)
    opt=invoke(common+['--opt','tight','--cycles',str(max_cycles)],work,180,'relaxation')
    if not opt['ok']:return dict(base,**opt,seconds=time.monotonic()-start)
    converged=bool(re.search(r'GEOMETRY OPTIMIZATION CONVERGED',opt['stdout'],re.I))
    base.update(converged=converged,relaxed_energy_eV=opt['energy_eV'],
        strain_eV=sp['energy_eV']-opt['energy_eV'],success=converged,
        failure=None if converged else 'optimization_not_converged',seconds=time.monotonic()-start)
    return base


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--max-cycles',type=int,default=200)
    args=p.parse_args();binary=shutil.which('xtb')
    if binary is None:raise FileNotFoundError('xtb binary not found')
    source=json.loads(args.samples.read_text())
    if not source['complete']:raise ValueError('Sampling panel is incomplete')
    references={r['validation_index']:r for r in source['references']}
    tasks=[]
    for index,reference in references.items():
        tasks.append(dict(reference,arm='reference',sample_id=0))
    for arm in source['arms']:
        for sample in arm['samples']:
            ref=references[sample['validation_index']]
            tasks.append(dict(sample,arm=arm['name'],symbols=ref['symbols'],charge_recorded=ref['charge_recorded']))
    report={'claim':'independent GFN2-xTB conditional sample strain development check, single training seed',
        'source_samples':str(args.samples),'source_sha256':hashlib.sha256(args.samples.read_bytes()).hexdigest(),
        'xtb_binary':binary,'xtb_sha256':hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
        'electronic_state':'stored unclipped charge; declared minimal electron-parity spin, original spin unknown',
        'limitations':['Strain does not establish valid connectivity or improved Boltzmann sampling',
            'Old validation exposure remains','No cherry-picking of failed evaluations'],
        'requested':len(tasks),'rows':[],'complete':False}
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'xtb.json').exists():raise ValueError('Refusing to overwrite evaluation')
    write_json(args.out/'xtb.json',report)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(evaluate,task,binary,args.out,args.max_cycles):task for task in tasks}
        for future in as_completed(futures):
            row=future.result();report['rows'].append(row)
            write_json(args.out/'xtb.json',report)
            print(json.dumps({k:row.get(k) for k in ['arm','validation_index','sample_id','success','strain_eV','failure','seconds']}),flush=True)
    report['rows'].sort(key=lambda row:(row['arm'],row['validation_index'],row['sample_id']))
    report['complete']=True
    write_json(args.out/'xtb.json',report)


if __name__=='__main__':main()
