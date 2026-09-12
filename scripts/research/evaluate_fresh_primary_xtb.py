#!/usr/bin/env python3
"""Frozen-geometry GFN2 cross-potential check of all primary paired endpoints."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import subprocess
import time
from ase.data import chemical_symbols
from cfm_mol.xtb_singlepoint import parse_singlepoint
from scripts.research.evaluate_chemical_policy import sha,write


def run_task(task,condition,binary,out,protocol):
    work=out/'details'/task['task_id'];work.mkdir(parents=True,exist_ok=False)
    numbers=condition['numbers'];charge=condition['charge'];spin=condition['spin_multiplicity']
    electrons=sum(numbers)-charge;unpaired=spin-1
    if unpaired<0 or unpaired>electrons or (electrons-unpaired)%2:
        raise ValueError('Recorded spin violates electron-count parity')
    xyz=work/'input.xyz'
    xyz.write_text(str(len(numbers))+'\nfixed primary endpoint; original charge and spin\n'+''.join(
        chemical_symbols[z]+' '+' '.join(f'{float(v):.16g}' for v in x)+'\n' for z,x in zip(numbers,task['positions'])))
    input_sha=sha(xyz)
    command=[str(binary),str(xyz.resolve()),'--gfn','2','--chrg',str(charge),'--uhf',str(unpaired),
        '--acc',str(protocol['accuracy']),'--grad']
    started=time.perf_counter()
    try:
        p=subprocess.run(command,cwd=work,capture_output=True,text=True,timeout=protocol['timeout_seconds'])
        stdout,stderr,code=p.stdout,p.stderr,p.returncode
        gradient=(work/'gradient').read_text() if (work/'gradient').exists() else ''
        result=parse_singlepoint(stdout,stderr,code,gradient,len(numbers))
    except subprocess.TimeoutExpired as exc:
        def decoded(value):return value.decode(errors='replace') if isinstance(value,bytes) else (value or '')
        stdout,stderr=decoded(exc.stdout),decoded(exc.stderr);code=None
        result=dict(success=False,failure='single_point_timeout')
    (work/'stdout.txt').write_text(stdout);(work/'stderr.txt').write_text(stderr)
    assert sha(xyz)==input_sha
    result.update(task_id=task['task_id'],method=task['method'],replica=task['replica'],parent_id=task['parent_id'],
        inversion_check=task['inversion_check'],command=command,returncode=code,input_xyz_sha256=input_sha,
        stdout_sha256=sha(work/'stdout.txt'),stderr_sha256=sha(work/'stderr.txt'),
        gradient_sha256=sha(work/'gradient') if (work/'gradient').exists() else None,
        original_charge=charge,original_spin_multiplicity=spin,uhf=unpaired,seconds=time.perf_counter()-started)
    if task['inversion_check']:result['mirrored_task_id']=task['mirrored_task_id']
    if result['success']:
        restraint=protocol['restraint_eV_A2']/2*sum(v*v for x in task['positions'] for v in x)
        result['restrained_potential_eV']=result['energy_eV']+restraint
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','tasks','binary','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=8)
    args=p.parse_args();root=args.project
    pp=root/'research/evidence/fresh_primary_xtb_protocol_v1.json';protocol=json.loads(pp.read_text())
    assert sha(args.binary)==protocol['xtb_binary_sha256']
    data=json.loads(args.tasks.read_text());assert data['complete'] and data['protocol_sha256']==sha(pp)
    assert len(data['tasks'])==protocol['maximum_single_point_attempts']
    assert len({t['task_id'] for t in data['tasks']})==len(data['tasks'])
    args.out.mkdir(parents=True,exist_ok=True);output=args.out/'results.json'
    if output.exists():raise FileExistsError(output)
    version=subprocess.run([str(args.binary),'--version'],capture_output=True,text=True,check=True)
    report=dict(complete=False,protocol_sha256=sha(pp),tasks_sha256=sha(args.tasks),xtb_binary_sha256=sha(args.binary),
        version_output=version.stdout+version.stderr,condition=data['condition'],requested_attempts=len(data['tasks']),
        rows=[],new_esen_queries=0,scientific_submission_ready=False)
    write(output,report)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run_task,t,data['condition'],args.binary,args.out,protocol) for t in data['tasks']]
        for future in as_completed(futures):
            row=future.result();report['rows'].append(row);write(output,report)
            if len(report['rows'])%32==0:print(json.dumps(dict(completed=len(report['rows']),failed=sum(not r['success'] for r in report['rows']))),flush=True)
    report['rows'].sort(key=lambda r:r['task_id'])
    report.update(complete=True,attempted=len(report['rows']),successful=sum(r['success'] for r in report['rows']),
        failed=sum(not r['success'] for r in report['rows']),scope='GFN2 single points and gradients at frozen coordinates, not geometry optimization, DFT verification or equilibrium validation.')
    write(output,report)


if __name__=='__main__':main()
