#!/usr/bin/env python3
"""Join immutable run provenance with live Slurm allocation and completed query counts."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess

from molecular_tempered_pilot import sha,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='append',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    submissions=[json.loads((run/'submission.json').read_text()) for run in args.run]
    ids=[row['job_id'].split(';')[0] for row in submissions]
    command=['sacct','-j',','.join(ids),'-X','--parsable2','--noheader',
        '--format=JobIDRaw,State,ElapsedRaw,AllocTRES%200']
    raw=subprocess.check_output(command,text=True);accounting={}
    for line in raw.splitlines():
        parts=line.split('|')
        if len(parts)<4:raise ValueError('Unexpected sacct row')
        job,state,elapsed,tres=parts[:4]
        accounting[job]={'state':state,'allocation_elapsed_seconds':int(elapsed),'allocated_tres':tres}
    rows=[]
    for run,submission,job in zip(args.run,submissions,ids):
        candidates=[run/'results.json',run/'sampling/results.json'];path=next((p for p in candidates if p.exists()),None)
        result=json.loads(path.read_text()) if path is not None else {}
        if job not in accounting:raise ValueError(f'Missing scheduler accounting for {job}')
        config=result.get('configuration',{});completed=result.get('complete',False)
        row={'run':str(run.resolve()),'job_id':job,'source_commit':submission['source_commit'],
            'submission_sha256':sha(run/'submission.json'),**accounting[job],
            'numerical_run_complete':completed,'reported_program_seconds':result.get('seconds'),
            'oracle_evaluations_this_run':result.get('oracle_evaluations') if completed else None,
            'cumulative_oracle_evaluations':result.get('cumulative_oracle_evaluations') if completed else None,
            'updates_this_run':(0 if result.get('gradient_diagnostics') else config.get('steps',0)-result.get('training_start_step',0)) if completed and 'training' in result and 'source_training_steps' not in result else None,
            'source_training_steps':result.get('source_training_steps'),
            'evaluation_particles':config.get('eval_particles'),'condition':result.get('condition'),
            'result_path':str(path.resolve()) if path is not None else None,'result_sha256':sha(path) if path is not None else None}
        rows.append(row)
    write_json(args.out,{'complete':True,'scope':__doc__,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scheduler_command':command,'raw_scheduler_output':raw,'rows':rows,
        'limitations':['Slurm allocation time includes pipeline setup and possible downstream assessment; program timers have narrower scopes.',
            'GPU allocation labels include MIG slices and full devices; their seconds are not interchangeable full-GPU hours.',
            'CPU counts describe reservations, not measured CPU utilization.',
            'Equal oracle queries do not establish equal wall time or hardware cost.',
            'Upstream FM and eSEN pretraining costs are not reconstructed here; comparisons condition on the shared checkpoints.',
            'Do not add cumulative query counts to their own ancestor-run counts.']})
    for row in rows:print(row['run'].split('/')[-1],row['state'],row['allocation_elapsed_seconds'],row['oracle_evaluations_this_run'])


if __name__=='__main__':main()
