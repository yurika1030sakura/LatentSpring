#!/usr/bin/env python3
"""Submit a bounded experiment using committed, immutable source files."""
import argparse
import datetime
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('name')
p.add_argument('--launcher', default='scripts/research/runtime_smoke.slurm')
p.add_argument('--gpu-hours', type=float, default=1.)
p.add_argument('--config')
p.add_argument('--seed', type=int)
p.add_argument('--launcher-arg', action='append', default=[])
p.add_argument('--submit', action='store_true')
args = p.parse_args()
root = Path(__file__).resolve().parents[2]
if not args.name.replace('_', '').replace('-', '').isalnum():
    raise ValueError('Use an alphanumeric experiment name')
rev = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip()
subprocess.run(['git','cat-file','-e',f'{rev}:{args.launcher}'],cwd=root,check=True)
if subprocess.run(['git','diff','--quiet','HEAD','--',args.launcher],cwd=root).returncode:
    raise ValueError('Commit launcher changes before submitting their source snapshot')
snapshot = root/'runs/source_snapshots'/rev
out = root/'runs'/args.name
command = ['sbatch', '--parsable', args.launcher, str(snapshot), str(out)]
if args.config is not None or args.seed is not None:
    if args.launcher_arg:
        raise ValueError('Use named smoke arguments or explicit launcher arguments, not both')
    command += [args.config or 'configs/research/runtime_smoke.yaml', str(9001 if args.seed is None else args.seed)]
command += args.launcher_arg
if not args.submit:
    print(json.dumps({'command':command, 'source_commit':rev}, indent=2))
else:
    if out.exists():
        raise ValueError('Output directory already exists; use a new experiment name')
    if not snapshot.exists():
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        raw = subprocess.check_output(['git', 'archive', rev], cwd=root)
        # Publish only a complete snapshot. Concurrent submissions must never
        # observe a directory whose archive is still being extracted.
        with tempfile.TemporaryDirectory(prefix=f'.{rev}.',dir=snapshot.parent) as stage:
            staged=Path(stage)/'source'
            staged.mkdir()
            with tarfile.open(fileobj=io.BytesIO(raw), mode='r:') as t:
                t.extractall(staged)
            try:
                staged.rename(snapshot)
            except OSError:
                if not snapshot.is_dir():raise
    if not (snapshot/args.launcher).is_file():
        raise ValueError('Incomplete source snapshot; refusing submission')
    # Use the launcher from the same snapshot, not the mutable working tree.
    command[2] = str(snapshot/args.launcher)
    out.mkdir(parents=True)
    submitted = subprocess.run(command, cwd=root, capture_output=True, text=True)
    job = submitted.stdout.strip() if submitted.returncode == 0 else None
    record = {'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'job_id':job, 'name':args.name, 'source_commit':rev,
        'snapshot':str(snapshot), 'output':str(out),
        'max_gpu_hours':args.gpu_hours, 'launcher':args.launcher, 'config':args.config, 'seed':args.seed,
        'launcher_args':args.launcher_arg,
        'submission_state':'submitted' if submitted.returncode == 0 else 'rejected',
        'submission_error':submitted.stderr.strip() if submitted.returncode else None}
    with (root/'research/jobs.jsonl').open('a') as f:
        f.write(json.dumps(record)+'\n')
    (out/'submission.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))
    if submitted.returncode:
        raise SystemExit(submitted.returncode)
