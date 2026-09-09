#!/usr/bin/env python3
"""Read saved weights and logged contributions; never infer success from a job ID."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('run',type=Path)
p.add_argument('--expected-steps',type=int,default=20)
args=p.parse_args()
checkpoints=sorted(args.run.rglob('last.ckpt'))
if len(checkpoints)!=1:
    raise RuntimeError('Expected exactly one last checkpoint in this isolated run')
checkpoint=checkpoints[0]
state=torch.load(checkpoint,map_location='cpu',weights_only=False)
values=[v for v in state['state_dict'].values() if v.is_floating_point()]
report={'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    'global_step':state['global_step'],'all_weights_finite':all(bool(torch.isfinite(v).all()) for v in values),
    'scalars':{},'claim':'runtime/optimization validation, not molecular performance'}
for event in args.run.rglob('events.out*'):
    accumulator=EventAccumulator(str(event),size_guidance={'scalars':0});accumulator.Reload()
    for tag in accumulator.Tags()['scalars']:
        if any(x in tag for x in ['L_energy','lambda_2','energy_n_groups','trace_noise','nan_skip','outlier','squared_replica']):
            rows=accumulator.Scalars(tag);ys=[row.value for row in rows]
            report['scalars'][tag]={'n':len(ys),'min':min(ys),'max':max(ys),'mean':sum(ys)/len(ys),
                'negative_count':sum(y<0 for y in ys),'positive_count':sum(y>0 for y in ys)}
if state['global_step']!=args.expected_steps or not report['all_weights_finite']:
    raise RuntimeError('Training step or finite-weight validation failed')
if 'train_lambda_2' in report['scalars'] and report['scalars']['train_lambda_2']['positive_count']==0:
    raise RuntimeError('Energy contribution was skipped at every logged step')
(args.run/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(report,indent=2))
