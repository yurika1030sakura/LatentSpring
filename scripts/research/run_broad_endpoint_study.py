"""Fit both endpoint controls, then generate raw coordinates on the frozen panel."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from scripts.research.audit_generator_output_support import assess
from scripts.research.train_electronic_fm import sha
from scripts.research.run_matched_generators import write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['project','protocol','teacher','out']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--seed-index',type=int,choices=[0,1],required=True);a=p.parse_args()
    spec=json.loads(a.protocol.read_text());ph=sha(a.protocol);assert spec['frozen']
    seed=spec['training_seeds'][a.seed_index];evaluation_seed=spec['evaluation_seeds'][a.seed_index]
    a.out.mkdir(parents=True,exist_ok=False);conditions=a.out/'conditions.json';write(conditions,dict(rows=spec['test_rows']))
    checkpoint=a.project/spec['checkpoint'];assert sha(checkpoint)==spec['checkpoint_sha256']
    parent=torch.load(checkpoint,map_location='cpu',weights_only=False)
    parent['research_protocol']={**parent['research_protocol'],'direct_endpoint_training':dict(
        mode='unadapted_parent',teacher_temperature_K=None,model_kT_eV=1.,terminal_noise_std_A=0.,protocol_sha256=ph)}
    base=a.out/'base.ckpt';torch.save(parent,base);del parent
    for method in ['reference_ft','relaxed_ft']:
        subprocess.run([sys.executable,'-u','-m','scripts.research.train_broad_endpoints',
            '--project',str(a.project),'--protocol',str(a.protocol),'--bank',str(a.teacher/'bank.pt'),
            '--out',str(a.out/method),'--seed',str(seed),'--method',method],check=True)
    rows=[]
    for method in spec['methods']:
        checkpoint=base if method=='base' else a.out/method/'last.ckpt'
        destination=a.out/(method+'_raw')
        subprocess.run([sys.executable,'-u','-m','scripts.research.generate_weighted_minima',
            '--checkpoint',str(checkpoint),'--config',str(a.project/spec['config']),
            '--conditions',str(conditions),'--out',str(destination),'--device','cuda',
            '--samples',str(spec['samples_per_condition']),'--midpoint-steps',str(spec['midpoint_steps']),
            '--seed',str(evaluation_seed),'--allow-legacy-pickle'],check=True)
        report=json.loads((destination/'generation.json').read_text());assert report['complete'] and report['primitive_network_calls_per_attempt']==128
        for row in report['rows']:
            file=destination/row['file'];assert sha(file)==row['sha256']
            with np.load(file) as f:x=torch.from_numpy(f['raw_positions'])
            result=assess(x,row['condition'],list(range(len(x))))
            rows.append(dict(arm=method,condition=row['condition'],raw_sha256=row['sha256'],**result))
    write(a.out/'audit.json',dict(complete=True,protocol_sha256=ph,seed=seed,rows=rows,scope=spec['scope']))
    quality=dict(frozen=True,native_protocol_sha256=ph,methods=spec['methods'],replica=seed,
        condition_count=len(spec['test_rows']),samples_per_condition=spec['samples_per_condition'],
        xtb_binary=spec['xtb_binary'],xtb_binary_sha256=spec['xtb_binary_sha256'],xtb=spec['xtb'],
        thresholds=spec['thresholds'],scope=spec['scope'])
    qp=a.out/'physical_protocol.json';write(qp,quality)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    subprocess.run([sys.executable,'-u','-m','scripts.research.score_native_minima',
        '--project',str(a.project),'--run',str(a.out),'--protocol',str(qp),'--out',str(a.out/'xtb')],env=env,check=True)
    result=json.loads((a.out/'xtb/audit.json').read_text());assert result['complete']
    write(a.out/'complete.json',dict(complete=True,protocol_sha256=ph,seed=seed,summary=result['summary'],
        raw_audit_sha256=sha(a.out/'audit.json'),xtb_audit_sha256=sha(a.out/'xtb/audit.json'),
        new_neural_outputs=768,new_gfn2_attempts=768,new_training_steps=4000,new_training_esen_queries=0))
    print(json.dumps(result['summary']),flush=True)


if __name__=='__main__':main()
