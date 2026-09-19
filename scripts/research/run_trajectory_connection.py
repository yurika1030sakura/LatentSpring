"""Fit and compare FM and parent-trajectory physical correction heads."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from cfm_mol.physical_connection import PhysicalConnection
from scripts.research.audit_generator_output_support import assess
from scripts.research.run_matched_generators import write
from scripts.research.train_electronic_fm import sha


def sampling_checkpoint(source,destination,spec,seed):
    state=torch.load(source,map_location='cpu',weights_only=False)
    if not state['research_protocol'].get('physical_connection'):
        torch.manual_seed(seed);head=PhysicalConnection(**spec['physical_connection'])
        state['state_dict']={**state['state_dict'],**{'vector_field.physical_connection.'+n:v for n,v in head.state_dict().items()}}
        state['research_protocol']={**state['research_protocol'],'physical_connection':spec['physical_connection']}
    if not state['research_protocol'].get('direct_endpoint_training'):
        state['research_protocol']['direct_endpoint_training']=dict(mode='unadapted_parent',teacher_temperature_K=None,model_kT_eV=1.,terminal_noise_std_A=0.)
    torch.save(state,destination)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['project','protocol','out']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--seed-index',type=int,choices=[0,1],required=True);a=p.parse_args();spec=json.loads(a.protocol.read_text());ph=sha(a.protocol)
    seed=spec['training_seeds'][a.seed_index];es=spec['evaluation_seeds'][a.seed_index]
    bank=a.project/spec['teacher_bank'];assert sha(bank)==spec['teacher_bank_sha256']
    from scripts.research.audit_trajectory_teacher import audit
    audit(a.project,(a.project/spec['trajectory_bank']).parent,spec,ph)
    a.out.mkdir(parents=True,exist_ok=False);conditions=a.out/'conditions.json';write(conditions,dict(rows=spec['test_rows']))
    for method in spec['methods'][1:]:
        if method in ['connection_force','connection_work']:
            subprocess.run([sys.executable,'-u','-m','scripts.research.train_trajectory_connection','--project',str(a.project),
                '--protocol',str(a.protocol),'--out',str(a.out/method),'--seed',str(seed),'--method',method],check=True)
            continue
        subprocess.run([sys.executable,'-u','-m','scripts.research.train_broad_endpoints','--project',str(a.project),
            '--protocol',str(a.protocol),'--bank',str(bank),'--out',str(a.out/method),'--seed',str(seed),'--method',method],check=True)
    sampling=a.out/'sampling';sampling.mkdir();sources={};rows=[]
    for method in spec['methods']:
        original=a.project/spec['checkpoint'] if method=='base' else a.out/method/'last.ckpt'
        checkpoint=sampling/(method+'.ckpt');sampling_checkpoint(original,checkpoint,spec,es)
        sources[method]=dict(original_sha256=sha(original),sampling_sha256=sha(checkpoint))
        folder=a.out/(method+'_raw')
        subprocess.run([sys.executable,'-u','-m','scripts.research.generate_weighted_minima','--checkpoint',str(checkpoint),
            '--config',str(a.project/spec['config']),'--conditions',str(conditions),'--out',str(folder),'--device','cuda',
            '--samples',str(spec['samples_per_condition']),'--midpoint-steps',str(spec['midpoint_steps']),
            '--seed',str(es),'--allow-legacy-pickle'],check=True)
        report=json.loads((folder/'generation.json').read_text())
        assert report['complete'] and report['backbone_calls_per_attempt']==128 and report['connection_calls_per_attempt']==64 and report['primitive_network_calls_per_attempt']==192
        for row in report['rows']:
            file=folder/row['file'];assert sha(file)==row['sha256']
            with np.load(file) as f:x=torch.from_numpy(f['raw_positions'])
            rows.append(dict(arm=method,condition=row['condition'],raw_sha256=row['sha256'],**assess(x,row['condition'],list(range(len(x))))))
    write(a.out/'audit.json',dict(complete=True,protocol_sha256=ph,seed=seed,rows=rows,scope=spec['scope']))
    qp=a.out/'physical_protocol.json'
    write(qp,dict(frozen=True,native_protocol_sha256=ph,methods=spec['methods'],replica=seed,condition_count=16,samples_per_condition=16,
        xtb_binary=spec['xtb_binary'],xtb_binary_sha256=spec['xtb_binary_sha256'],xtb=spec['xtb'],thresholds=spec['thresholds'],scope=spec['scope']))
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    subprocess.run([sys.executable,'-u','-m','scripts.research.score_native_minima','--project',str(a.project),
        '--run',str(a.out),'--protocol',str(qp),'--out',str(a.out/'xtb')],env=env,check=True)
    result=json.loads((a.out/'xtb/audit.json').read_text());assert result['complete']
    write(a.out/'complete.json',dict(complete=True,protocol_sha256=ph,seed=seed,sampling_checkpoints=sources,
        raw_audit_sha256=sha(a.out/'audit.json'),xtb_audit_sha256=sha(a.out/'xtb/audit.json'),summary=result['summary'],
        new_neural_outputs=1024,new_training_steps=6000,new_gfn2_attempts=1024,new_esen_queries=0))


if __name__=='__main__':main()
