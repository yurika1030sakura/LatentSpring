"""Fixed follow-up of direct-target models on composition-disjoint molecules."""
import argparse
import copy
import hashlib
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


def freeze(root,path):
    assert not path.exists()
    native=json.loads((root/'research/evidence/weighted_minima_native_v1.json').read_text())
    old=json.loads((root/'research/evidence/fresh_physics_s0_v1.json').read_text())
    panelpath=root/old['condition_manifest'];panel=json.loads(panelpath.read_text())
    rows=[panel['rows'][i] for i in old['conditions']]
    rows.sort(key=lambda c:hashlib.sha256(('native-minima-transfer-v1:'+c['composition_hex']).encode()).hexdigest())
    chosen=rows[:8];trained=json.loads((root/native['conditions']).read_text())['rows']
    key=lambda c:(tuple(sorted(c['atomic_numbers'])),c['charge'],c['spin_multiplicity'])
    assert not {key(c) for c in chosen}&{key(c) for c in trained}
    models={};inputs={native['config']:sha(root/native['config'])}
    for seed in [260938,260939,260940]:
        run=root/('runs/weighted_minima_native_v1/results' if seed==260938 else f'runs/weighted_minima_repeats_v1/results/s{seed}')
        models[str(seed)]={}
        for arm in ['base','proposal_min','full_work_min']:
            checkpoint=run/('base.ckpt' if arm=='base' else arm+'/last.ckpt')
            name=str(checkpoint.relative_to(root));models[str(seed)][arm]=name;inputs[name]=sha(checkpoint)
    write(path,dict(frozen=True,config=native['config'],models=models,inputs=inputs,rows=chosen,
        samples_per_condition=16,evaluation_seed=49601,source_panel=str(panelpath.relative_to(root)),
        source_panel_sha256=sha(panelpath),primary_threshold_eV_A=5.,thresholds=[.1,.5,1.,2.,5.,10.],
        primary_contrasts=['full_work_min minus base','proposal_min minus base','full_work_min minus proposal_min'],
        selection='First8 of the existing24 held-out compositions by salted hash; no geometry, energy or model outcome used for this subset choice. All three fitted seeds and both target controls retained.',
        scope='Fixed exploratory follow-up on previously evaluated composition-disjoint molecules. Same checkpoint, sampler,128 calls and zero terminal noise. No tuning or new training. This is not a new untouched confirmation panel.'))


def run(root,protocol,out):
    spec=json.loads(protocol.read_text());assert spec['frozen'];out.mkdir(parents=True,exist_ok=False)
    for p,h in spec['inputs'].items():assert sha(root/p)==h
    conditions=out/'conditions.json';write(conditions,dict(rows=spec['rows']))
    summaries={};xtb=json.loads((root/'research/evidence/weighted_minima_native_xtb_v1.json').read_text())
    for offset,(seed,models) in enumerate(spec['models'].items()):
        directory=out/f's{seed}';directory.mkdir();rows=[]
        for arm,path in models.items():
            destination=directory/(arm+'_raw')
            subprocess.run([sys.executable,'-u','-m','scripts.research.generate_weighted_minima',
                '--checkpoint',str(root/path),'--config',str(root/spec['config']),
                '--conditions',str(conditions),'--out',str(destination),'--device','cuda',
                '--samples',str(spec['samples_per_condition']),'--seed',str(spec['evaluation_seed']+offset),
                '--allow-legacy-pickle'],check=True)
            report=json.loads((destination/'generation.json').read_text());assert report['complete']
            for row in report['rows']:
                file=destination/row['file'];assert sha(file)==row['sha256']
                with np.load(file) as f:x=torch.from_numpy(f['raw_positions'])
                rows.append(dict(arm=arm,condition=row['condition'],raw_sha256=row['sha256'],**assess(x,row['condition'],list(range(len(x))))))
        write(directory/'audit.json',dict(complete=True,protocol_sha256=sha(protocol),rows=rows,scope=spec['scope']))
        quality={k:v for k,v in xtb.items() if k!='native_audit_sha256'}
        quality.update(native_protocol_sha256=sha(protocol),condition_count=len(spec['rows']),samples_per_condition=spec['samples_per_condition'],scope=spec['scope'])
        qp=directory/'physical_protocol.json';write(qp,quality)
        environment=os.environ.copy();environment.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        subprocess.run([sys.executable,'-u','-m','scripts.research.score_native_minima','--project',str(root),
            '--run',str(directory),'--protocol',str(qp),'--out',str(directory/'xtb')],check=True,env=environment)
        result=json.loads((directory/'xtb/audit.json').read_text());summaries[seed]=result['summary']
        write(out/'progress.json',dict(complete=False,summaries=summaries))
    summary={}
    for arm in ['base','proposal_min','full_work_min']:
        values=[v[arm] for v in summaries.values()]
        summary[arm]=dict(graph_validity=float(np.mean([v['graph_valid']/v['attempted'] for v in values])),
            joint_yield=[dict(threshold=t,by_seed=[next(r['mean'] for r in v['joint_yield'] if r['threshold']==t) for v in values]) for t in spec['thresholds']])
    write(out/'complete.json',dict(complete=True,protocol_sha256=sha(protocol),by_seed=summaries,summary=summary,
        new_neural_outputs=1152,new_training_steps=0,new_gfn2_attempts=1152,new_esen_queries=0,scope=spec['scope']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ['project','protocol']:parser.add_argument('--'+k,type=Path,required=True)
    parser.add_argument('--out',type=Path);parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
    if args.freeze:freeze(args.project,args.protocol)
    else:run(args.project,args.protocol,args.out)
