#!/usr/bin/env python3
"""Independent fixed-coordinate GFN2 readout for both adapted generator families."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import subprocess
import time
import torch
from scripts.research.evaluate_fresh_primary_xtb import run_task
from scripts.research.run_matched_generators import write
from scripts.research.train_electronic_fm import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','protocol','generation','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--workers',type=int,default=8);args=p.parse_args();torch.set_num_threads(1)
    spec=json.loads(args.protocol.read_text());ph=sha(args.protocol);assert spec['frozen'] and not args.out.exists()
    completed=json.loads((args.generation/'complete.json').read_text());assert completed['complete'] and completed['protocol_sha256']==ph
    binary=Path(spec['xtb_binary']);assert sha(binary)==spec['xtb_binary_sha256']
    panel=json.loads((args.project/spec['panel']).read_text())['test_rows'];tasks=[];sources=[]
    for i,c in enumerate(panel):
        condition=dict(c,numbers=c['atomic_numbers'])
        for method in ['distance_base','distance_physical','gaga_base','gaga_physical','reference']:
            if method=='reference':
                x=torch.load(args.generation/f'physical_eval/reference_c{i}.pt',map_location='cpu',weights_only=False)['positions'][0]
                for inverse in [False,True]:
                    task=dict(task_id=f'reference_c{i}_'+('minus' if inverse else 'plus'),method=method,replica=spec['model_seed'],parent_id=i,
                        inversion_check=inverse,positions=(-x if inverse else x).tolist(),condition_index=i,sample_index=-1,mirrored_task_id=f'reference_c{i}_plus')
                    tasks.append((task,condition))
            else:
                file=args.generation/f'evaluation/{method}_c{i}.pt';sample=torch.load(file,map_location='cpu',weights_only=False)
                reportfile=args.generation/f'evaluation/{method}_results.json';report=json.loads(reportfile.read_text())
                assert report['complete'] and sha(file)==report['rows'][i]['sample_sha256']
                assert sample['condition']['composition_hex']==c['composition_hex'] and len(sample['positions'])==32
                sources.append(dict(method=method,condition_index=i,sample_sha256=sha(file),report_sha256=sha(reportfile)))
                for j,y in enumerate(sample['positions']):
                    tasks.append((dict(task_id=f'{method}_c{i}_s{j}',method=method,replica=spec['model_seed'],parent_id=i*32+j,
                        inversion_check=False,positions=y.tolist(),condition_index=i,sample_index=j),condition))
    assert len(tasks)==spec['expected_gfn2_attempts_per_seed'];args.out.mkdir(parents=True)
    write(args.out/'tasks.json',dict(complete=True,protocol_sha256=ph,tasks=[dict(task=t,condition=c) for t,c in tasks],sources=sources))
    version=subprocess.run([str(binary),'--version'],capture_output=True,text=True,check=True)
    result=dict(complete=False,protocol_sha256=ph,tasks_sha256=sha(args.out/'tasks.json'),rows=[],requested_attempts=len(tasks),xtb_binary_sha256=sha(binary),version=version.stdout+version.stderr)
    start=time.perf_counter()
    def work(task,condition):
        row=run_task(task,condition,binary,args.out,spec['xtb']);row.update(condition_index=task['condition_index'],sample_index=task['sample_index']);return row
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for future in as_completed([executor.submit(work,t,c) for t,c in tasks]):
            result['rows'].append(future.result())
            if len(result['rows'])%256==0:
                result['seconds']=time.perf_counter()-start;write(args.out/'results.json',result)
                print(json.dumps(dict(completed=len(result['rows']),failed=sum(not r['success'] for r in result['rows']))),flush=True)
    result['rows'].sort(key=lambda r:r['task_id']);result.update(complete=True,attempted=len(result['rows']),failed=sum(not r['success'] for r in result['rows']),seconds=time.perf_counter()-start)
    write(args.out/'results.json',result)


if __name__=='__main__':main()
