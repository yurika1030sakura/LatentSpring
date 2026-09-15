#!/usr/bin/env python3
"""Independent all-output GFN2 scores for the new fixed-budget students."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import subprocess
import torch

from scripts.research.evaluate_fresh_primary_xtb import run_task
from scripts.research.evaluate_generator_quality import write
from scripts.research.train_electronic_fm import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['project','protocol','generation','out']:
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args();torch.set_num_threads(1)
    spec=json.loads(args.protocol.read_text());ph=sha(args.protocol)
    done=json.loads((args.generation/'complete.json').read_text())
    assert spec['frozen'] and done['complete'] and done['protocol_sha256']==ph and not args.out.exists()
    binary=Path(spec['xtb_binary']);assert sha(binary)==spec['xtb_binary_sha256']
    panelpath=args.project/spec['condition_manifest'];assert sha(panelpath)==spec['condition_manifest_sha256']
    panel=json.loads(panelpath.read_text());poolpath=args.project/panel['source_pool']
    assert sha(poolpath)==panel['source_pool_sha256'];pool=json.loads(poolpath.read_text())['rows']
    tasks=[];sources=[]
    for i,c in enumerate(panel['rows']):
        c=dict(c,numbers=c['atomic_numbers']);reference=pool[c['pool_index']]
        assert reference['condition']['raw_index']==c['raw_index']
        for method in ['reference']+spec['methods']:
            if method=='reference':positions=torch.tensor(reference['reference_positions'],dtype=torch.float64)[None]
            else:
                path=args.generation/'evaluation'/f'{method}_c{i}.pt';reportfile=args.generation/'evaluation'/f'{method}_results.json'
                report=json.loads(reportfile.read_text());assert report['complete'] and report['protocol_sha256']==ph
                assert sha(path)==report['rows'][i]['sample_sha256']
                saved=torch.load(path,weights_only=False,map_location='cpu');positions=saved['positions']
                assert saved['condition']['composition_hex']==c['composition_hex']
                sources.append(dict(method=method,condition_index=i,sample_sha256=sha(path),report_sha256=sha(reportfile)))
            for j,x in enumerate(positions):
                for minus in ([False,True] if method=='reference' else [False]):
                    task=dict(task_id=f'{method}_c{i}_s{j}_'+('minus' if minus else 'plus'),method=method,
                        replica=spec['model_seed'],parent_id=i*32+j,inversion_check=minus,condition_index=i,sample_index=j,
                        positions=(-x if minus else x).tolist())
                    if minus:task['mirrored_task_id']=f'{method}_c{i}_s{j}_plus'
                    tasks.append(dict(task=task,condition=c))
    assert len(tasks)==2352;args.out.mkdir(parents=True)
    write(args.out/'tasks.json',dict(protocol_sha256=ph,tasks=tasks,sources=sources))
    version=subprocess.run([str(binary),'--version'],capture_output=True,text=True,check=True)
    report=dict(complete=False,protocol_sha256=ph,tasks_sha256=sha(args.out/'tasks.json'),rows=[],requested_attempts=len(tasks),
        xtb_binary_sha256=sha(binary),version=version.stdout+version.stderr)
    def work(item):
        row=run_task(item['task'],item['condition'],binary,args.out,spec['xtb'])
        row.update(condition_index=item['task']['condition_index'],sample_index=item['task']['sample_index']);return row
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures=[executor.submit(work,item) for item in tasks]
        for future in as_completed(futures):
            report['rows'].append(future.result())
            if len(report['rows'])%64==0:
                write(args.out/'results.json',report)
                print(json.dumps(dict(completed=len(report['rows']),failed=sum(not r['success'] for r in report['rows']))),flush=True)
    report['rows'].sort(key=lambda r:r['task_id'])
    report.update(complete=True,attempted=len(report['rows']),successful=sum(r['success'] for r in report['rows']))
    write(args.out/'results.json',report)


if __name__=='__main__':main()
