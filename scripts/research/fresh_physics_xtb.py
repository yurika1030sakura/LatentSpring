#!/usr/bin/env python3
"""Independent GFN2 readout of every frozen fresh-panel output and reference."""
import argparse,json,time,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import torch
from scripts.research.evaluate_fresh_primary_xtb import run_task
from scripts.research.train_electronic_fm import sha
from scripts.research.evaluate_chemical_policy import write


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ['project','protocol','generation','out']:p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--workers',type=int,default=8);a=p.parse_args();torch.set_num_threads(1)
 if a.out.exists():raise FileExistsError(a.out)
 spec=json.loads(a.protocol.read_text());ph=sha(a.protocol);assert spec['frozen'];done=json.loads((a.generation/'complete.json').read_text());assert done['complete'] and done['protocol_sha256']==ph
 binary=Path(spec['xtb_binary']);assert sha(binary)==spec['xtb_binary_sha256'];version=subprocess.run([str(binary),'--version'],capture_output=True,text=True,check=True)
 panel=json.loads((a.project/spec['condition_manifest']).read_text());pool=json.loads((a.project/panel['source_pool']).read_text());tasks=[];sources=[]
 for i,c in enumerate(panel['rows']):
  condition=dict(c,numbers=c['atomic_numbers']);reference=pool['rows'][c['pool_index']];assert reference['condition']['raw_index']==c['raw_index'];x=reference['reference_positions']
  for inverse in [False,True]:
   task=dict(task_id=f'reference_c{i}_'+('minus' if inverse else 'plus'),method='reference',replica=spec['model_seed'],parent_id=i,inversion_check=inverse,positions=[[-v for v in row] for row in x] if inverse else x,condition_index=i,sample_index=-1,mirrored_task_id=f'reference_c{i}_plus')
   tasks.append((task,condition))
  for method in spec['methods']:
   file=a.generation/'evaluation'/f'{method}_c{i}.pt';d=torch.load(file,map_location='cpu',weights_only=False);reportfile=a.generation/'evaluation'/f'{method}_results.json';report=json.loads(reportfile.read_text());assert report['complete'] and report['protocol_sha256']==ph and sha(file)==report['rows'][i]['sample_sha256']
   assert d['condition']['composition_hex']==c['composition_hex'] and len(d['positions'])==spec['samples_per_condition'];sources.append(dict(method=method,condition_index=i,sample_sha256=sha(file),report_sha256=sha(reportfile)))
   for j,y in enumerate(d['positions'].tolist()):tasks.append((dict(task_id=f'{method}_c{i}_s{j}',method=method,replica=spec['model_seed'],parent_id=i*spec['samples_per_condition']+j,inversion_check=False,positions=y,condition_index=i,sample_index=j),condition))
 assert len(tasks)==spec['xtb_attempts_per_seed'];a.out.mkdir(parents=True);start=time.perf_counter()
 write(a.out/'tasks.json',dict(complete=True,protocol_sha256=ph,tasks=[dict(task=t,condition=c) for t,c in tasks],sources=sources))
 output=a.out/'results.json';report=dict(complete=False,protocol_sha256=ph,tasks_sha256=sha(a.out/'tasks.json'),rows=[],requested_attempts=len(tasks),version=version.stdout+version.stderr,xtb_binary_sha256=sha(binary))
 write(output,report)
 def work(t,c):
  result=run_task(t,c,binary,a.out,spec['xtb']);result.update(condition_index=t['condition_index'],sample_index=t['sample_index']);return result
 with ThreadPoolExecutor(max_workers=a.workers) as executor:
  futures=[executor.submit(work,t,c) for t,c in tasks]
  for future in as_completed(futures):
   report['rows'].append(future.result())
   if len(report['rows'])%64==0:
    report['seconds']=time.perf_counter()-start;write(output,report);print(json.dumps(dict(completed=len(report['rows']),failed=sum(not r['success'] for r in report['rows']),seconds=report['seconds'])),flush=True)
 report['rows'].sort(key=lambda r:r['task_id']);report.update(complete=True,attempted=len(report['rows']),successful=sum(r['success'] for r in report['rows']),failed=sum(not r['success'] for r in report['rows']),seconds=time.perf_counter()-start,scope='All-output gas-phase GFN2 single points and gradients at fixed geometries. Failures retained; independent approximate evaluator, not DFT or equilibrium validation.');write(output,report)


if __name__=='__main__':main()
