#!/usr/bin/env python3
"""Independent xTB assessment of every condition in the frozen development panel."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import shutil

import torch

from assess_work_panel import quantiles,paired_outcomes
from eval_position_xtb import evaluate
from molecular_tempered_pilot import sha,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    binary=shutil.which('xtb')
    if binary is None:raise FileNotFoundError('xtb binary not found')
    source=args.baseline/'sampling/panel.json';references=args.baseline/'references.json'
    generation=json.loads(source.read_text());ref=json.loads(references.read_text())
    if not generation['complete'] or not ref['complete']:raise ValueError('Both source stages must be complete')
    if generation['manifest_sha256']!=ref['source_manifest_sha256']:raise ValueError('Source manifest mismatch')
    output=args.out/'assessment.json'
    if output.exists():raise FileExistsError(output)
    ref_rows={r['panel_index']:r for r in ref['rows']};tasks=[];sampling_failures=[]
    for item in generation['rows']:
        index=item['panel_index'];reference=ref_rows[index];condition=reference['condition']
        if any(item['condition'][key]!=condition[key] for key in ['atomic_numbers','charge','spin_multiplicity']):raise ValueError('Reference physical condition mismatch')
        common={'validation_index':index,'symbols':reference['symbols'],'charge_recorded':condition['charge'],'spin':condition['spin_multiplicity']}
        tasks.append({**common,'arm':'reference','sample_id':0,'positions':reference['positions']})
        if not item['success']:
            sampling_failures.append({'panel_index':index,'requested_samples_per_resolution':generation['samples_per_condition'],'error':item['error']})
            continue
        directory=args.baseline/'sampling'/f'condition_{index:02d}';detail=directory/'results.json'
        if sha(detail)!=item['results_sha256']:raise ValueError('Condition source results changed')
        manifest=json.loads(detail.read_text())
        for steps in [16,64]:
            file=f'fm_midpoint_{steps}_samples.pt';path=directory/file
            entry=next(r for r in manifest['samples'] if r['file']==file)
            if sha(path)!=entry['sha256']:raise ValueError('Source samples changed')
            data=torch.load(str(path),map_location='cpu',weights_only=False)
            if len(data['positions'])!=generation['samples_per_condition']:raise ValueError('Incomplete generated condition')
            for sample_id,x in enumerate(data['positions']):tasks.append({**common,'arm':f'fm{steps}','sample_id':sample_id,'positions':x.tolist()})
    args.out.mkdir(parents=True,exist_ok=True)
    report={'complete':False,'scope':__doc__,'source_generation_sha256':sha(source),'source_references_sha256':sha(references),
        'manifest_sha256':generation['manifest_sha256'],'xtb_binary_sha256':sha(Path(binary)),
        'requested_conditions':len(generation['rows']),'generated_conditions':generation['successful_conditions'],
        'sampling_failures':sampling_failures,'requested_xtb_tasks':len(tasks),'rows':[],
        'limitations':['Generation failures are distinct from unattempted xTB evaluations.',
            'Success-only strain is reported with all attempt/failure counts.',
            'Development data and one pretrained checkpoint; no Boltzmann sampling claim.']}
    write_json(output,report)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(evaluate,task,binary,args.out,200) for task in tasks]
        for future in as_completed(futures):
            row=future.result();report['rows'].append(row);write_json(output,report)
            print(json.dumps({k:row.get(k) for k in ['arm','validation_index','sample_id','success','strain_eV','failure']}),flush=True)
    report['rows'].sort(key=lambda r:(r['validation_index'],r['arm'],r['sample_id']));summaries=[]
    for index in ref_rows:
        item={'panel_index':index,'condition':ref_rows[index]['condition'],'arms':[]}
        for arm in ['reference','fm16','fm64']:
            rows=[r for r in report['rows'] if r['validation_index']==index and r['arm']==arm]
            successful=[r for r in rows if r['success']]
            item['arms'].append({'arm':arm,'attempted':len(rows),'converged':len(successful),
                'strain_eV_success_only':quantiles([r['strain_eV'] for r in successful])})
        first=[r for r in report['rows'] if r['validation_index']==index and r['arm']=='fm16']
        second=[r for r in report['rows'] if r['validation_index']==index and r['arm']=='fm64']
        if first and second:item['paired_16_64']=paired_outcomes(first,second)
        summaries.append(item)
    report.update(complete=True,summaries=summaries);write_json(output,report)


if __name__=='__main__':main()
