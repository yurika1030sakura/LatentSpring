#!/usr/bin/env python3
"""Pool independently seeded frozen-work evaluations, retaining every stream."""
import argparse
import json
import math
from pathlib import Path

import torch

from cfm_mol.triatomic_reference import invariant_observables
from molecular_tempered_pilot import sha,write_json


def combine(label,runs):
    records=[];samples=[];sources={};draw_seeds=set();first=None
    for run in runs:
        p=run/'normalizer.json';comparison=json.loads(p.read_text());q=run/'sampling/results.json';source=json.loads(q.read_text())
        if not comparison['complete'] or not source['complete'] or source['source_evaluation_stream']:
            raise ValueError('Require complete independent fresh-stream evaluations')
        if comparison['source_results_sha256']!=sha(q):raise ValueError('Evaluation source changed')
        if source['evaluation_seed_rule']!='seed + 100003*batch_index':raise ValueError('Unknown evaluation RNG rule')
        recipe=source['configuration'];seeds={recipe['seed']+100003*i for i in range(recipe['eval_particles']//recipe['batch'])}
        if seeds&draw_seeds:raise ValueError('Evaluation streams overlap')
        draw_seeds.update(seeds)
        signature=[source['trained_checkpoint_sha256'],source['energy_zero_eV'],source['condition'],
            recipe['kT'],recipe['restraint'],source['oracle_sha256'],comparison['source_reference_sha256']]
        if first is None:first=signature
        if signature!=first:raise ValueError('Cannot pool different trained proposals or targets')
        rows=[row for row in comparison['rows'] if row['stage']=='final']
        if len(rows)!=1:raise ValueError('One final-stage comparison required')
        sample=run/'sampling/final_samples.pt'
        if comparison['sample_sources'][str(sample.resolve())]!=sha(sample):raise ValueError('Sample source changed')
        data=torch.load(str(sample),map_location='cpu',weights_only=False)
        if len(data['work'])!=rows[0]['particles']:raise ValueError('Sample count changed')
        records.append(rows[0]);samples.append(data)
        for path in [p,q,sample]:sources[str(path.resolve())]=sha(path)
    work=torch.cat([d['work'] for d in samples]).double();positions=torch.cat([d['positions'] for d in samples]).double()
    n=len(work);weights=torch.softmax(-work,0);ess=float(weights.square().sum().reciprocal());moments={}
    for key,value in invariant_observables(positions.numpy()).items():
        value=torch.as_tensor(value);mean=float(weights@value);reference=records[0]['moments'][key]
        moments[key]={'weighted_mean':mean,'difference':mean-reference['reference_mean'],
            'empirical_estimate_se':float((n/(n-1)*(weights*(value-mean)).square().sum()).sqrt()) if ess>=10 else None,
            'reference_delta_se':reference['reference_delta_se']}
    logdiff=float(torch.logsumexp(torch.tensor([row['log_normalizer_difference']+math.log(row['particles']) for row in records],dtype=torch.float64),0)-math.log(n))
    return {'arm':label,'runs':[str(p.resolve()) for p in runs],'streams':[{k:v for k,v in row.items() if k!='moments'} for row in records],
        'combined':{'particles':n,'ess':ess,'maximum_weight':float(weights.max()),'log_normalizer_difference':logdiff,
            'normalizer_ratio_minus_one':math.expm1(logdiff),'empirical_normalizer_relative_se':math.sqrt((n/ess-1)/(n-1)),
            'moments':moments},'sources_sha256':sources}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--group',nargs=3,action='append',required=True,metavar=('LABEL','RUN1','RUN2'))
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    rows=[combine(label,[Path(one),Path(two)]) for label,one,two in args.group]
    report={'complete':True,'scope':__doc__,'rows':rows,'limitations':[
        'Reference uncertainty is additional and shared across arms.',
        'The two streams within each arm are independent; streams can be shared across different arms.',
        'Finite-weight delta errors do not detect entirely missed tails or establish convergence.',
        'One molecular condition and unequal numbers of training seeds do not establish broad superiority.'],
        'scientific_submission_ready':False,'script_sha256':sha(Path(__file__))}
    write_json(args.out,report)
    for row in rows:print(row['arm'],{k:v for k,v in row['combined'].items() if k!='moments'})


if __name__=='__main__':main()
