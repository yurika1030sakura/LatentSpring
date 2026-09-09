#!/usr/bin/env python3
"""Assess fixed work-sampler outputs with geometry diagnostics and independent xTB.

All geometries enter contact/diversity diagnostics. A seeded subset, selected
without energies or work weights, enters GFN2 single-point and relaxation checks.
Failures remain in every denominator. This is conditional method development,
not a certificate of chemical validity or equilibrium mode coverage.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import itertools
import json
from pathlib import Path
import re
import shutil

import numpy as np
import torch
from rdkit import Chem

from cfm_mol.geometry_diagnostics import distance_profile,profile_rms,contact_summary
from cfm_mol.nonequilibrium import normalized_weights,WeightedPaths
from checkpoint_panel import write_json
from eval_position_xtb import evaluate


def select_indices(total,count,seed):
    if not 1<=count<=total:raise ValueError('Subset must fit the complete sample panel')
    return sorted(torch.randperm(total,generator=torch.Generator().manual_seed(seed))[:count].tolist())


def paired_outcomes(left,right):
    left={r['sample_id']:r for r in left};right={r['sample_id']:r for r in right}
    if left.keys()!=right.keys():raise ValueError('Paired comparison requires identical sample IDs')
    counts=dict(better=0,worse=0,tie=0,both_failed=0)
    for key,a in left.items():
        b=right[key]
        if not a['success'] and not b['success']:category='both_failed'
        elif not a['success']:category='better'
        elif not b['success']:category='worse'
        else:
            delta=b['strain_eV']-a['strain_eV']
            category='tie' if abs(delta)<1e-6 else 'better' if delta<0 else 'worse'
        counts[category]+=1
    return {'attempted_pairs':len(left),**counts,
        'ranking':'converged before failed; then lower strain. This ranking is not chemical validity.'}


def quantiles(values):
    return dict(zip(['q10','median','q90'],map(float,np.quantile(values,[.1,.5,.9])))) if values else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples',nargs=2,action='append',metavar=('ARM','PATH'),required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--xtb-count',type=int,default=32)
    p.add_argument('--seed',type=int,default=9059);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--max-cycles',type=int,default=200)
    args=p.parse_args();binary=shutil.which('xtb')
    if binary is None:raise FileNotFoundError('xtb binary not found')
    names=[name for name,_ in args.samples]
    if len(set(names))!=len(names) or any(not re.fullmatch(r'[A-Za-z0-9_]+',name) for name in names):
        raise ValueError('Unique alphanumeric arm names required')
    if args.workers<1 or args.max_cycles<1:raise ValueError('Positive worker and optimization counts required')
    args.out.mkdir(parents=True,exist_ok=True);output=args.out/'assessment.json'
    if output.exists():raise FileExistsError(output)
    table=Chem.GetPeriodicTable();condition=None;count=None;tasks=[];geometry=[];sources=[]
    for name,file in args.samples:
        path=Path(file);parent=path.parent/'results.json'
        if not json.loads(parent.read_text())['complete']:raise ValueError('Require completed source training')
        data=torch.load(str(path),map_location='cpu',weights_only=False)
        if condition is None:condition=data['condition']
        if any(data['condition'][key]!=condition[key] for key in ['numbers','charge','spin_multiplicity']):
            raise ValueError('Matched panel physical conditions differ')
        x=data['positions'].double();numbers=condition['numbers'];n=len(numbers)
        if x.ndim!=3 or x.shape[1:]!=(n,3) or not torch.isfinite(x).all():raise ValueError('Invalid saved geometry')
        if count is None:count=len(x)
        if len(x)!=count:raise ValueError('Matched panels must have identical sample counts')
        work=data.get('work')
        if work is not None and work.shape!=(count,):raise ValueError('One work value is required per sample')
        selected=select_indices(count,args.xtb_count,args.seed)
        weights=normalized_weights(-work.double()).numpy() if work is not None else None
        radii=[table.GetRcovalent(int(z)) for z in numbers]
        contacts=[contact_summary(item.numpy(),radii) for item in x]
        profiles=[distance_profile(item.numpy(),numbers) for item in x]
        pair_distances=[profile_rms(a,b) for a,b in itertools.combinations(profiles,2)]
        profile_vectors=np.asarray([np.concatenate([profile[k] for k in sorted(profile)]) for profile in profiles])
        mean=np.sum(weights[:,None]*profile_vectors,axis=0) if weights is not None else None
        geometry.append({'arm':name,'particles':count,'contact_rows':contacts,
            'overlap_count':sum(row['overlap_pairs']>0 for row in contacts),
            'multiple_contact_components_count':sum(row['contact_components']>1 for row in contacts),
            'contact_components':quantiles([row['contact_components'] for row in contacts]),
            'pair_distance_profile_rms_A':quantiles(pair_distances),
            'weighted_profile_variance_A2':float(np.sum(weights[:,None]*(profile_vectors-mean)**2)/profile_vectors.shape[1]) if weights is not None else None,
            'weights':WeightedPaths(x,-work.double(),{}).summary() if work is not None else None,
            'weight_scope':'finite-path importance weights' if weights is not None else 'unavailable: no proposal density was evaluated'})
        for index in selected:
            tasks.append({'arm':name,'validation_index':condition.get('source_row',0),'sample_id':index,
                'positions':x[index].tolist(),'symbols':[table.GetElementSymbol(int(z)) for z in numbers],
                'charge_recorded':condition['charge'],'spin':condition['spin_multiplicity']})
        sources.append({'arm':name,'condition':data['condition'],'samples':str(path.resolve()),'samples_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'results_sha256':hashlib.sha256(parent.read_bytes()).hexdigest()})
    report={'complete':False,'scope':__doc__,'condition':condition,'sources':sources,'geometry':geometry,
        'source_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
            [Path(__file__).resolve(),Path(__file__).with_name('eval_position_xtb.py').resolve()]},
        'xtb_subset_indices':selected,'subset_seed':args.seed,'requested_xtb':len(tasks),'xtb_rows':[],
        'xtb_binary':binary,'xtb_sha256':hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
        'contact_factor':1.25,'overlap_factor':.6,'radii_source':'RDKit GetRcovalent',
        'limitations':['Contact graphs are distance heuristics; fragments are allowed by the declared target.',
            'Distance profiles are incomplete invariants, not aligned molecular RMSD or mode labels.',
            'Importance ESS describes these weights, not proven mode coverage.',
            'One training seed and one condition do not establish reproducible molecular benefit.',
            'Strain statistics condition on convergence; report failures alongside them.']}
    write_json(output,report)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(evaluate,task,binary,args.out,args.max_cycles) for task in tasks]
        for future in as_completed(futures):
            row=future.result();report['xtb_rows'].append(row);write_json(output,report)
            print(json.dumps({k:row.get(k) for k in ['arm','sample_id','success','strain_eV','failure']}),flush=True)
    report['xtb_rows'].sort(key=lambda row:(row['arm'],row['sample_id']))
    summaries=[]
    for name in names:
        rows=[row for row in report['xtb_rows'] if row['arm']==name]
        successes=[row for row in rows if row['success']]
        summaries.append({'arm':name,'attempted':len(rows),'converged':len(successes),
            'strain_eV_success_only':quantiles([row['strain_eV'] for row in successes]),
            'initial_max_force_norm_eV_A':quantiles([row['initial_max_force_norm_eV_A'] for row in rows if 'initial_max_force_norm_eV_A' in row]),
            'force_values_present':sum('initial_max_force_norm_eV_A' in row for row in rows)})
    report['xtb_summaries']=summaries;report['paired']=[]
    for first,second in itertools.combinations(names,2):
        report['paired'].append({'before':first,'after':second,**paired_outcomes(
            [row for row in report['xtb_rows'] if row['arm']==first],
            [row for row in report['xtb_rows'] if row['arm']==second])})
    report['complete']=True;write_json(output,report)
    print(json.dumps({'xtb':summaries,'paired':report['paired']}),flush=True)


if __name__=='__main__':main()
