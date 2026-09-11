#!/usr/bin/env python3
"""Audit the prescribed eight-condition geometry campaign without hiding failures."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    expected={'base','convex_s0','convex_s1','affine_s0','affine_s1','typed_s0','typed_s1'}
    report={'complete':False,'scope':__doc__,'conditions':[],'expected_conditions':8,'expected_attempts':8*7*32,
        'scientific_submission_ready':False,'limitations':['Partial completion is not an all-eight result.',
            'GFN2 convergence and strain are independent diagnostics, not full chemical validity or target-mode coverage.',
            'Report failed calculations together with success-only strain statistics.']}
    for index in range(8):
        path=args.runs_root/f'parity_geometry_condition_{index:02d}_v1/assessment.json'
        row={'index':index,'state':'unresolved'};report['conditions'].append(row)
        if not path.exists():continue
        d=json.loads(path.read_text())
        if not d.get('complete'):
            row.update(recorded_attempts=len(d.get('xtb_rows',[])));continue
        if (d['identity_namespace']!='development_manifest' or d['identity_index']!=index
                or not d['shared_parent_contract_verified'] or d['requested_xtb']!=224
                or len(d['xtb_rows'])!=224 or set(r['arm'] for r in d['xtb_rows'])!=expected):
            raise ValueError('Geometry condition, parent contract or expected denominator differs')
        for source in d['sources']:
            sample=Path(source['samples'])
            if sha(sample)!=source['samples_sha256'] or sha(sample.parent/'results.json')!=source['results_sha256']:
                raise ValueError('Assessed source artifact changed')
        summaries=[]
        for arm in sorted(expected):
            records=[r for r in d['xtb_rows'] if r['arm']==arm]
            if len(records)!=32 or sorted(r['sample_id'] for r in records)!=d['xtb_subset_indices']:
                raise ValueError('Missing/duplicate or mismatched geometry parent')
            successful=[r for r in records if r['success']]
            summary=next(r for r in d['xtb_summaries'] if r['arm']==arm)
            if len(successful)!=summary['converged']:raise ValueError('Success denominator disagrees')
            if successful:
                np.testing.assert_allclose(np.median([r['strain_eV'] for r in successful]),
                    summary['strain_eV_success_only']['median'],atol=1e-12,rtol=1e-12)
            failures={}
            for r in records:
                if not r['success']:
                    key=r.get('failure','unspecified');failures[key]=failures.get(key,0)+1
            summaries.append({**summary,'failures':failures})
        row.update(state='complete',assessment_sha256=sha(path),condition=d['condition'],
            attempts=224,converged=sum(r['success'] for r in d['xtb_rows']),
            failures=sum(not r['success'] for r in d['xtb_rows']),summaries=summaries,
            paired=d['paired'],geometry=[{k:v for k,v in r.items() if k!='contact_rows'} for r in d['geometry']])
    report['completed_conditions']=sum(r['state']=='complete' for r in report['conditions'])
    report['unresolved_conditions']=8-report['completed_conditions']
    report['completed_attempts']=sum(r.get('attempts',0) for r in report['conditions'])
    report['converged']=sum(r.get('converged',0) for r in report['conditions'])
    report['failed_calculations']=sum(r.get('failures',0) for r in report['conditions'])
    report['complete']=True;args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ['completed_conditions','unresolved_conditions','completed_attempts','converged','failed_calculations']}))


if __name__=='__main__':main()
