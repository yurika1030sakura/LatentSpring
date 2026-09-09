#!/usr/bin/env python3
"""Summarize completed displacement-head experiments and matched prior controls."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics as st


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();sources={}
    def read(relative):
        path=args.runs/relative;value=json.loads(path.read_text())
        if value.get('complete') is False:raise ValueError(f'Incomplete source {relative}')
        sources[relative]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        return value
    baseline=read('position_assessment_10000_v1/xtb/xtb.json')
    reference={(r['validation_index'],r['sample_id']):r for r in baseline['rows'] if r['arm']=='position_10000'}
    results=[]
    for name in ['displacement_development_rho0_v1','displacement_development_rho01_v2']:
        training=read(name+'/training/validation.json');protocol=read(name+'/training/protocol.json')
        samples=read(name+'/assessment/sampling/samples.json')
        xtb=read(name+'/assessment/xtb/xtb.json');panel=read(name+'/density/panel.json')
        rows=[r for r in xtb['rows'] if r['arm']!='reference'];ok=[r for r in rows if r['success']]
        assert {(r['validation_index'],r['sample_id']) for r in rows}==set(reference)
        paired={'better':0,'worse':0,'both_failed':0,'tie':0}
        for right in rows:
            left=reference[(right['validation_index'],right['sample_id'])]
            if not left['success'] and not right['success']:key='both_failed'
            elif not left['success']:key='better'
            elif not right['success']:key='worse'
            else:
                change=right['strain_eV']-left['strain_eV']
                key='tie' if abs(change)<1e-6 else 'better' if change<0 else 'worse'
            paired[key]+=1
        density=[{'parent':r['parent_id'],'n_atoms':r['n_atoms'],
            'max_mean_centered_drift_nat':r['resolutions'][-1]['max_mean_centered_change_from_previous'],
            'max_replica_centered_drift_nat':r['resolutions'][-1]['max_centered_change_from_previous']}
            for r in panel['rows']]
        result={'run':name,'position_parameterization':protocol['position_parameterization'],
            'geometry_softening':protocol['geometry_softening'],'terminal_time':protocol['terminal_time'],
            'seed':protocol['seed'],'steps':training['global_step'],
            'training_seconds':training['seconds'],'peak_GiB':training['peak_gpu_bytes']/2**30,
            'xtb_attempted':len(rows),'xtb_converged':len(ok),
            'median_strain_eV_success_only':st.median(r['strain_eV'] for r in ok),
            'mean_strain_eV_success_only':st.mean(r['strain_eV'] for r in ok),
            'coordinate_rms_64_128_max_A':max(r['coordinate_rms_64_128_A'] for r in samples['arms'][0]['samples']),
            'paired_vs_endpoint_10000':paired,'density':density,
            'density_parents_above_0_1_nat':sum(r['max_mean_centered_drift_nat']>.1 for r in density)}
        results.append(result)
    output={'sources':sources,'results':results,
        'claim':'one-seed architecture development, no energy training',
        'limitations':['Endpoint versus displacement comparisons change both the head target and terminal time',
            'Two displacement runs share initialization and training seed; they are not independent seed replication',
            'Successful-only strain medians use different subsets; failure-aware paired counts are also reported',
            'The necessary numerical gate is failed by both runs; no converged-density claim']}
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k!='density'} for r in results],indent=2))


if __name__=='__main__':main()
