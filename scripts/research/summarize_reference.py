#!/usr/bin/env python3
"""Compare adaptive references, quadrature orders and fixed-step densities."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def centered(values):
    a=np.asarray(values,dtype=np.float64)
    if a.ndim!=2:raise ValueError('Expected replica by geometry array')
    return a-a.mean(-1,keepdims=True)


def difference(left,right):
    a,b=centered(left),centered(right)
    if a.shape!=b.shape:raise ValueError('Replica dimensions differ')
    return {'maximum_centered_replica_change_nat':float(np.abs(a-b).max()),
            'maximum_centered_mean_change_nat':float(np.abs(a.mean(0)-b.mean(0)).max())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--comparison-panel',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--allow-partial',action='store_true')
    args=p.parse_args();sources={}
    def read(path):
        data=json.loads(path.read_text())
        if not data['complete'] and not args.allow_partial:
            raise ValueError(f'Incomplete source: {path}')
        sources[str(path.resolve())]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                                    'complete':data['complete']}
        return data
    reference=read(args.reference);source=read(Path(reference['source_panel']))
    if hashlib.sha256(Path(reference['source_panel']).read_bytes()).hexdigest()!=reference['source_panel_sha256']:
        raise ValueError('Original panel changed after reference execution')
    comparison=read(args.comparison_panel) if args.comparison_panel is not None else None
    if comparison is not None:
        for key in ['checkpoint_sha256','config_sha256','shard_sha256','probe_policy',
                    'position_parameterization','terminal_time','geometry_softening','replicas']:
            if source[key]!=comparison[key]:raise ValueError(f'Incompatible comparison {key}')
        if comparison.get('perturbation_indices')!=source.get('perturbation_indices'):
            raise ValueError('Different perturbation geometry selection')
    rows=[]
    for row in reference['rows']:
        original=next(r for r in source['rows'] if r['parent_id']==row['parent_id'])
        result={'parent_id':row['parent_id'],'tolerances':[]}
        previous=None
        for item in row['references']:
            value={'rtol':item['rtol'],'success':item['success']}
            if not item['success']:
                value['error']=item['error']
                result['tolerances'].append(value)
                continue
            value.update(nfe=item['nfe'],seconds=item['total_seconds'],
                quadrature_2_to_4=difference(item['estimates']['2']['log_q'],item['estimates']['4']['log_q']),
                midpoint_float64_vs_reference=difference(row['midpoint_float64'],item['estimates']['4']['log_q']),
                source_final_vs_reference=difference(original['resolutions'][-1]['log_q'],item['estimates']['4']['log_q']))
            if previous is not None:
                value['previous_tolerance_to_current']=difference(previous,item['estimates']['4']['log_q'])
            if comparison is not None:
                matching=[r for r in comparison['rows'] if r['parent_id']==row['parent_id']]
                if matching and matching[0]['resolutions']:
                    other=matching[0]
                    if other['geometry_sha256']!=row['geometry_sha256']:raise ValueError('Geometry changed')
                    resolution=other['resolutions'][-1]
                    value['comparison_vs_reference']={'steps':resolution['steps'],
                        **difference(resolution['log_q'],item['estimates']['4']['log_q'])}
            previous=item['estimates']['4']['log_q']
            result['tolerances'].append(value)
        rows.append(result)
    report={'complete':all(s['complete'] for s in sources.values()),'sources':sources,'rows':rows,
            'scope':'Development diagnostic; partial reports and successful numerical solves are not certification',
            'selection':reference['selection'],
            'all_reference_solves_succeeded':reference['all_reference_solves_succeeded'] if reference['complete'] else None}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
