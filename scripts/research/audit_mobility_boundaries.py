#!/usr/bin/env python3
"""Inspect saved failed geometry probes without additional physical queries."""
import argparse,json
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from cfm_mol.chemical_moves import covalent_radii,infer_chemical_graph
from scripts.research.audit_masked_angular import sha


def components(adjacency):
    remaining=set(range(len(adjacency)));result=[]
    while remaining:
        reached={min(remaining)};front=list(reached)
        while front:
            i=front.pop()
            for j in np.flatnonzero(adjacency[i]):
                if int(j) not in reached:reached.add(int(j));front.append(int(j))
        remaining-=reached;result.append(sorted(reached))
    return result


def inspect(x,radii):
    distance=np.linalg.norm(x[:,None]-x[None,:],axis=-1);scale=radii[:,None]+radii[None,:]
    ratios=distance/scale;off=np.triu(np.ones_like(ratios,dtype=bool),1)
    return distance,ratios,components(ratios<=1.25),np.argwhere(off&(ratios<.6)).tolist()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run','audit','out']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();rows=[];sources={};expected_stops=0
    for index in [1,2,3,5]:
        directory=a.run/f'condition_{index:02d}';rp=directory/'results.json';ap=a.audit/f'condition_{index:02d}/results.json'
        r=json.loads(rp.read_text());audit=json.loads(ap.read_text())
        expected_stops+=sum(v['status']=='step_too_small' for v in r['arms'])
        assert r['complete'] and audit['complete'] and audit['full_replay'] and sha(rp)==audit['source_results_sha256']
        assert sha(directory/'trace.pt')==r['trace_sha256']==audit['trace_sha256']
        sources[str(index)]=dict(results_sha256=sha(rp),trace_sha256=r['trace_sha256'],audit_sha256=sha(ap))
        d=torch.load(directory/'trace.pt',map_location='cpu',weights_only=False);z=r['condition']['numbers'];radii=covalent_radii(z).numpy()
        for arm in d['arms']:
            if arm['status']!='step_too_small':continue
            event=arm['events'][-1];assert not event['queried']
            state=d['states'][arm['current_state_id']];x=state['positions'].numpy();y=event['positions'].numpy()
            dx,rx,cx,ox=inspect(x,radii);dy,ry,cy,oy=inspect(y,radii)
            assert len(cx)==1 and not ox
            changed=[];kind='overlap' if oy else 'disconnected' if len(cy)>1 else 'graph_change'
            if kind=='graph_change':
                try:graph=infer_chemical_graph(event['positions'],z,r['condition']['charge'])
                except ValueError:
                    kind='chemical_perception_failure'
                    limiting=np.argwhere(np.triu((rx<=1.25)!=(ry<=1.25),1)).tolist()
                else:
                    changed=torch.nonzero(torch.triu(graph['bond_orders']!=arm['frozen_bonds'],diagonal=1)).tolist();assert changed
                    limiting=changed
            elif kind=='overlap':limiting=oy
            else:limiting=np.argwhere(np.triu((rx<=1.25)&(ry>1.25),1)).tolist();assert limiting
            force=state['force_eV_A'].numpy()-.1*x;projection=arm['basis'].numpy()@arm['basis'].numpy().T;projected=projection@force
            constraints=[]
            for i,j in limiting:
                normal=np.zeros_like(x);normal[i]=(x[i]-x[j])/dx[i,j];normal[j]=-normal[i]
                normal=projection@normal;norm=np.linalg.norm(normal)
                radial=float(np.sum(projected*normal));tangent=projected-normal*(radial/(norm*norm)) if norm>0 else projected
                constraints.append(dict(atoms=[i,j],atomic_numbers=[z[i],z[j]],current_distance_A=float(dx[i,j]),rejected_distance_A=float(dy[i,j]),
                    current_radius_ratio=float(rx[i,j]),rejected_radius_ratio=float(ry[i,j]),current_bond_order=float(arm['frozen_bonds'][i,j]),
                    projected_force_dot_distance_gradient_eV_A=radial,
                    force_after_removing_one_distance_normal_max_eV_A=float(np.linalg.norm(tangent,axis=1).max())))
            rows.append(dict(index=index,pair_id=arm['pair_id'],parent=arm['parent'],endpoint=arm['endpoint'],mobility=arm['mobility'],kind=kind,
                failure_reason=event['failure_reason'],candidate_evaluations=arm['evaluations'],current_state_id=arm['current_state_id'],
                rejected_components=cy,constraints=constraints,projected_force_max_eV_A=float(np.linalg.norm(projected,axis=1).max())))
    assert len(rows)==expected_stops
    result=dict(complete=True,sources=sources,stopped_arms=rows,counts=dict(Counter(v['kind'] for v in rows)),new_physical_queries=0,
        scope='Reconstruct limiting atom pairs at the saved final rejected probes. Single-distance force decomposition is descriptive, not a constrained stationarity certificate or proof that a feasible descending direction exists.')
    if a.out.exists():raise FileExistsError(a.out)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(counts=result['counts'],stopped_arms=rows),indent=2))


if __name__=='__main__':main()
