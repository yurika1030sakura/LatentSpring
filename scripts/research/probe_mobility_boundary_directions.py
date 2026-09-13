#!/usr/bin/env python3
"""Find geometrically feasible, force-predicted descent at saved boundary stops."""
import argparse,json
from pathlib import Path
import torch
from cfm_mol.chemical_moves import infer_chemical_graph
from scripts.research.audit_masked_angular import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run','boundaries','out']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();b=json.loads(a.boundaries.read_text());assert b['complete']
    results=[]
    for index in [1,2,3,5]:
        directory=a.run/f'condition_{index:02d}';rp=directory/'results.json';tp=directory/'trace.pt'
        assert sha(rp)==b['sources'][str(index)]['results_sha256'] and sha(tp)==b['sources'][str(index)]['trace_sha256']
        r=json.loads(rp.read_text());d=torch.load(tp,map_location='cpu',weights_only=False)
        for stopped in [v for v in b['stopped_arms'] if v['index']==index]:
            arm=next(v for v in d['arms'] if all(v[k]==stopped[k] for k in ['pair_id','endpoint','mobility']))
            state=d['states'][arm['current_state_id']];x=state['positions'];basis=arm['basis'];force=state['force_eV_A']-.1*x
            allowed=basis@(basis.T@force)
            for mode in ['force','distance_projected_force']:
                direction=allowed.clone();active=[]
                if mode=='distance_projected_force':
                    for constraint in stopped['constraints']:
                        # Remote bond-order reassignment is not a local distance boundary.
                        if abs(constraint['current_radius_ratio']-1.25)>1e-6:continue
                        i,j=constraint['atoms'];normal=torch.zeros_like(x);normal[i]=(x[i]-x[j])/(x[i]-x[j]).norm();normal[j]=-normal[i]
                        normal=basis@(basis.T@normal)
                        # Positive normal points out of the frozen adjacency's distance side.
                        if constraint['current_bond_order']==0:normal=-normal
                        active.append(normal)
                    for _ in range(3):
                        for normal in active:
                            norm=normal.norm()
                            if norm<=1e-14:continue
                            dot=(normal*direction).sum();margin=.001*direction.norm()*norm
                            if dot>-margin:direction-=(dot+margin)/norm.square()*normal
                maximum=float(direction.norm(dim=1).max());attempts=[];accepted=None
                if maximum>0:
                    direction*=.05/maximum
                    for backtrack in range(23):
                        delta=direction*(.5**backtrack);predicted=float((force*delta).sum())
                        if predicted<=0:break
                        y=x+delta;y-=y.mean(0);reason=None
                        try:
                            graph=infer_chemical_graph(y,r['condition']['numbers'],r['condition']['charge'])
                            if not torch.equal(graph['bond_orders'],arm['frozen_bonds']):raise ValueError('Perceived graph differs from frozen endpoint graph')
                        except ValueError as exc:reason=str(exc)
                        attempts.append(dict(backtrack=backtrack,max_atom_step_A=float(delta.norm(dim=1).max()),failure_reason=reason))
                        if reason is None:
                            recovered=y-x;torch.testing.assert_close(recovered,basis@(basis.T@recovered),atol=1e-12,rtol=0)
                            accepted=dict(positions=y.tolist(),max_atom_step_A=float(recovered.norm(dim=1).max()),force_predicted_decrease_eV=float((force*recovered).sum()))
                            break
                results.append(dict(index=index,pair_id=arm['pair_id'],parent=arm['parent'],endpoint=arm['endpoint'],mobility=arm['mobility'],
                    state_id=arm['current_state_id'],mode=mode,active_distance_normals=len(active),attempts=attempts,feasible=accepted))
    report=dict(complete=True,boundary_sha256=sha(a.boundaries),settings=dict(max_atom_step_A=.05,backtracks=23,inward_fraction=.001,projection_passes=3),
        rows=results,feasible_counts={m:sum(v['feasible'] is not None for v in results if v['mode']==m) for m in ['force','distance_projected_force']},
        new_physical_queries=0,scope=f"Geometry and first-order force prediction only. No actual potential decrease, stationary constrained optimum, sampler gain or learned advantage is established. All{len(b['stopped_arms'])} saved minimum-step stops are retained.")
    if a.out.exists():raise FileExistsError(a.out)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(feasible_counts=report['feasible_counts'],predictions=[{k:v[k] for k in ['index','pair_id','endpoint','mobility','mode']}|dict(feasible={k:z for k,z in v['feasible'].items() if k!='positions'} if v['feasible'] else None) for v in results]),indent=2))


if __name__=='__main__':main()
