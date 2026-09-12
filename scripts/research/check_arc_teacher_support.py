#!/usr/bin/env python3
"""Geometry-only real-context qualification of normalized distance-constrained arcs."""
import argparse
import json
from pathlib import Path
import torch

from cfm_mol.chemical_moves import covalent_radii,infer_chemical_graph
from cfm_mol.geodesic_arc_proposal import root_distance_constraints,draw_direction,direction_log_prob
from scripts.research.evaluate_chemical_policy import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();root=Path(__file__).resolve().parents[2]
    pp=root/'research/evidence/arc_teacher_support_protocol_v1.json';protocol=json.loads(pp.read_text())
    ap=args.project/'runs/multicomposition_angular_audit_v1/results.json';audit=json.loads(ap.read_text())
    assert audit['complete'] and sha(ap)==protocol['probe_audit_sha256']
    rows=[];traces=[]
    for ar in audit['rows']:
        index=ar['index'];directory=args.project/f'runs/multicomposition_angular_probe_v1/condition_{index:02d}'
        report=json.loads((directory/'results.json').read_text());assert sha(directory/'results.json')==ar['results_sha256']
        if ar['zero_support']:continue
        assert sha(directory/'trace.pt')==ar['trace_sha256']
        saved=torch.load(directory/'trace.pt',map_location='cpu',weights_only=False)
        numbers=report['condition']['numbers'];charge=report['condition']['charge'];radii=covalent_radii(numbers)
        for context,diag in zip(saved['contexts'],report['diagnostics']):
            if not diag['full_rank']:continue
            state=saved['states'][context['center_state_id']];root_pair=context['root'];leaf,anchor=root_pair;x=state['positions']
            radius=(x[leaf]-x[anchor]).norm();base=(x[leaf]-x[anchor])/radius
            normals,limits=root_distance_constraints(x,root_pair,radius,radii,margin=protocol['distance_margin_A'])
            site=torch.tensor(diag['physical_site_parameter'],dtype=torch.float64)
            harmonic=torch.tensor(diag['confinement_parameter'],dtype=torch.float64)
            params={'fitted':torch.tensor(diag['fitted_parameter'],dtype=torch.float64),
                    'site64':site,'site64_confinement':site+harmonic,'uniform':torch.zeros(3,dtype=torch.float64)}
            for number,method in enumerate(protocol['methods']):
                seed=protocol['seed']+100000000*index+100000*context['context']+100*number
                rng=torch.Generator().manual_seed(seed)
                record=dict(condition=index,context=context['context'],parent=context['parent_id'],method=method,
                    negative_center_axial_parameter=diag['center_axial_parameter']<0,attempts=protocol['draws_per_context'],
                    drawn=0,valid=0,reverse_density_finite=0,validator_errors=0)
                samples=[]
                for draw in range(protocol['draws_per_context']):
                    u,q,trace=draw_direction(base,normals,limits,params[method],generator=rng,
                        max_segment_width=protocol['max_segment_width_rad'],chart_tolerance=protocol['chart_tolerance'])
                    sample=dict(draw=draw,direction=u,forward_log_q=q,trace=trace,valid=False)
                    samples.append(sample)
                    if u is None:continue
                    record['drawn']+=1
                    y=x.clone();y[leaf]=y[anchor]+radius*u;y-=y.mean(0)
                    reverse,_=direction_log_prob(base,u,normals,limits,params[method],
                        max_segment_width=protocol['max_segment_width_rad'],chart_tolerance=protocol['chart_tolerance'])
                    sample['reverse_log_q']=reverse;record['reverse_density_finite']+=int(torch.isfinite(reverse))
                    try:
                        graph=infer_chemical_graph(y,numbers,charge)
                        if not torch.equal(graph['bond_orders'],state['graph']['bond_orders']):
                            raise ValueError('Different conditioning graph')
                        record['valid']+=1;sample['valid']=True
                    except (ValueError,IndexError,RuntimeError) as exc:
                        sample.update(error_type=type(exc).__name__,reason=str(exc));record['validator_errors']+=not isinstance(exc,ValueError)
                rows.append(record);traces.append(dict(condition=index,context=context['context'],method=method,seed=seed,
                    samples=samples,generator_state=rng.get_state()))
        print(json.dumps(dict(condition=index,completed_contexts=report['contexts'])),flush=True)
    summaries=[]
    for method in protocol['methods']:
        for group in ['all','negative_axial','nonnegative_axial']:
            subset=[r for r in rows if r['method']==method and (group=='all' or r['negative_center_axial_parameter']==(group=='negative_axial'))]
            summaries.append(dict(method=method,group=group,contexts=len(subset),
                **{k:sum(r[k] for r in subset) for k in ['attempts','drawn','valid','reverse_density_finite','validator_errors']}))
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'results.json').exists() or (args.out/'trace.pt').exists():raise FileExistsError(args.out)
    torch.save(traces,args.out/'trace.pt')
    result=dict(complete=True,protocol_sha256=sha(pp),probe_audit_sha256=sha(ap),trace_sha256=sha(args.out/'trace.pt'),
        rows=rows,summaries=summaries,new_physical_queries=0,new_model_fitted=False,scientific_submission_ready=False,
        scope='Training-context single-root structural-support and reverse-density screen. Same-graph fixed-radius geometry only; not joint graph-exchange, energy acceptance, trained-AI benefit or equilibrium validation.')
    write(args.out/'results.json',result);print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
