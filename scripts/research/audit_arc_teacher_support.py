#!/usr/bin/env python3
"""Replay real arc draws and independently integrate their first-draw densities."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch

from cfm_mol.chemical_moves import covalent_radii,infer_chemical_graph
from cfm_mol.geodesic_arc_proposal import root_distance_constraints,draw_direction,direction_log_prob
from cfm_mol.spherical_proposal import vmf_sample
from scripts.research.audit_masked_angular import equal,sha

TAU=2*math.pi
NODES,WEIGHTS=np.polynomial.legendre.leggauss(64)
FRACTIONS=(NODES+1)/2


def independent_intervals(base,tangent,normals,limits):
    # An arrangement of all crossing angles, independent of sequential clipping.
    endpoints=[0.,TAU]
    aa=normals@base;bb=normals@tangent
    for a,b,c in zip(aa,bb,limits):
        amplitude=math.hypot(a,b)
        if amplitude<1e-14:
            if c<0:return []
            continue
        z=c/amplitude
        if z>=1:continue
        if z<=-1:return []
        center=math.atan2(b,a);offset=math.acos(z)
        endpoints.extend([(center-offset)%TAU,(center+offset)%TAU])
    endpoints=sorted(set(endpoints));intervals=[]
    for a,b in zip(endpoints[:-1],endpoints[1:]):
        mid=(a+b)/2;direction=math.cos(mid)*base+math.sin(mid)*tangent
        if np.all(normals@direction<=limits):
            if intervals and intervals[-1][1]==a:intervals[-1]=(intervals[-1][0],b)
            else:intervals.append((a,b))
    return intervals


def quadrature_angle_logp(theta,base,tangent,normals,limits,eta,width,score=None):
    arcs=independent_intervals(base,tangent,normals,limits);segments=[]
    for left,right in arcs:
        n=max(1,math.ceil((right-left)/width))
        for i in range(n):segments.append((left+(right-left)*i/n,left+(right-left)*(i+1)/n))
    if not segments:return -math.inf
    edges=np.array(segments)
    directions=np.cos(edges)[...,None]*base+np.sin(edges)[...,None]*tangent
    heights=directions@eta if score is None else score(directions);delta=heights[:,1]-heights[:,0];peak=float(heights.max())
    # Independent Gauss-Legendre integration of the implemented interpolant.
    density=np.exp(heights[:,0,None]+delta[:,None]*FRACTIONS-peak)
    integrals=(edges[:,1]-edges[:,0])*(density@WEIGHTS)/2
    logz=peak+math.log(float(integrals.sum()))
    for (a,b),(lo,hi) in zip(edges,heights):
        if a-1e-12<=theta<=b+1e-12:return float(lo+(hi-lo)*(theta-a)/(b-a)-logz)
    return -math.inf


def independent_direction_logp(observed,base,normals,limits,eta,width,score=None):
    cosine=float(observed@base);v=observed-cosine*base;sine=float(np.linalg.norm(v))
    assert sine>1e-10
    if sine<1e-5:
        # Independent second projection in the numerically sensitive pole chart.
        v=v-float(v@base)/float(base@base)*base
        sine=float(np.linalg.norm(v))
    tangent=v/sine;theta=math.atan2(sine,cosine)
    first=quadrature_angle_logp(theta,base,tangent,normals,limits,eta,width,score=score)
    second=quadrature_angle_logp(TAU-theta,base,-tangent,normals,limits,eta,width,score=score)
    return float(np.logaddexp(first,second)-math.log(TAU)-math.log(sine))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','run','out']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();root=Path(__file__).resolve().parents[2]
    pp=root/'research/evidence/arc_teacher_support_protocol_v1.json';protocol=json.loads(pp.read_text())
    report=json.loads((args.run/'results.json').read_text())
    assert report['complete'] and report['protocol_sha256']==sha(pp)
    assert sha(args.run/'trace.pt')==report['trace_sha256']
    records=torch.load(args.run/'trace.pt',map_location='cpu',weights_only=False)
    by_key={(r['condition'],r['context'],r['method']):r for r in records};assert len(by_key)==len(records)
    source_audit_path=args.project/'runs/multicomposition_angular_audit_v1/results.json'
    source_audit=json.loads(source_audit_path.read_text())
    assert source_audit['complete'] and sha(source_audit_path)==protocol['probe_audit_sha256']==report['probe_audit_sha256']
    old_directory=args.project/'runs/conditional_teacher_support_v1'
    old_report=json.loads((old_directory/'results.json').read_text());assert old_report['complete']
    assert sha(old_directory/'trace.pt')==old_report['trace_sha256']
    old_protocol=json.loads((root/'research/evidence/conditional_teacher_support_protocol_v1.json').read_text())
    assert old_report['protocol_sha256']==sha(root/'research/evidence/conditional_teacher_support_protocol_v1.json')
    old_records=torch.load(old_directory/'trace.pt',map_location='cpu',weights_only=False)
    old_by_key={(r['condition'],r['context'],r['method']):r for r in old_records}
    rows=[];old_rows=[];density_checks=0;maximum_error=0.
    for ar in source_audit['rows']:
        if ar['zero_support']:continue
        index=ar['index'];directory=args.project/f'runs/multicomposition_angular_probe_v1/condition_{index:02d}'
        assert sha(directory/'trace.pt')==ar['trace_sha256'] and sha(directory/'results.json')==ar['results_sha256']
        source=torch.load(directory/'trace.pt',map_location='cpu',weights_only=False)
        header=json.loads((directory/'results.json').read_text());numbers=header['condition']['numbers'];charge=header['condition']['charge']
        radii=covalent_radii(numbers)
        for context,diag in zip(source['contexts'],header['diagnostics']):
            if not diag['full_rank']:continue
            cid=context['context'];state=source['states'][context['center_state_id']];leaf,anchor=context['root'];x=state['positions']
            radius=(x[leaf]-x[anchor]).norm();base=(x[leaf]-x[anchor])/radius
            normals,limits=root_distance_constraints(x,(leaf,anchor),radius,radii,margin=protocol['distance_margin_A'])
            site=torch.tensor(diag['physical_site_parameter'],dtype=torch.float64);harmonic=torch.tensor(diag['confinement_parameter'],dtype=torch.float64)
            params=dict(fitted=torch.tensor(diag['fitted_parameter'],dtype=torch.float64),site64=site,
                site64_confinement=site+harmonic,uniform=torch.zeros(3,dtype=torch.float64))
            for m,method in enumerate(protocol['methods']):
                record=by_key[(index,cid,method)];seed=protocol['seed']+100000000*index+100000*cid+100*m
                assert record['seed']==seed;generator=torch.Generator().manual_seed(seed)
                row=dict(condition=index,context=cid,parent=context['parent_id'],method=method,
                    negative_center_axial_parameter=diag['center_axial_parameter']<0,attempts=protocol['draws_per_context'],
                    drawn=0,valid=0,reverse_density_finite=0,validator_errors=0)
                for draw,sample in enumerate(record['samples']):
                    assert sample['draw']==draw
                    u,q,trace=draw_direction(base,normals,limits,params[method],generator=generator,
                        max_segment_width=protocol['max_segment_width_rad'],chart_tolerance=protocol['chart_tolerance'])
                    equal(u,sample['direction']);equal(q,sample['forward_log_q']);equal(trace,sample['trace'])
                    if u is None:assert not sample['valid'];continue
                    row['drawn']+=1
                    reverse,_=direction_log_prob(base,u,normals,limits,params[method],
                        max_segment_width=protocol['max_segment_width_rad'],chart_tolerance=protocol['chart_tolerance'])
                    equal(reverse,sample['reverse_log_q']);row['reverse_density_finite']+=int(torch.isfinite(reverse))
                    y=x.clone();y[leaf]=x[anchor]+radius*u;y-=y.mean(0)
                    try:
                        graph=infer_chemical_graph(y,numbers,charge)
                        if not torch.equal(graph['bond_orders'],state['graph']['bond_orders']):raise ValueError('Different conditioning graph')
                        assert sample['valid'];row['valid']+=1
                    except (ValueError,IndexError,RuntimeError) as exc:
                        assert not sample['valid'] and sample['error_type']==type(exc).__name__ and sample['reason']==str(exc)
                        row['validator_errors']+=not isinstance(exc,ValueError)
                    if draw==0:
                        for observed,origin,expected in [(u,base,q),(base,u,reverse)]:
                            value=independent_direction_logp(observed.numpy(),origin.numpy(),normals.numpy(),limits.numpy(),params[method].numpy(),protocol['max_segment_width_rad'])
                            error=abs(value-float(expected));maximum_error=max(maximum_error,error);assert error<1e-7
                            density_checks+=1
                equal(generator.get_state(),record['generator_state']);rows.append(row)
            for m,method in enumerate(old_protocol['methods']):
                record=old_by_key[(index,cid,method)];seed=old_protocol['seed']+100000000*index+100000*cid+100*m
                assert record['seed']==seed;generator=torch.Generator().manual_seed(seed)
                directions,random=vmf_sample(params[method][None].expand(old_protocol['draws_per_context'],-1),generator=generator)
                equal(directions,record['directions']);equal(random,record['random']);equal(generator.get_state(),record['generator_state'])
                row=dict(condition=index,context=cid,parent=context['parent_id'],method=method,
                    negative_center_axial_parameter=diag['center_axial_parameter']<0,attempts=len(directions),valid=0,validator_errors=0)
                for u,outcome in zip(directions,record['outcomes']):
                    y=x.clone();y[leaf]=x[anchor]+radius*u;y-=y.mean(0)
                    try:
                        graph=infer_chemical_graph(y,numbers,charge)
                        if not torch.equal(graph['bond_orders'],state['graph']['bond_orders']):raise ValueError('Different conditioning graph')
                        assert outcome['valid'];row['valid']+=1
                    except (ValueError,IndexError,RuntimeError) as exc:
                        assert not outcome['valid'] and outcome['error_type']==type(exc).__name__ and outcome['reason']==str(exc)
                        row['validator_errors']+=not isinstance(exc,ValueError)
                old_rows.append(row)
        print(json.dumps(dict(condition=index,completed=True)),flush=True)
    assert rows==report['rows'] and old_rows==old_report['rows']
    result=dict(complete=True,arc_results_sha256=sha(args.run/'results.json'),arc_trace_sha256=sha(args.run/'trace.pt'),
        fullsphere_results_sha256=sha(old_directory/'results.json'),fullsphere_trace_sha256=sha(old_directory/'trace.pt'),
        all_arc_and_fullsphere_draws_replayed=True,all_graph_checks_repeated=True,
        independent_first_draw_forward_reverse_density_checks=density_checks,maximum_log_density_error=maximum_error,
        new_physical_queries=0,scientific_submission_ready=False)
    if args.out.exists():raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
