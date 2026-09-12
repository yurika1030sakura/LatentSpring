#!/usr/bin/env python3
"""Replay joint arc proposals and independently integrate real marginal densities."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch

from cfm_mol.chemical_moves import covalent_radii,infer_chemical_graph
from cfm_mol.chemical_path_guide import exchanged_bond_graph
from cfm_mol.joint_arc_geometry import marginal_joint_arc_proposal
from cfm_mol.joint_chemical_geometry import joint_geometry_proposal,distinct_anchor_actions
from scripts.research.audit_arc_teacher_support import independent_direction_logp
from scripts.research.audit_joint_chemical import independent_log_q
from scripts.research.audit_masked_angular import equal,sha


def independent_energy_log_score(directions,record):
    """NumPy evaluation of the serialized conditional radial energy readout."""
    c={k:v.detach().cpu().numpy() for k,v in record['context'].items()}
    point=float(c['radius'])*directions
    distance=np.sqrt(np.sum((point[...,None,:]-c['masked'])**2,axis=-1)+1e-12)
    scaled=distance/c['pair_radii']
    radial=np.exp(-.5*((scaled[...,None]-record['query_centers'].detach().cpu().numpy())/record['query_width'])**2)
    gate=np.where(distance<record['cutoff'],.5*(1+np.cos(np.pi*distance/record['cutoff'])),0.)*(c['roles']!=1)
    residual=np.sum(np.sum(radial*c['coefficients'],axis=-1)*gate,axis=-1)/np.sqrt(np.sum(c['roles']!=1))
    energy=np.sum(directions*c['linear_energy_parameter'],axis=-1)+residual
    return -energy/record['kT']


def independent_arc_q(trace):
    values=[]
    for component in trace['components']:
        if component is None or component['failed']:
            values.append(-math.inf);continue
        ell=component['log_radii'].numpy();mean=component['radial_means'].numpy();sigma=component['radial_width']
        value=float(np.sum(-.5*((ell-mean)/sigma)**2-math.log(sigma*math.sqrt(2*math.pi))-3*ell))
        for step in component['steps']:
            score=(lambda u:independent_energy_log_score(u,step['energy_score'])) if 'energy_score' in step else None
            value+=independent_direction_logp(step['direction'].numpy(),step['base_direction'].numpy(),
                step['normals'].numpy(),step['limits'].numpy(),step['eta'].numpy(),component['max_segment_width'],score=score)
        assert abs(value-float(component['log_coordinate_density']))<1e-7
        values.append(value)
    return float(np.logaddexp(*values)-math.log(2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','run','out']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();root=Path(__file__).resolve().parents[2]
    pp=root/'research/evidence/joint_arc_support_protocol_v1.json';protocol=json.loads(pp.read_text())
    report=json.loads((args.run/'results.json').read_text());assert report['complete'] and report['protocol_sha256']==sha(pp)
    assert sha(args.run/'trace.pt')==report['trace_sha256']
    traces=torch.load(args.run/'trace.pt',map_location='cpu',weights_only=False)
    ap=args.project/'runs/proposal_training_preparation_audit_v1/results.json';audit=json.loads(ap.read_text())
    assert audit['complete'] and sha(ap)==protocol['preparation_audit_sha256']
    contexts={}
    for r in audit['rows']:
        if r['zero_support']:continue
        path=args.project/f"runs/proposal_training_preparation_v1/condition_{r['index']:02d}/trace.pt"
        assert sha(path)==r['trace_sha256'];data=torch.load(path,map_location='cpu',weights_only=False)
        by_parent={p:data['states'][sid] for p,sid in zip(data['parent_ids'],data['history_state_ids'][-1])}
        contexts[r['index']]=(data['condition'],by_parent)
    checked=0;maximum_error=0.;independent=0
    for trace,row in zip(traces,report['rows']):
        index,parent,trial,method=[trace[k] for k in ['condition','parent','trial','method']]
        assert tuple(row[k] for k in ['condition','parent','trial','method'])==(index,parent,trial,method)
        condition,parents=contexts[index];old=parents[parent];numbers=torch.tensor(condition['numbers']);radii=covalent_radii(numbers)
        electronic=torch.tensor([condition['charge'],condition['spin_multiplicity'],protocol['kT_eV']],dtype=torch.float64)
        seed=protocol['seed']+100000000*index+100000*parent+1000*trial;assert seed==trace['seed']
        rng=torch.Generator().manual_seed(seed);actions=distinct_anchor_actions(numbers,old['graph']['bond_orders'])
        pick=int(torch.randint(len(actions),(1,),generator=rng));action=actions[pick];order=int(torch.randint(2,(1,),generator=rng))
        assert pick==trace['choice_index'] and action==trace['action'] and order==trace['order']
        equal(rng.get_state(),trace['prefix_generator_state'])
        proposal=joint_geometry_proposal if method=='legacy_site64' else marginal_joint_arc_proposal
        kind='site' if method=='legacy_site64' else method
        kwargs=dict(kind=kind,order=order,radial_width=protocol['radial_width'],site_concentration=64.)
        y,q,forward=proposal(old['positions'],old['graph']['bond_orders'],numbers,electronic,radii,action,generator=rng,**kwargs)
        equal(y,trace['positions']);equal(q,trace['log_forward']);equal(forward,trace['forward']);equal(rng.get_state(),trace['generator_state'])
        assert row['drawn']==(y is not None)
        if y is None:
            assert not row['geometry_supported'] and not row['reverse_density_positive'] and row['failure']==forward['failure']
            checked+=1;continue
        desired=exchanged_bond_graph(old['graph']['bond_orders'],action);i,j,k,l=action;inverse=(i,j,l,k)
        try:
            graph=infer_chemical_graph(y,numbers.tolist(),condition['charge'])
            if not torch.equal(graph['bond_orders'],desired):raise ValueError('Different desired graph')
            assert row['geometry_supported']
            if inverse not in distinct_anchor_actions(numbers,graph['bond_orders']):raise ValueError('Inverse graph action ineligible')
            _,qr,reverse=proposal(y,desired,numbers,electronic,radii,inverse,observed=old['positions'],**kwargs)
            equal(qr,trace['log_reverse']);equal(reverse,trace['reverse'])
            assert row['reverse_density_positive']==bool(torch.isfinite(qr))
            if method!='legacy_site64' and torch.isfinite(qr):
                assert row['order_marginalization_rescued']==(not bool(torch.isfinite(reverse['order_log_densities'][order])))
            if trial==0:
                if method=='legacy_site64':
                    values=[independent_log_q(old['positions'],y,old['graph']['bond_orders'],numbers,electronic,radii,
                        action,order,'site',None,dict(radial_width=.05,site_concentration=64.),forward),
                        independent_log_q(y,old['positions'],desired,numbers,electronic,radii,inverse,order,'site',None,
                        dict(radial_width=.05,site_concentration=64.),reverse)]
                else:values=[independent_arc_q(forward),independent_arc_q(reverse)]
                for value,expected in zip(values,[q,qr]):
                    if math.isfinite(value) and torch.isfinite(expected):
                        error=abs(value-float(expected));maximum_error=max(maximum_error,error);assert error<1e-7
                    else:assert not math.isfinite(value) and not torch.isfinite(expected)
                    independent+=1
        except (ValueError,IndexError,RuntimeError) as exc:
            assert row['failure']==str(exc) and row['error_type']==type(exc).__name__
        checked+=1
    assert checked==len(report['rows'])==protocol['maximum_attempts']
    result=dict(complete=True,results_sha256=sha(args.run/'results.json'),trace_sha256=sha(args.run/'trace.pt'),
        all_attempts_replayed=checked,all_random_streams_replayed=True,all_geometry_checks_repeated=True,
        independent_first_trial_forward_reverse_densities=independent,maximum_log_density_error=maximum_error,
        new_physical_queries=0,scientific_submission_ready=False)
    if args.out.exists():raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
