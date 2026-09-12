#!/usr/bin/env python3
"""Replay training probes and independently solve their angular normal equations."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from cfm_mol.chemical_sampler import ChemicalTarget
from cfm_mol.terminal_rotation import terminal_rotation_actions
from cfm_mol.normalized_site_guide import NormalizedSiteGuide
from scripts.research.audit_masked_angular import ReplayOracle,equal,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','run','out']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();root=args.project
    pp=root/'research/evidence/angular_curvature_probe_protocol_v1.json';protocol=json.loads(pp.read_text())
    report=json.loads((args.run/'results.json').read_text());assert report['complete'] and report['protocol_sha256']==sha(pp)
    tp=root/'runs/chemical_policy_table_v1/training.pt';assert sha(tp)==protocol['training_artifact_sha256']
    assert sha(args.run/'trace.pt')==report['trace_sha256']
    data=torch.load(tp,map_location='cpu',weights_only=False);saved=torch.load(args.run/'trace.pt',map_location='cpu',weights_only=False)
    rng=torch.Generator().manual_seed(protocol['seed']);selected=[]
    for index,parent in enumerate(data['source_parent_ids'][:protocol['parents']]):
        c=next((r for r in data['table'] if r['parent_id']==parent and r['valid'] and r['action_index']>0 and r['action'][2]!=r['action'][3]),None)
        if c:
            i,j,k,l=c['action'];leaf,anchor=[(i,l),(j,k)][int(torch.randint(2,(1,),generator=rng))];sid=c['new_state_id']
        else:
            sid=data['warm_state_ids'][index];roots=terminal_rotation_actions(data['condition']['numbers'],data['states'][sid]['graph']['bond_orders'])
            leaf,anchor=roots[int(torch.randint(len(roots),(1,),generator=rng))]
        selected.append(dict(parent_id=parent,training_state_id=sid,leaf=leaf,anchor=anchor,source_kind='post_exchange' if c else 'warm'))
    assert selected==saved['selected']==report['selected']
    oracle=ReplayOracle(saved['query_trace']);target=ChemicalTarget(oracle,data['condition'],data['kT_eV'],data['restraint_eV_A2'])
    centers=target.evaluate([target.coordinate_state(data['states'][r['training_state_id']]['positions']) for r in selected],phase='centers')
    candidates=[];valid_rows=[]
    for context,(row,center) in enumerate(zip(selected,centers)):
        leaf,anchor=row['leaf'],row['anchor'];x=center['positions'];radius=(x[leaf]-x[anchor]).norm();u=(x[leaf]-x[anchor])/radius
        noise=torch.randn(3,dtype=x.dtype,generator=rng);a=noise-(noise*u).sum()*u;a/=a.norm();b=torch.cross(u,a,dim=0)
        for number,(role,axis,sign) in enumerate([('fit',a,1),('fit',a,-1),('fit',b,1),('fit',b,-1),('check',(a+b)/math.sqrt(2),1),('check',(a+b)/math.sqrt(2),-1)]):
            row_saved=saved['probes'][6*context+number];angle=protocol['fit_angle_rad'] if role=='fit' else protocol['check_angle_rad']
            y=x.clone();y[leaf]=x[anchor]+radius*(math.cos(angle)*u+sign*math.sin(angle)*axis);y-=y.mean(0)
            equal(y,row_saved['positions']);equal(axis,row_saved['axis'])
            assert (context,number,role,sign,angle)==tuple(row_saved[k] for k in ['context','number','role','sign','angle_rad'])
            try:
                candidate=target.coordinate_state(y)
                if not torch.equal(candidate['graph']['bond_orders'],center['graph']['bond_orders']):raise ValueError('Probe changed the conditioning graph')
                assert row_saved['valid'];candidates.append(candidate);valid_rows.append(row_saved)
            except ValueError as exc:assert not row_saved['valid'] and row_saved['failure']==str(exc)
    target.evaluate(candidates,phase='angular_probes')
    for row,state in zip(valid_rows,candidates):assert row['state_id']==state['state_id']
    equal(target.states,saved['states']);equal(target.query_trace,saved['query_trace']);equal(rng.get_state(),saved['generator_state'])
    assert oracle.evaluated==report['new_raw_queries']==report['requested_raw_queries'] and oracle.index==len(oracle.queries)
    models=[]
    for rep in [0,1]:
        path=root/f'runs/normalized_site_train_v1/vector_s{rep}/model.pt';assert sha(path)==protocol['model_sha256'][rep]
        ck=torch.load(path,map_location='cpu',weights_only=False);m=NormalizedSiteGuide(**ck['configuration']).double();m.load_state_dict(ck['state_dict']);models.append(m.eval())
    checked=0;maximum_parameter_error=0.
    for context,(row,center) in enumerate(zip(selected,centers)):
        leaf,anchor=row['leaf'],row['anchor'];diag=report['diagnostics'][context]
        def values(state):
            x=state['positions'].numpy();v=x[leaf]-x[anchor];radius=np.linalg.norm(v);u=v/radius
            force=state['force_eV_A'].numpy()-target.restraint*x;force-=force.mean(0)
            s=radius*(np.eye(3)-np.outer(u,u))@force[leaf]/target.kT
            return u,s
        fit=[center]+[target.states[r['state_id']] for r in saved['probes'] if r['context']==context and r['role']=='fit' and r['valid']]
        us,ss=zip(*(values(s) for s in fit));matrix=sum(np.eye(3)-np.outer(u,u) for u in us)
        full_rank=np.linalg.eigvalsh(matrix)[0]>1e-5;assert bool(full_rank)==diag['full_rank']
        if not full_rank:continue
        eta=np.linalg.solve(matrix,np.sum(ss,axis=0));err=float(np.max(np.abs(eta-np.array(diag['fitted_parameter']))));maximum_parameter_error=max(err,maximum_parameter_error);assert err<1e-7
        x=center['positions'].numpy();relative=x-x[anchor];relative[leaf]=0.;radius=np.linalg.norm(x[leaf]-x[anchor])
        neighbors=(center['graph']['bond_orders'][anchor].numpy()>0);neighbors[leaf]=False
        vv=relative[neighbors];away=-(vv/np.linalg.norm(vv,axis=1,keepdims=True)).sum(0);site=10*away/max(np.linalg.norm(away),1e-12)
        harmonic=target.restraint*radius*relative.sum(0)/(len(x)*target.kT)
        residual=eta-site-harmonic;capped=site+harmonic+residual*min(1.,64/max(np.linalg.norm(residual),1e-20))
        with torch.no_grad():
            old=[m(center['positions'][None],center['graph']['bond_orders'][None],torch.tensor(target.numbers),torch.tensor([target.condition['charge'],target.condition['spin_multiplicity'],target.kT],dtype=torch.float64),torch.tensor([[leaf,anchor]]))[0][0,0].numpy() for m in models]
        for check in diag['checks']:
            probe=saved['probes'][6*context+check['number']];state=target.states[probe['state_id']];u,s=values(state)
            for name,parameter in dict(site=site,site_confinement=site+harmonic,local_oracle_vmf=eta,local_oracle_capped64=capped,frozen_s0=old[0],frozen_s1=old[1]).items():
                mse=np.mean(((np.eye(3)-np.outer(u,u))@parameter-s)**2);assert abs(mse-check['force_score_mse'][name])<1e-6
            assert abs(-float(eta@(u-us[0]))-check['fitted_work_over_kT'])<1e-7
            assert abs(float((state['potential_eV']-center['potential_eV'])/target.kT)-check['actual_work_over_kT'])<1e-7
            checked+=1
    result=dict(complete=True,results_sha256=sha(args.run/'results.json'),trace_sha256=sha(args.run/'trace.pt'),
        full_geometry_and_random_stream_replay=True,independent_normal_equation_checks=report['full_rank_contexts'],heldout_force_and_work_checks=checked,
        maximum_parameter_error=maximum_parameter_error,new_physical_queries=0,scientific_submission_ready=False)
    if args.out.exists():raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
