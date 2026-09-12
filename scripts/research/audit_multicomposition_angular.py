#!/usr/bin/env python3
"""Replay new training probes and independently check geometry, densities and fits."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from cfm_mol.chemical_sampler import ChemicalTarget
from scripts.research.audit_joint_chemical import independent_log_q
from scripts.research.audit_masked_angular import ReplayOracle, equal, sha
from scripts.research.probe_multicomposition_angular import load_inputs, collect


def independent_checks(saved, report, target, protocol):
    states=saved['states'];kT=target.kT;gamma=target.restraint
    counts=dict(probe_geometries=0,joint_coordinate_densities=0,identified_parameters=0,heldout_force_work_checks=0)
    maximum_parameter_error=0.
    numbers=torch.tensor(target.numbers,dtype=torch.long)
    electronic=torch.tensor([target.condition['charge'],target.condition['spin_multiplicity'],kT],dtype=torch.float64)
    for trial in saved['exchange_trials']:
        if not trial['forward_count']:continue
        old=states[trial['warm_state_id']];x=old['positions'];y=trial['proposal_positions']
        q=independent_log_q(x,y,old['graph']['bond_orders'],numbers,electronic,target.radii,
            trial['action'],trial['order'],'site',None,protocol,trial['forward'])
        assert abs(q-float(trial['log_forward_coordinate']))<1e-7
        counts['joint_coordinate_densities']+=1
        if trial['valid']:
            new=states[trial['new_state_id']]
            qr=independent_log_q(y,x,new['graph']['bond_orders'],numbers,electronic,target.radii,
                trial['inverse_action'],trial['order'],'site',None,protocol,trial['reverse'])
            assert abs(qr-float(trial['log_reverse_coordinate']))<1e-7
            counts['joint_coordinate_densities']+=1
    def angular(state,root):
        leaf,anchor=root;x=state['positions'].numpy();v=x[leaf]-x[anchor];r=np.linalg.norm(v);u=v/r
        f=state['force_eV_A'].numpy()-gamma*x;f=f-f.mean(0)
        return u,r*(np.eye(3)-np.outer(u,u))@f[leaf]/kT
    for context,diag in zip(saved['contexts'],report['diagnostics']):
        cid=context['context'];assert diag['context']==cid and diag['parent_id']==context['parent_id']
        leaf,anchor=context['root'];center=states[context['center_state_id']]
        x=center['positions'].numpy();radius=np.linalg.norm(x[leaf]-x[anchor]);u0=(x[leaf]-x[anchor])/radius
        noise=context['frame_noise'].numpy();a=noise-np.dot(noise,u0)*u0;a/=np.linalg.norm(a);b=np.cross(u0,a)
        related=[r for r in saved['probes'] if r['context']==cid];assert len(related)==6
        spec=[('fit',a,1),('fit',a,-1),('fit',b,1),('fit',b,-1),('check',(a+b)/math.sqrt(2),1),('check',(a+b)/math.sqrt(2),-1)]
        for number,(row,(role,axis,sign)) in enumerate(zip(related,spec)):
            angle=protocol['fit_angle_rad'] if role=='fit' else protocol['check_angle_rad']
            assert (row['number'],row['role'],row['sign'],row['angle_rad'])==(number,role,sign,angle)
            np.testing.assert_allclose(row['axis'].numpy(),axis,atol=1e-10,rtol=1e-10)
            y=x.copy();y[leaf]=x[anchor]+radius*(math.cos(angle)*u0+sign*math.sin(angle)*axis);y-=y.mean(0)
            np.testing.assert_allclose(row['positions'].numpy(),y,atol=1e-10,rtol=1e-10)
            passive=np.arange(len(x))!=leaf
            np.testing.assert_allclose(y[passive]-y[anchor],x[passive]-x[anchor],atol=1e-10,rtol=1e-10)
            assert abs(np.linalg.norm(y[leaf]-y[anchor])-radius)<1e-10
            counts['probe_geometries']+=1
        fit_ids=[center['state_id']]+[r['state_id'] for r in related if r['role']=='fit' and r['valid']]
        assert fit_ids==diag['fit_state_ids']
        values=[angular(states[sid],(leaf,anchor)) for sid in fit_ids]
        matrix=sum(np.eye(3)-np.outer(u,u) for u,_ in values)
        eigenvalues=np.linalg.eigvalsh(matrix)
        np.testing.assert_allclose(eigenvalues,diag['design_eigenvalues'],atol=1e-10,rtol=1e-10)
        assert bool(eigenvalues[0]>protocol['rank_threshold'])==diag['full_rank']
        if not diag['full_rank']:
            assert 'fitted_parameter' not in diag;continue
        eta=np.linalg.solve(matrix,np.sum([s for _,s in values],axis=0))
        error=float(np.max(np.abs(eta-np.array(diag['fitted_parameter']))));maximum_parameter_error=max(maximum_parameter_error,error)
        np.testing.assert_allclose(eta,diag['fitted_parameter'],atol=1e-7,rtol=1e-9)
        counts['identified_parameters']+=1
        residual=np.concatenate([(np.eye(3)-np.outer(u,u))@eta-s for u,s in values])
        np.testing.assert_allclose(np.mean(residual**2),diag['fit_score_mse'],atol=1e-7,rtol=1e-9)
        relative=x-x[anchor];relative[leaf]=0
        neighbors=center['graph']['bond_orders'][anchor].numpy()>0;neighbors[leaf]=False
        vectors=relative[neighbors];away=-(vectors/np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True),1e-12)).sum(0)
        site=64*away/max(np.linalg.norm(away),1e-12);harmonic=gamma*radius*relative.sum(0)/(len(x)*kT)
        np.testing.assert_allclose(site,diag['physical_site_parameter'],atol=1e-8,rtol=1e-9)
        np.testing.assert_allclose(harmonic,diag['confinement_parameter'],atol=1e-8,rtol=1e-9)
        checks=[r for r in related if r['role']=='check' and r['valid']]
        assert len(checks)==len(diag['checks'])
        for row,check in zip(checks,diag['checks']):
            assert row['state_id']==check['state_id'] and row['number']==check['number']
            state=states[row['state_id']];u,s=angular(state,(leaf,anchor))
            for name,param in dict(fitted=eta,site64=site,site64_confinement=site+harmonic).items():
                mse=np.mean(((np.eye(3)-np.outer(u,u))@param-s)**2)
                np.testing.assert_allclose(mse,check['force_score_mse'][name],atol=1e-7,rtol=1e-9)
                np.testing.assert_allclose(-param@(u-u0),check['predicted_work_over_kT'][name],atol=1e-7,rtol=1e-9)
            actual=float((state['potential_eV']-center['potential_eV'])/kT)
            assert abs(actual-check['actual_work_over_kT'])<1e-9
            counts['heldout_force_work_checks']+=1
    counts['maximum_parameter_error']=maximum_parameter_error
    return counts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','run','out']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();root=Path(__file__).resolve().parents[2];rows=[];total=0
    for index in range(8):
        pp,protocol,prepared,warm,physical=load_inputs(root,args.project,index)
        directory=args.run/f'condition_{index:02d}';report=json.loads((directory/'results.json').read_text())
        assert report['complete'] and report['protocol_sha256']==sha(pp)
        assert report['condition']==prepared['condition'] and report['parent_ids']==prepared['parent_ids']
        assert report['preparation_audit_sha256']==protocol['preparation_audit_sha256']
        assert report['preparation_results_sha256']==sha(args.project/protocol['preparation_run']/f'condition_{index:02d}/results.json')
        assert report['split']==protocol['condition_splits'][str(index)] and report['stream']=='fresh_training' and not report['model_fitted']
        assert report['new_raw_queries']==report['requested_raw_queries']
        result=dict(index=index,results_sha256=sha(directory/'results.json'),raw_queries=report['new_raw_queries'],zero_support=warm is None)
        if warm is None:
            assert report['zero_support'] and report['new_raw_queries']==report['contexts']==0 and not (directory/'trace.pt').exists()
        else:
            assert sha(directory/'trace.pt')==report['trace_sha256'] and not report['zero_support']
            saved=torch.load(directory/'trace.pt',map_location='cpu',weights_only=False)
            oracle=ReplayOracle(saved['query_trace']);target=ChemicalTarget(oracle,prepared['condition'],physical['kT_eV'],physical['restraint_eV_A2'])
            replay={};collect(target,warm,protocol,index,replay);equal(replay,saved)
            assert oracle.index==len(oracle.queries) and oracle.evaluated==report['new_raw_queries']
            assert report['contexts']==len(saved['contexts'])==len(report['diagnostics'])
            assert report['full_rank_contexts']==sum(d['full_rank'] for d in report['diagnostics'])
            assert report['exchange_attempts']==len(saved['exchange_trials']) and report['supported_exchanges']==sum(r['valid'] for r in saved['exchange_trials'])
            assert report['probe_attempts']==len(saved['probes']) and report['supported_probes']==sum(r['valid'] for r in saved['probes'])
            assert report['new_raw_queries']==2*(report['supported_exchanges']+report['supported_probes'])
            result.update(trace_sha256=sha(directory/'trace.pt'),**independent_checks(saved,report,target,protocol))
        total+=report['new_raw_queries'];rows.append(result);print(json.dumps(result),flush=True)
    assert total<=protocol['maximum_total_new_raw_queries']
    result=dict(complete=True,protocol_sha256=sha(pp),rows=rows,raw_queries_in_probes=total,new_physical_queries=0,
        all_eight_conditions_retained=True,full_geometry_and_random_stream_replay=True,
        independent_numpy_geometry_and_parameter_checks=True,scientific_submission_ready=False)
    if args.out.exists():raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
