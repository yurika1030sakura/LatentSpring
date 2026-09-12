#!/usr/bin/env python3
"""Counterfactual TRAINING-only angular probes diagnose density concentration."""
import argparse
import json
import math
from pathlib import Path
import torch
from cfm_mol.chemical_sampler import ChemicalTarget
from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.terminal_rotation import terminal_rotation_actions
from cfm_mol.normalized_site_guide import NormalizedSiteGuide,physical_site_parameter
from cfm_mol.masked_angular_guide import masked_angular_context
from cfm_mol.local_site_guide import confinement_parameter
from scripts.research.evaluate_chemical_policy import sha,write


def score(state,leaf,anchor,kT,restraint):
    vector=state['positions'][leaf]-state['positions'][anchor];radius=vector.norm();u=vector/radius
    force=state['force_eV_A']-restraint*state['positions'];force=force-force.mean(0)
    raw=radius*(force[leaf]-torch.dot(force[leaf],u)*u)/kT
    return u,raw


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','out','oracle-python','oracle-checkpoint']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();root=Path(__file__).resolve().parents[2]
    pp=root/'research/evidence/angular_curvature_probe_protocol_v1.json';protocol=json.loads(pp.read_text())
    tp=args.project/'runs/chemical_policy_table_v1/training.pt'
    assert sha(tp)==protocol['training_artifact_sha256']
    data=torch.load(tp,map_location='cpu',weights_only=False);assert data['stream']=='training'
    physical=json.loads((root/'research/evidence/parity_training_protocol_v1.json').read_text())
    assert sha(args.oracle_checkpoint)==physical['raw_oracle_sha256']
    selected=[]
    rng=torch.Generator().manual_seed(protocol['seed'])
    for index,parent in enumerate(data['source_parent_ids'][:protocol['parents']]):
        candidate=next((r for r in data['table'] if r['parent_id']==parent and r['valid'] and
            r['action_index']>0 and r['action'][2]!=r['action'][3]),None)
        if candidate:
            i,j,k,l=candidate['action'];roots=[(i,l),(j,k)]
            leaf,anchor=roots[int(torch.randint(2,(1,),generator=rng))];sid=candidate['new_state_id']
        else:
            sid=data['warm_state_ids'][index]
            roots=terminal_rotation_actions(data['condition']['numbers'],data['states'][sid]['graph']['bond_orders'])
            leaf,anchor=roots[int(torch.randint(len(roots),(1,),generator=rng))]
        selected.append(dict(parent_id=parent,training_state_id=sid,leaf=leaf,anchor=anchor,
            source_kind='post_exchange' if candidate else 'warm'))
    assert len(selected)==protocol['parents']
    args.out.mkdir(parents=True,exist_ok=True);output=args.out/'results.json'
    if output.exists():raise FileExistsError(output)
    report=dict(complete=False,protocol_sha256=sha(pp),training_sha256=sha(tp),selected=selected,
        condition=data['condition'],development_coordinates_loaded=False,reference_coordinates_loaded=False,
        scientific_submission_ready=False)
    write(output,report);oracle=target=None;probe_rows=[]
    try:
        oracle=EnergyOracle(args.oracle_python,root/'scripts/research/oracle_worker.py',args.oracle_checkpoint,
            numbers=data['condition']['numbers'],charge=data['condition']['charge'],spin_multiplicity=data['condition']['spin_multiplicity'],device='cuda',batch_size=32)
        assert not oracle.handshake['tf32'] and oracle.handshake['base_precision_dtype']=='torch.float32'
        target=ChemicalTarget(oracle,data['condition'],data['kT_eV'],data['restraint_eV_A2'])
        centers=target.evaluate([target.coordinate_state(data['states'][r['training_state_id']]['positions']) for r in selected],phase='centers')
        for row,state in zip(selected,centers):
            old=data['states'][row['training_state_id']]
            torch.testing.assert_close(state['energy_eV'],old['energy_eV'],atol=1e-4,rtol=0)
            torch.testing.assert_close(state['force_eV_A'],old['force_eV_A'],atol=1e-4,rtol=0)
        probes=[]
        for context,(row,state) in enumerate(zip(selected,centers)):
            leaf,anchor=row['leaf'],row['anchor'];x=state['positions'];radius=(x[leaf]-x[anchor]).norm();u=(x[leaf]-x[anchor])/radius
            noise=torch.randn(3,dtype=x.dtype,generator=rng);a=noise-(noise*u).sum()*u;a=a/a.norm()
            b=torch.cross(u,a,dim=0);diagonal=(a+b)/math.sqrt(2)
            specifications=[('fit',a,1),('fit',a,-1),('fit',b,1),('fit',b,-1),('check',diagonal,1),('check',diagonal,-1)]
            for number,(role,axis,sign) in enumerate(specifications):
                angle=protocol['fit_angle_rad'] if role=='fit' else protocol['check_angle_rad']
                v=math.cos(angle)*u+sign*math.sin(angle)*axis
                y=x.clone();y[leaf]=x[anchor]+radius*v;y-=y.mean(0)
                item=dict(context=context,number=number,role=role,angle_rad=angle,axis=axis,sign=sign,positions=y,valid=False,state_id=None)
                try:
                    proposed=target.coordinate_state(y)
                    if not torch.equal(proposed['graph']['bond_orders'],state['graph']['bond_orders']):raise ValueError('Probe changed the conditioning graph')
                    item['valid']=True;probes.append((item,proposed))
                except ValueError as exc:item['failure']=str(exc)
                probe_rows.append(item)
        target.evaluate([state for _,state in probes],phase='angular_probes')
        for row,state in probes:row['state_id']=state['state_id']
        models=[]
        for replica in [0,1]:
            path=args.project/f'runs/normalized_site_train_v1/vector_s{replica}/model.pt'
            assert sha(path)==protocol['model_sha256'][replica]
            saved=torch.load(path,map_location='cpu',weights_only=False)
            model=NormalizedSiteGuide(**saved['configuration']).double();model.load_state_dict(saved['state_dict']);model.eval();models.append(model)
        numbers=torch.tensor(target.numbers,dtype=torch.long);electronic=torch.tensor([target.condition['charge'],target.condition['spin_multiplicity'],target.kT],dtype=torch.float64)
        diagnostics=[]
        for context,(row,center) in enumerate(zip(selected,centers)):
            leaf,anchor=row['leaf'],row['anchor'];root_index=torch.tensor([[leaf,anchor]])
            masked,radius,roles=masked_angular_context(center['positions'][None],root_index)
            site=physical_site_parameter(masked,roles,center['graph']['bond_orders'][None],root_index,10.)[0]
            harmonic=confinement_parameter(masked,radius,electronic[2:3],target.restraint)[0]
            fit=[center]+[target.states[r['state_id']] for r in probe_rows if r['context']==context and r['role']=='fit' and r['valid']]
            fit_scores=[score(s,leaf,anchor,target.kT,target.restraint) for s in fit]
            matrices=torch.stack([torch.eye(3,dtype=torch.float64)-u[:,None]*u[None] for u,_ in fit_scores])
            design=matrices.reshape(-1,3);rhs=torch.stack([s for _,s in fit_scores]).flatten()
            eigenvalues=torch.linalg.eigvalsh(design.T@design)
            item=dict(context=context,parent_id=row['parent_id'],fit_points=len(fit),design_eigenvalues=eigenvalues.tolist(),
                full_rank=bool(eigenvalues[0]>1e-5),checks=[])
            if item['full_rank']:
                fitted=torch.linalg.lstsq(design,rhs).solution
                residual=fitted-site-harmonic
                capped=site+harmonic+residual*min(1.,64/float(residual.norm().clamp_min(1e-20)))
                with torch.no_grad():
                    old=[m(center['positions'][None],center['graph']['bond_orders'][None],numbers,electronic,root_index)[0][0,0] for m in models]
                item.update(fitted_parameter=fitted.tolist(),fitted_residual_norm=float(residual.norm()),
                    old_residual_norms=[float((eta-site).norm()) for eta in old])
                u0,_=score(center,leaf,anchor,target.kT,target.restraint)
                for probe in probe_rows:
                    if probe['context']!=context or probe['role']!='check' or not probe['valid']:continue
                    state=target.states[probe['state_id']];u,s=score(state,leaf,anchor,target.kT,target.restraint)
                    params={'site':site,'site_confinement':site+harmonic,'local_oracle_vmf':fitted,'local_oracle_capped64':capped,'frozen_s0':old[0],'frozen_s1':old[1]}
                    errors={name:float((eta-(eta*u).sum()*u-s).square().mean()) for name,eta in params.items()}
                    item['checks'].append(dict(number=probe['number'],force_score_mse=errors,
                        actual_work_over_kT=float((state['potential_eV']-center['potential_eV'])/target.kT),
                        fitted_work_over_kT=-float(fitted@(u-u0)),score_norm=float(s.norm())))
            diagnostics.append(item)
        torch.save(dict(states=target.states,query_trace=target.query_trace,probes=probe_rows,selected=selected,
            generator_state=rng.get_state()),args.out/'trace.pt')
        assert oracle.evaluated==oracle.requested_evaluations and oracle.evaluated<=protocol['maximum_raw_queries']
        report.update(complete=True,diagnostics=diagnostics,probe_attempts=len(probe_rows),supported_probes=len(probes),
            new_raw_queries=oracle.evaluated,requested_raw_queries=oracle.requested_evaluations,trace_sha256=sha(args.out/'trace.pt'),
            full_rank_contexts=sum(d['full_rank'] for d in diagnostics),oracle_runtime=oracle.handshake,
            scope='Exploratory fixed-context TRAINING-only stiffness and off-axis force/work diagnostics. Local least-squares vMF is an approximation, not a sampling result or a new physical law.')
        write(output,report)
    except Exception as exc:
        if target is not None:torch.save(dict(states=target.states,query_trace=target.query_trace,probes=probe_rows,selected=selected),args.out/'failed_trace.pt')
        report.update(failure=f'{type(exc).__name__}: {exc}',new_raw_queries=oracle.evaluated if oracle else 0,
            requested_raw_queries=oracle.requested_evaluations if oracle else 0);write(output,report);raise
    finally:
        if oracle is not None:oracle.close()


if __name__=='__main__':main()
