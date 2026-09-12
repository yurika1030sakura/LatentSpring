#!/usr/bin/env python3
"""Collect bounded fixed-context angular labels on disjoint training compositions."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from cfm_mol.chemical_sampler import ChemicalTarget
from cfm_mol.conditional_angular_probe import angular_probes, angular_force, identify_parameter
from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.joint_chemical_geometry import distinct_anchor_actions, joint_geometry_proposal
from cfm_mol.local_site_guide import confinement_parameter
from cfm_mol.masked_angular_guide import masked_angular_context
from cfm_mol.normalized_site_guide import physical_site_parameter
from cfm_mol.terminal_rotation import terminal_rotation_actions
from scripts.research.evaluate_chemical_policy import sha, write


def load_inputs(root, project, index):
    pp=root/'research/evidence/multicomposition_angular_probe_protocol_v1.json'
    protocol=json.loads(pp.read_text())
    ap=project/protocol['preparation_audit'];audit=json.loads(ap.read_text())
    assert sha(ap)==protocol['preparation_audit_sha256'] and audit['complete'] and audit['full_producer_replay']
    assert audit['all_eight_conditions_retained'] and audit['protocol_sha256']==protocol['preparation_protocol_sha256']
    row=audit['rows'][index];assert row['index']==index
    manifest_path=root/'research/evidence/proposal_training_panel_v1.json'
    assert sha(manifest_path)==protocol['training_manifest_sha256']
    manifest=json.loads(manifest_path.read_text())
    nonempty=[r['index'] for r in audit['rows'] if r['parent_ids']]
    held=[]
    for lo,hi in [(8,12),(13,24)]:
        eligible=[i for i in nonempty if lo<=manifest['rows'][i]['n_atoms']<=hi]
        held.append(min(eligible,key=lambda i:hashlib.sha256(
            f"{protocol['split_seeds']['composition']}|{manifest['rows'][i]['composition_hex']}".encode()).hexdigest()))
    directory=project/protocol['preparation_run']/f'condition_{index:02d}'
    report=json.loads((directory/'results.json').read_text())
    assert report['complete'] and report['stream']=='fresh_training'
    assert sha(directory/'results.json')==row['results_sha256'] and report['parent_ids']==row['parent_ids']
    assert report['protocol_sha256']==protocol['preparation_protocol_sha256']
    split=protocol['condition_splits'][str(index)]
    combined=split['fit_parent_ids']+split['withheld_parent_ids']+split['withheld_composition_parent_ids']
    assert sorted(combined)==sorted(row['parent_ids']) and len(combined)==len(set(combined))
    order=sorted(row['parent_ids'],key=lambda pid:hashlib.sha256(
        f"{protocol['split_seeds']['parent']}|{index}|{pid}".encode()).hexdigest())
    expected=dict(role='zero_support' if not order else ('withheld_composition' if index in held else 'fit_composition'),
        fit_parent_ids=sorted(order[4:]) if index not in held else [],
        withheld_parent_ids=sorted(order[:4]) if index not in held else [],
        withheld_composition_parent_ids=row['parent_ids'] if index in held else [])
    assert split==expected
    physical_path=root/'research/evidence/parity_training_protocol_v1.json'
    assert sha(physical_path)==protocol['physical_protocol_sha256']
    physical=json.loads(physical_path.read_text())
    warm=None
    if row['parent_ids']:
        assert sha(directory/'trace.pt')==row['trace_sha256']==report['trace_sha256']
        warm=torch.load(directory/'trace.pt',map_location='cpu',weights_only=False)
        assert warm['parent_ids']==report['parent_ids'] and warm['condition']==report['condition']
    else:assert report['zero_support'] and row['zero_support']
    return pp,protocol,report,warm,physical


def collect(target, warm, protocol, index, progress):
    rng=torch.Generator().manual_seed(protocol['probe_seed']+100003*index)
    numbers=torch.tensor(target.numbers,dtype=torch.long)
    electronic=torch.tensor([target.condition['charge'],target.condition['spin_multiplicity'],target.kT],dtype=torch.float64)
    parents=warm['parent_ids'];warm_states=[]
    assert not target.states and not target.query_trace
    assert protocol['warm_roots_per_parent']==1 and protocol['exchange_trials_per_parent']==2
    progress.update(parent_ids=parents,inherited_warm_states=[],warm_root_choices=[],exchange_trials=[],contexts=[],probes=[])
    for parent,old_id in zip(parents,warm['history_state_ids'][-1]):
        old=warm['states'][old_id];state=target.coordinate_state(old['positions'])
        torch.testing.assert_close(state['graph']['bond_orders'],old['graph']['bond_orders'])
        for key in ['energy_eV','force_eV_A','potential_eV','score']:state[key]=old[key].clone()
        state.update(state_id=len(target.states),query_batch=None,query_row=None,inherited_preparation_state_id=old_id)
        target.states.append(state);warm_states.append(state)
        progress['inherited_warm_states'].append(dict(parent_id=parent,state_id=state['state_id'],preparation_state_id=old_id))
    contexts=[];candidates=[]
    for parent,old in zip(parents,warm_states):
        roots=terminal_rotation_actions(target.numbers,old['graph']['bond_orders'])
        choice=dict(parent_id=parent,state_id=old['state_id'],eligible_roots=len(roots))
        if roots:
            pick=int(torch.randint(len(roots),(1,),generator=rng));root=roots[pick]
            choice.update(choice_index=pick,root=root)
            contexts.append((dict(parent_id=parent,kind='warm',root=root,trial=None),old))
        else:choice['failure']='No eligible warm root'
        progress['warm_root_choices'].append(choice)
        for trial in range(protocol['exchange_trials_per_parent']):
            actions=distinct_anchor_actions(numbers,old['graph']['bond_orders'])
            row=dict(parent_id=parent,trial=trial,warm_state_id=old['state_id'],forward_count=len(actions),valid=False,new_state_id=None)
            progress['exchange_trials'].append(row)
            if not actions:
                row['failure']='No eligible different-anchor exchange';continue
            pick=int(torch.randint(len(actions),(1,),generator=rng));action=actions[pick]
            order=int(torch.randint(2,(1,),generator=rng))
            y,q,forward=joint_geometry_proposal(old['positions'],old['graph']['bond_orders'],numbers,electronic,
                target.radii,action,kind='site',order=order,generator=rng,
                radial_width=protocol['radial_width'],site_concentration=protocol['site_concentration'])
            i,j,k,l=action;inverse=(i,j,l,k)
            row.update(choice_index=pick,action=action,order=order,proposal_positions=y,forward=forward,log_forward_coordinate=q)
            try:
                candidate=target.coordinate_state(y)
                if not torch.equal(candidate['graph']['bond_orders'],forward['desired_bonds']):
                    raise ValueError('Exchange endpoint differs from the desired graph')
                if inverse not in distinct_anchor_actions(numbers,candidate['graph']['bond_orders']):
                    raise ValueError('Inverse exchange is ineligible')
                recovered,qr,reverse=joint_geometry_proposal(y,candidate['graph']['bond_orders'],numbers,electronic,
                    target.radii,inverse,kind='site',order=order,observed=old['positions'],
                    radial_width=protocol['radial_width'],site_concentration=protocol['site_concentration'])
                torch.testing.assert_close(recovered,old['positions'],atol=1e-9,rtol=1e-9)
                row.update(valid=True,inverse_action=inverse,reverse=reverse,log_reverse_coordinate=qr)
                candidates.append((row,candidate))
                for root in [(i,l),(j,k)]:contexts.append((dict(parent_id=parent,kind='exchange',root=root,trial=trial),candidate))
            except ValueError as exc:row['failure']=str(exc)
    target.evaluate([state for _,state in candidates],phase='exchange_centers')
    for row,state in candidates:row['new_state_id']=state['state_id']
    probes=[]
    for cid,(context,center) in enumerate(contexts):
        descriptor=dict(context,context=cid,center_state_id=center['state_id'])
        generated,noise=angular_probes(center['positions'],context['root'],generator=rng,
            fit_angle=protocol['fit_angle_rad'],check_angle=protocol['check_angle_rad'])
        descriptor['frame_noise']=noise;progress['contexts'].append(descriptor)
        for item in generated:
            row=dict(item,context=cid,valid=False,state_id=None)
            progress['probes'].append(row)
            try:
                candidate=target.coordinate_state(row['positions'])
                if not torch.equal(candidate['graph']['bond_orders'],center['graph']['bond_orders']):
                    raise ValueError('Angular probe changed the conditioning graph')
                row['valid']=True;probes.append((row,candidate))
            except ValueError as exc:row['failure']=str(exc)
    target.evaluate([state for _,state in probes],phase='angular_probes')
    for row,state in probes:row['state_id']=state['state_id']
    progress.update(generator_state=rng.get_state(),condition=target.condition,
        states=target.states,query_trace=target.query_trace)
    assert target.oracle.evaluated==2*(len(candidates)+len(probes))
    assert len(contexts)<=protocol['maximum_contexts_per_parent']*len(parents)
    assert target.oracle.evaluated<=protocol['maximum_new_raw_queries_per_parent']*len(parents)


def diagnose(saved, physical, protocol):
    diagnostics=[];states=saved['states']
    for context in saved['contexts']:
        cid=context['context'];root=context['root'];center=states[context['center_state_id']]
        fit_ids=[center['state_id']]+[r['state_id'] for r in saved['probes'] if r['context']==cid and r['role']=='fit' and r['valid']]
        obs=[angular_force(states[i],root,physical['kT_eV'],physical['restraint_eV_A2']) for i in fit_ids]
        item=identify_parameter(torch.stack([v[0] for v in obs]),torch.stack([v[1] for v in obs]),rank_threshold=protocol['rank_threshold'])
        item.update(context=cid,parent_id=context['parent_id'],kind=context['kind'],root=root,fit_state_ids=fit_ids,checks=[])
        if item['full_rank']:
            eta=torch.tensor(item['fitted_parameter'],dtype=torch.float64)
            masked,radius,roles=masked_angular_context(center['positions'][None],torch.tensor([root]))
            site=physical_site_parameter(masked,roles,center['graph']['bond_orders'][None],torch.tensor([root]),64.)[0]
            harmonic=confinement_parameter(masked,radius,torch.tensor([physical['kT_eV']],dtype=torch.float64),physical['restraint_eV_A2'])[0]
            item.update(fitted_concentration=float(eta.norm()),center_axial_parameter=float(eta@obs[0][0]),
                physical_site_parameter=site.tolist(),confinement_parameter=harmonic.tolist())
            for row in saved['probes']:
                if row['context']!=cid or row['role']!='check' or not row['valid']:continue
                state=states[row['state_id']];u,score=angular_force(state,root,physical['kT_eV'],physical['restraint_eV_A2'])
                params={'fitted':eta,'site64':site,'site64_confinement':site+harmonic}
                mse={name:float((parameter-(parameter*u).sum()*u-score).square().mean()) for name,parameter in params.items()}
                actual=float((state['potential_eV']-center['potential_eV'])/physical['kT_eV'])
                works={name:-float(parameter@(u-obs[0][0])) for name,parameter in params.items()}
                item['checks'].append(dict(number=row['number'],state_id=row['state_id'],force_score_mse=mse,
                    actual_work_over_kT=actual,predicted_work_over_kT=works))
        diagnostics.append(item)
    return diagnostics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','out','oracle-python','oracle-checkpoint']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--index',type=int,choices=range(8),required=True)
    args=p.parse_args();root=Path(__file__).resolve().parents[2]
    pp,protocol,prepared,warm,physical=load_inputs(root,args.project,args.index)
    args.out.mkdir(parents=True,exist_ok=True);output=args.out/'results.json'
    if output.exists():raise FileExistsError(output)
    report=dict(complete=False,index=args.index,protocol_sha256=sha(pp),condition=prepared['condition'],
        parent_ids=prepared['parent_ids'],preparation_results_sha256=sha(args.project/protocol['preparation_run']/f'condition_{args.index:02d}/results.json'),
        preparation_audit_sha256=protocol['preparation_audit_sha256'],split=protocol['condition_splits'][str(args.index)],
        stream='fresh_training',new_raw_queries=0,model_fitted=False,scientific_submission_ready=False)
    write(output,report)
    if warm is None:
        report.update(complete=True,zero_support=True,requested_raw_queries=0,contexts=0,diagnostics=[])
        write(output,report);print(json.dumps(report));return
    assert sha(args.oracle_checkpoint)==physical['raw_oracle_sha256']
    oracle=target=None;progress={}
    try:
        oracle=EnergyOracle(args.oracle_python,root/'scripts/research/oracle_worker.py',args.oracle_checkpoint,
            numbers=prepared['condition']['numbers'],charge=prepared['condition']['charge'],
            spin_multiplicity=prepared['condition']['spin_multiplicity'],device='cuda',batch_size=32)
        assert oracle.handshake['base_precision_dtype']=='torch.float32' and not oracle.handshake['tf32']
        target=ChemicalTarget(oracle,prepared['condition'],physical['kT_eV'],physical['restraint_eV_A2'])
        collect(target,warm,protocol,args.index,progress)
        torch.save(progress,args.out/'trace.pt')
        assert oracle.evaluated==oracle.requested_evaluations
        diagnostics=diagnose(progress,physical,protocol)
        report.update(complete=True,zero_support=False,trace_sha256=sha(args.out/'trace.pt'),contexts=len(progress['contexts']),
            exchange_attempts=len(progress['exchange_trials']),supported_exchanges=sum(r['valid'] for r in progress['exchange_trials']),
            probe_attempts=len(progress['probes']),supported_probes=sum(r['valid'] for r in progress['probes']),
            full_rank_contexts=sum(d['full_rank'] for d in diagnostics),diagnostics=diagnostics,
            new_raw_queries=oracle.evaluated,requested_raw_queries=oracle.requested_evaluations,oracle_runtime=oracle.handshake,
            scope='Training-only fixed-context force/finite-work data. All single-draw proposal and probe failures retained. Local vMF teachers are approximations; no learner fit, molecular performance or equilibrium claim.')
        write(output,report);print(json.dumps({k:report[k] for k in ['index','contexts','supported_exchanges','supported_probes','full_rank_contexts','new_raw_queries']}))
    except Exception as exc:
        if target is not None:
            progress.update(states=target.states,query_trace=target.query_trace)
            torch.save(progress,args.out/'failed_trace.pt')
        report.update(failure=f'{type(exc).__name__}: {exc}',new_raw_queries=oracle.evaluated if oracle else 0,
            requested_raw_queries=oracle.requested_evaluations if oracle else 0)
        write(output,report);raise
    finally:
        if oracle is not None:oracle.close()


if __name__=='__main__':main()
