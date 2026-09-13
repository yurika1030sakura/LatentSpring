#!/usr/bin/env python3
"""Finite-work labels for every eligible bare edit on original FIT sources only."""
import argparse,json
from pathlib import Path
import torch
from cfm_mol.chemical_sampler import ChemicalTarget
from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.edit_bridge_sampler import ZeroBridgeField,propose_edit
from cfm_mol.joint_chemical_geometry import distinct_anchor_actions
from scripts.research.audit_masked_angular import ReplayOracle,equal,sha
from scripts.research.evaluate_chemical_policy import write


def build(root,project,out):
    dp=project/'runs/screen_force_pairs_v1';header=json.loads((dp/'results.json').read_text());assert header['complete'] and sha(dp/'data.pt')==header['data_sha256']
    data=torch.load(dp/'data.pt',map_location='cpu',weights_only=False);parents=sorted({(r['index'],r['parent']) for r in data if r['role']=='fit'});assert len(parents)==36
    sources=[];catalogue=[];cache={};physical=json.loads((root/'research/evidence/parity_training_protocol_v1.json').read_text())
    for index,parent in parents:
        offset,row=min(((i,r) for i,r in enumerate(data) if r['index']==index and r['parent']==parent and r['role']=='fit' and r['replica']==0),key=lambda item:item[1]['step'])
        path=project/row['force_source_trace'];assert sha(path)==row['source_trace_sha256']
        if path not in cache:cache[path]=torch.load(path,map_location='cpu',weights_only=False)
        old=cache[path]['states'][row['source_force_provenance']['state_id']];torch.testing.assert_close(old['positions'],row['x'],atol=0,rtol=0)
        condition=dict(numbers=row['numbers'].tolist(),charge=old['charge'],spin_multiplicity=old['spin_multiplicity']);target=ChemicalTarget(None,condition,physical['kT_eV'],physical['restraint_eV_A2'])
        equal(target.coordinate_state(old['positions'])['graph'],old['graph']);source_id=len(sources)
        sources.append(dict(source_id=source_id,index=index,parent=parent,fit_only=True,condition=condition,state=old,data_index=offset,source_trace=row['force_source_trace'],source_trace_sha256=row['source_trace_sha256']))
        actions=distinct_anchor_actions(target.numbers,old['graph']['bond_orders']);assert actions
        for action in actions:
            candidate,record=propose_edit(target,old,action,torch.zeros_like(old['positions']),0,0,method='bare_edit',field=ZeroBridgeField(),
                bridge_options=dict(steps_per_side=0,kick_step=.2,drift_step=.02),arc_options={})
            catalogue.append(dict(source_id=source_id,index=index,parent=parent,action=action,record=record,candidate=candidate))
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);torch.save(dict(sources=sources,catalogue=catalogue),out/'plan.pt')
    counts={str(i):dict(sources=sum(s['index']==i for s in sources),actions=sum(r['index']==i for r in catalogue),valid=sum(r['index']==i and r['candidate'] is not None for r in catalogue)) for i in [1,2,3,5]}
    for c in counts.values():c['maximum_raw_queries']=2*(c['sources']+c['valid'])
    write(out/'results.json',dict(complete=True,plan_sha256=sha(out/'plan.pt'),source_data_sha256=header['data_sha256'],counts=counts,maximum_raw_queries=sum(c['maximum_raw_queries'] for c in counts.values()),
        new_physical_queries=0,selection='All36 original FIT parents, earliest replica0 source, all eligible bare terminal edits. Preserve invalid actions. No energy-based selection or evaluation parents.'))


def score(target,sources,catalogue):
    initial=[target.coordinate_state(s['state']['positions']) for s in sources];target.evaluate(initial,phase='catalogue_sources');mapping={s['source_id']:v for s,v in zip(sources,initial)}
    for s,v in zip(sources,initial):
        equal(v['graph'],s['state']['graph']);assert abs(float(v['potential_eV']-s['state']['potential_eV']))<1e-4
    pending=[];pairs=[]
    for spec in catalogue:
        old=mapping[spec['source_id']]
        candidate,record=propose_edit(target,old,tuple(spec['action']),torch.zeros_like(old['positions']),0,0,method='bare_edit',field=ZeroBridgeField(),
            bridge_options=dict(steps_per_side=0,kick_step=.2,drift_step=.02),arc_options={})
        assert (candidate is None)==(spec['candidate'] is None)
        if candidate is not None:torch.testing.assert_close(candidate['positions'],spec['candidate']['positions'],atol=1e-12,rtol=0);pending.append(candidate)
        pairs.append((spec,old,candidate,record))
    target.evaluate(pending,phase='all_valid_bare_candidates');result=[]
    for spec,old,new,record in pairs:
        row=dict(source_id=spec['source_id'],parent=spec['parent'],action=spec['action'],valid=new is not None,source_state_id=old['state_id'],record=record)
        if new is not None:
            delta=float(new['potential_eV']-old['potential_eV']);harmonic=target.restraint/2*float(new['positions'].square().sum()-old['positions'].square().sum())
            linear=-float((old['force_eV_A']*(new['positions']-old['positions'])).sum())+harmonic
            row.update(candidate_state_id=new['state_id'],potential_change_eV=delta,linear_prediction_eV=linear,log_volume=record['log_volume'],
                base_log_ratio=-delta/target.kT+record['log_volume'],connectivity_changed=old['graph']['connectivity_smiles']!=new['graph']['connectivity_smiles'])
        result.append(row)
    return dict(rows=result,states=target.states,query_trace=target.query_trace)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','out']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['run','oracle-python','oracle-checkpoint']:p.add_argument('--'+name,type=Path)
    p.add_argument('--index',type=int);p.add_argument('--phase',choices=['build','score','audit'],required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[2]
    if a.phase=='build':build(root,a.project,a.out);return
    pp=root/'research/evidence/chemical_work_catalogue_protocol_v1.json';protocol=json.loads(pp.read_text());assert protocol['frozen'] and a.index in protocol['condition_indices']
    directory=a.project/protocol['plan_run'];assert sha(directory/'plan.pt')==protocol['plan_sha256'];plan=torch.load(directory/'plan.pt',map_location='cpu',weights_only=False)
    sources=[s for s in plan['sources'] if s['index']==a.index];catalogue=[r for r in plan['catalogue'] if r['index']==a.index];condition=sources[0]['condition']
    assert len(sources)==9 and all(s['fit_only'] and s['condition']==condition for s in sources)
    physical_path=root/protocol['physical_protocol'];assert sha(physical_path)==protocol['physical_protocol_sha256'];physical=json.loads(physical_path.read_text())
    a.out.mkdir(parents=True,exist_ok=True);output=a.out/'results.json'
    if output.exists():raise FileExistsError(output)
    report=dict(complete=False,index=a.index,phase=a.phase,protocol_sha256=sha(pp),new_raw_queries=0,scientific_submission_ready=False,scope=protocol['interpretation']);write(output,report);oracle=target=None
    try:
        if a.phase=='audit':
            producer=json.loads((a.run/'results.json').read_text());assert producer['complete'] and producer['protocol_sha256']==sha(pp) and sha(a.run/'trace.pt')==producer['trace_sha256']
            expected=torch.load(a.run/'trace.pt',map_location='cpu',weights_only=False);oracle=ReplayOracle(expected['query_trace'])
        else:
            assert sha(a.oracle_checkpoint)==physical['raw_oracle_sha256']
            oracle=EnergyOracle(a.oracle_python,root/'scripts/research/oracle_worker.py',a.oracle_checkpoint,numbers=condition['numbers'],charge=condition['charge'],spin_multiplicity=condition['spin_multiplicity'],device='cuda',batch_size=32)
            assert oracle.handshake['base_precision_dtype']=='torch.float32' and not oracle.handshake['tf32']
        target=ChemicalTarget(oracle,condition,physical['kT_eV'],physical['restraint_eV_A2']);actual=score(target,sources,catalogue)
        assert oracle.evaluated==protocol['counts'][str(a.index)]['maximum_raw_queries']
        if a.phase=='audit':
            equal(actual,expected);assert oracle.index==len(oracle.queries) and oracle.evaluated==producer['new_raw_queries']
            report.update(complete=True,full_replay=True,source_results_sha256=sha(a.run/'results.json'),trace_sha256=producer['trace_sha256'],raw_queries_in_producer=oracle.evaluated)
        else:
            assert oracle.evaluated==oracle.requested_evaluations
            torch.save(actual,a.out/'trace.pt');report.update(complete=True,new_raw_queries=oracle.evaluated,requested_raw_queries=oracle.requested_evaluations,trace_sha256=sha(a.out/'trace.pt'),sources=len(sources),actions=len(catalogue),valid=sum(r['valid'] for r in actual['rows']),oracle_runtime=oracle.handshake)
        write(output,report);print(json.dumps({k:v for k,v in report.items() if k!='oracle_runtime'}))
    except Exception as exc:
        if target is not None:torch.save(dict(states=target.states,query_trace=target.query_trace),a.out/'failed_trace.pt')
        report.update(failure=f'{type(exc).__name__}: {exc}')
        if oracle is not None and a.phase=='score':report.update(new_raw_queries=oracle.evaluated,requested_raw_queries=oracle.requested_evaluations)
        write(output,report);raise
    finally:
        if oracle is not None and a.phase=='score':oracle.close()


if __name__=='__main__':main()
