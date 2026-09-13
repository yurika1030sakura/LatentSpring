#!/usr/bin/env python3
"""Actual molecular proposal comparison on12 fixed internal evaluation parents."""
import argparse,json,math,time
from pathlib import Path
import torch
from cfm_mol.chemical_sampler import ChemicalTarget
from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.edit_conditioned_bridge import EditBridgeField,edit_bridge,center
from cfm_mol.edit_bridge_sampler import ZeroBridgeField,AnalyticBridgeField,propose_edit,finish_edit
from cfm_mol.joint_chemical_geometry import distinct_anchor_actions
from scripts.research.audit_masked_angular import ReplayOracle,equal,sha
from scripts.research.audit_joint_arc_support import independent_arc_q
from scripts.research.evaluate_chemical_policy import write


def inputs(root,project,index,protocol_path=None):
    pp=protocol_path or root/'research/evidence/edit_bridge_evaluation_protocol_v1.json';protocol=json.loads(pp.read_text());assert protocol['frozen']
    physical_path=root/protocol['physical_protocol'];assert sha(physical_path)==protocol['physical_protocol_sha256'];physical=json.loads(physical_path.read_text())
    data=None
    if protocol.get('source_kind')!='prepared_evaluation':
        dp=project/protocol['data_run'];assert sha(dp/'data.pt')==protocol['data_sha256'];data=torch.load(dp/'data.pt',map_location='cpu',weights_only=False)
    selected=[];condition=None;cache={}
    for spec in protocol['sources'][str(index)]:
        if protocol.get('source_kind')=='prepared_evaluation':
            path=project/spec['trace'];rp=path.parent/'results.json';ap=project/spec['audit']
            r=json.loads(rp.read_text());audit=json.loads(ap.read_text())
            assert r['complete'] and r['assigned_role']=='evaluation_only' and audit['complete'] and audit['full_replay']
            assert sha(rp)==spec['results_sha256']==audit['source_results_sha256'] and sha(ap)==spec['audit_sha256']
            assert sha(path)==spec['trace_sha256']==audit['trace_sha256']
            if path not in cache:cache[path]=torch.load(path,map_location='cpu',weights_only=False)
            source=cache[path];assert source['parent_ids'][spec['parent_offset']]==spec['parent']
            assert source['history_state_ids'][-1][spec['parent_offset']]==spec['state_id']
            state=source['states'][spec['state_id']];c=r['condition']
            if condition is None:condition=c
            assert condition==c;selected.append((spec,state));continue
        row=data[spec['data_index']];assert row['role']=='withheld_parent' and row['index']==index and row['parent']==spec['parent'] and row['replica']==0
        assert row['step']==spec['step'] and row['force_source_trace']==spec['trace'] and row['source_force_provenance']['state_id']==spec['state_id']
        path=project/spec['trace']
        if path not in cache:
            assert sha(path)==spec['trace_sha256'];cache[path]=torch.load(path,map_location='cpu',weights_only=False)
        state=cache[path]['states'][spec['state_id']];torch.testing.assert_close(state['positions'],row['x'],atol=0,rtol=0)
        c=dict(numbers=row['numbers'].tolist(),charge=int(row['electronic'][0]),spin_multiplicity=int(row['electronic'][1]))
        if condition is None:condition=c
        assert condition==c;selected.append((spec,state))
    assert len(selected)==3
    models={'physical_arc':None,'zero_bridge':ZeroBridgeField(),'analytic_bridge':AnalyticBridgeField(**protocol['analytic_field'])}
    for name,spec in protocol['models'].items():
        path=project/spec['path'];assert sha(path)==spec['sha256'];saved=torch.load(path,map_location='cpu',weights_only=False)
        if spec.get('architecture')=='mobility':
            from cfm_mol.edit_mobility import MobilizedEditField
            model=MobilizedEditField(**saved['configuration']).double()
        else:model=EditBridgeField(**saved['configuration']).double()
        model.load_state_dict(saved['state_dict']);model.eval();model.requires_grad_(False);models[name]=model
    return pp,protocol,physical,models,condition,selected


def run(target,models,sources,protocol,index):
    initial=[target.coordinate_state(old['positions']) for spec,old in sources];target.evaluate(initial,phase='shared_initial_sources')
    for state,(spec,old) in zip(initial,sources):
        equal(state['graph'],old['graph'])
        assert abs(float(state['potential_eV']-old['potential_eV']))<=protocol['source_energy_tolerance_eV']
        assert float((state['force_eV_A']-old['force_eV_A']).abs().max())<=protocol['source_force_tolerance_eV_A']
    initial_calls=target.oracle.evaluated;plan=[]
    for offset,(spec,old) in enumerate(sources):
        actions=distinct_anchor_actions(target.numbers,initial[offset]['graph']['bond_orders']);assert actions
        for trial in range(protocol['trials_per_source']):
            seed=protocol['seed']+1000000*index+1000*spec['parent']+trial;g=torch.Generator().manual_seed(seed)
            action=actions[int(torch.randint(len(actions),(1,),generator=g))]
            momentum=center(torch.randn(old['positions'].shape,dtype=torch.float64,generator=g));order=int(torch.randint(2,(1,),generator=g))
            log_uniform=float(torch.rand((),dtype=torch.float64,generator=g).log())
            plan.append(dict(source=offset,parent=spec['parent'],trial=trial,seed=seed,action=action,momentum=momentum,order=order,log_uniform=log_uniform,arc_seed=seed+100000000))
    attempts=[];timing={}
    for method in protocol['method_order'][str(index)]:
        start=time.monotonic();pending=[];rows=[]
        for item in plan:
            candidate,row=propose_edit(target,initial[item['source']],item['action'],item['momentum'],item['order'],item['arc_seed'],
                method=method,field=models[method],bridge_options=protocol['bridge'],arc_options=protocol['arc_options'])
            row.update(**{k:item[k] for k in ['source','parent','trial','seed']});rows.append((candidate,row,item))
            if candidate is not None:pending.append(candidate)
        proposal_seconds=time.monotonic()-start;start=time.monotonic();before=target.oracle.evaluated
        target.evaluate(pending,phase=method+'/candidates');physical_seconds=time.monotonic()-start
        assert target.oracle.evaluated-before==2*len(pending)
        for candidate,row,item in rows:attempts.append(finish_edit(target,initial[item['source']],candidate,row,item['log_uniform']))
        timing[method]=dict(proposal_seconds=proposal_seconds,physical_dispatch_seconds=physical_seconds,raw_calls=2*len(pending))
    return dict(attempts=attempts,states=target.states,query_trace=target.query_trace,initial_raw_queries=initial_calls,plan=plan),timing


def audit_arithmetic(saved,target,models,protocol):
    checked=inverses=0;by_method={}
    for row in saved['attempts']:
        if not row['scored']:continue
        old=saved['states'][row['old_state_id']];new=saved['states'][row['new_state_id']];q=saved['query_trace'][new['query_batch']];j=new['query_row']
        energy=(float(q['raw_energy_eV'][j])+float(q['inverted_energy_eV'][j]))/2+target.restraint/2*float(new['positions'].square().sum())
        delta=energy-float(old['potential_eV'])
        if row['method']=='physical_arc':correction=independent_arc_q(row['reverse'])-independent_arc_q(row['forward'])
        else:
            correction=.5*float(row['input_momentum'].square().sum()-row['output_momentum'].square().sum())+row['log_volume']
            count=by_method.get(row['method'],0)
            if count<protocol['inverse_checks_per_method_condition']:
                recovered,p,_=edit_bridge(new['positions'],row['output_momentum'],new['graph']['bond_orders'],old['graph']['bond_orders'],
                    torch.tensor(target.numbers),old['positions'].new_tensor([old['charge'],old['spin_multiplicity'],target.kT]),
                    row['inverse_action'],models[row['method']],**protocol['bridge'])
                torch.testing.assert_close(recovered,old['positions'],atol=1e-9,rtol=0);torch.testing.assert_close(p,row['input_momentum'],atol=1e-9,rtol=0);inverses+=1
            by_method[row['method']]=count+1
        forward=len(distinct_anchor_actions(target.numbers,old['graph']['bond_orders']));reverse=len(distinct_anchor_actions(target.numbers,new['graph']['bond_orders']))
        ratio=-delta/target.kT+correction+math.log(forward/reverse)
        assert abs(ratio-row['log_acceptance_ratio'])<1e-7
        assert abs(-math.exp(min(0.,ratio))*delta-row['expected_utility_eV'])<1e-8
        assert row['accepted']==(row['log_uniform']<min(0.,ratio));checked+=1
    return dict(independent_MH_checks=checked,trained_and_control_inverse_checks=inverses)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','out']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['run','oracle-python','oracle-checkpoint']:p.add_argument('--'+name,type=Path)
    p.add_argument('--protocol',type=Path)
    p.add_argument('--index',type=int,required=True);p.add_argument('--phase',choices=['evaluate','audit'],required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[2];pp,protocol,physical,models,condition,sources=inputs(root,a.project,a.index,root/a.protocol if a.protocol else None)
    a.out.mkdir(parents=True,exist_ok=True);output=a.out/'results.json'
    if output.exists():raise FileExistsError(output)
    report=dict(complete=False,index=a.index,phase=a.phase,protocol_sha256=sha(pp),condition=condition,sources=protocol['sources'][str(a.index)],new_raw_queries=0,
        scientific_submission_ready=False,scope=protocol['interpretation']);write(output,report);oracle=target=None;start=time.monotonic()
    try:
        if a.phase=='audit':
            producer=json.loads((a.run/'results.json').read_text());assert producer['complete'] and producer['protocol_sha256']==sha(pp) and sha(a.run/'trace.pt')==producer['trace_sha256']
            expected=torch.load(a.run/'trace.pt',map_location='cpu',weights_only=False);oracle=ReplayOracle(expected['query_trace'])
        else:
            assert sha(a.oracle_checkpoint)==physical['raw_oracle_sha256']
            oracle=EnergyOracle(a.oracle_python,root/'scripts/research/oracle_worker.py',a.oracle_checkpoint,numbers=condition['numbers'],charge=condition['charge'],
                spin_multiplicity=condition['spin_multiplicity'],device='cuda',batch_size=32)
            assert oracle.handshake['base_precision_dtype']=='torch.float32' and not oracle.handshake['tf32']
        target=ChemicalTarget(oracle,condition,physical['kT_eV'],physical['restraint_eV_A2']);actual,timing=run(target,models,sources,protocol,a.index)
        assert oracle.evaluated==actual['initial_raw_queries']+sum(r['raw_cost'] for r in actual['attempts'])
        assert len(actual['attempts'])==3*protocol['trials_per_source']*len(protocol['methods'])
        if a.phase=='audit':
            equal(actual,expected);assert oracle.index==len(oracle.queries) and oracle.evaluated==producer['new_raw_queries']
            report.update(complete=True,full_replay=True,source_results_sha256=sha(a.run/'results.json'),trace_sha256=producer['trace_sha256'],
                raw_queries_in_producer=oracle.evaluated,**audit_arithmetic(actual,target,models,protocol))
        else:
            assert oracle.evaluated==oracle.requested_evaluations<=protocol['maximum_new_raw_queries']//4
            torch.save(actual,a.out/'trace.pt');compact=[{k:r[k] for k in ['method','source','parent','trial','seed','valid','scored','accepted','raw_cost','expected_utility_eV','expected_constitutional_flow','expected_invariant_jump_A2']} for r in actual['attempts']]
            report.update(complete=True,rows=compact,new_raw_queries=oracle.evaluated,requested_raw_queries=oracle.requested_evaluations,initial_raw_queries=actual['initial_raw_queries'],
                method_timing=timing,oracle_evaluation_seconds=oracle.evaluation_seconds,oracle_runtime=oracle.handshake,trace_sha256=sha(a.out/'trace.pt'))
        report['elapsed_seconds']=time.monotonic()-start;write(output,report)
        print(json.dumps({k:v for k,v in report.items() if k not in ['rows','sources','condition','oracle_runtime']}))
    except Exception as exc:
        if target is not None:torch.save(dict(states=target.states,query_trace=target.query_trace),a.out/'failed_trace.pt')
        report.update(failure=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-start)
        if oracle is not None and a.phase=='evaluate':report.update(new_raw_queries=oracle.evaluated,requested_raw_queries=oracle.requested_evaluations)
        write(output,report);raise
    finally:
        if oracle is not None and a.phase=='evaluate':oracle.close()


if __name__=='__main__':main()
