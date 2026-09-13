#!/usr/bin/env python3
"""Strict per-parent query-budget chains for the frozen mobility architecture."""
import argparse,json,math,time
from pathlib import Path
import torch
from cfm_mol.chemical_sampler import ChemicalTarget
from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.edit_conditioned_bridge import center
from cfm_mol.edit_bridge_sampler import propose_edit,finish_edit
from cfm_mol.joint_chemical_geometry import distinct_anchor_actions
from cfm_mol.terminal_rotation import uniform_internal_transition
from scripts.research.evaluate_edit_bridge import inputs,audit_arithmetic
from scripts.research.audit_masked_angular import ReplayOracle,equal,sha
from scripts.research.evaluate_chemical_policy import write


def run_chain(target,model,sources,method,model_name,replica,index,protocol,*,joint_kernel=None):
    start_count=target.oracle.evaluated
    states=target.evaluate([target.coordinate_state(s['positions']) for spec,s in sources],phase=f'{model_name}/r{replica}/initial')
    for state,(spec,old) in zip(states,sources):
        equal(state['graph'],old['graph']);assert abs(float(state['potential_eV']-old['potential_eV']))<=protocol['source_energy_tolerance_eV']
    counts=[2]*len(states);initial=[s['state_id'] for s in states];history=[initial];query_history=[list(counts)];attempts=[]
    cap=protocol['query_cap_per_parent'];start=time.monotonic()
    for step in range(protocol['maximum_microsteps']):
        active=[j for j,c in enumerate(counts) if c<cap]
        if not active:break
        kind=protocol['schedule'][step%len(protocol['schedule'])];pending=[];prepared=[]
        if kind=='force_rotation':
            for j in active:
                parent=sources[j][0]['parent'];seed=protocol['evaluation_seeds'][replica]+1000000*index+1000*parent+step
                before=target.oracle.evaluated
                proposed,rows=uniform_internal_transition(target,[states[j]],kind=kind,generator=torch.Generator().manual_seed(seed),phase=f'{model_name}/r{replica}/step_{step}/rotation_{j}')
                row=rows[0];row.update(scored=row['valid'],raw_cost=2*int(row['valid']),parent=parent,parent_offset=j,step=step,seed=seed)
                assert row['raw_cost']==target.oracle.evaluated-before
                states[j]=proposed[0];counts[j]+=row['raw_cost'];assert counts[j]<=cap;attempts.append(row)
            assert sum(counts)==target.oracle.evaluated-start_count
            history.append([s['state_id'] for s in states]);query_history.append(list(counts));continue
        for j in active:
            old=states[j];parent=sources[j][0]['parent'];seed=protocol['evaluation_seeds'][replica]+1000000*index+1000*parent+step
            g=torch.Generator().manual_seed(seed)
            if kind=='local':
                choice=int(torch.randint(len(protocol['local_scales']),(1,),generator=g));std=protocol['local_scales'][choice]*target.kT**.5
                new,row=target.propose(old,0,g,std);logu=float(torch.rand((),dtype=torch.float64,generator=g).log())
                row.update(local_scale_choice=choice,proposal_std=std)
            elif joint_kernel is not None:
                # The acceptance draw is independent of variable catalogue/panel
                # construction and common across all new comparison arms.
                logu=float(torch.rand((),dtype=torch.float64,generator=g).log())
                new,row=joint_kernel.propose(target,old,g,seed)
            else:
                actions=distinct_anchor_actions(target.numbers,old['graph']['bond_orders'])
                if actions:
                    action=actions[int(torch.randint(len(actions),(1,),generator=g))];p=center(torch.randn(old['positions'].shape,dtype=torch.float64,generator=g))
                    order=int(torch.randint(2,(1,),generator=g));logu=float(torch.rand((),dtype=torch.float64,generator=g).log())
                    new,row=propose_edit(target,old,action,p,order,seed+100000000,method=model_name,field=model,
                        bridge_options=dict(protocol['bridge'],**protocol.get('bridge_by_method',{}).get(model_name,{})),arc_options=protocol['arc_options'])
                else:
                    new=None;logu=float(torch.rand((),dtype=torch.float64,generator=g).log())
                    row=dict(method=model_name,old_state_id=old['state_id'],new_state_id=-1,valid=False,scored=False,accepted=False,raw_cost=0,failure_reason='No eligible chemical edit')
            row.update(kind=kind,parent=parent,parent_offset=j,step=step,seed=seed)
            prepared.append((j,old,new,row,logu))
            if new is not None:pending.append(new)
        before=target.oracle.evaluated;target.evaluate(pending,phase=f'{model_name}/r{replica}/step_{step}/{kind}')
        assert target.oracle.evaluated-before==2*len(pending)
        for j,old,new,row,logu in prepared:
            if kind=='local':
                target.finish_record(old,new,row,row['proposal_std']);ratio=float(row['base_log_ratio'])
                row.update(scored=new is not None,raw_cost=2*int(new is not None),log_acceptance_ratio=ratio,log_uniform=logu,accepted=new is not None and logu<min(0.,ratio))
            else:finish_edit(target,old,new,row,logu)
            if row['accepted']:states[j]=new
            counts[j]+=row['raw_cost'];assert counts[j]<=cap
            attempts.append(row)
        assert sum(counts)==target.oracle.evaluated-start_count
        history.append([s['state_id'] for s in states]);query_history.append(list(counts))
    result=dict(method=method,model_name=model_name,replica=replica,parent_ids=[spec['parent'] for spec,s in sources],initial_state_ids=initial,
        final_state_ids=[s['state_id'] for s in states],queries_per_parent=counts,cap_reached=[c==cap for c in counts],attempts=attempts,
        history_state_ids=history,query_count_history=query_history)
    return result,time.monotonic()-start


def run(target,models,sources,protocol,index):
    chains=[];timing={}
    for spec in protocol['arm_order'][str(index)]:
        method=spec['method'];replica=spec['replica'];name=method if method in models else f'mobility_{method}_s{replica}'
        chain,seconds=run_chain(target,models[name],sources,method,name,replica,index,protocol)
        chains.append(chain);timing[f'{method}_s{replica}']=seconds
    return dict(chains=chains,states=target.states,query_trace=target.query_trace),timing


def audit_chains(saved,target,models,protocol):
    local=rotations=0;joints=[]
    for chain in saved['chains']:
        counts=[2]*len(chain['parent_ids'])
        for row in chain['attempts']:
            counts[row['parent_offset']]+=row['raw_cost']
            if row['kind']=='joint':joints.append(row);continue
            if not row['scored']:continue
            old=saved['states'][row['old_state_id']];new=saved['states'][row['new_state_id']]
            if row['kind']=='force_rotation':
                def vmf(u,eta):
                    k=float(eta.norm());normalizer=math.log(4*math.pi)+(math.log(math.sinh(k)/k) if k<20 and k>1e-8 else k+math.log1p(-math.exp(-2*k))-math.log(2*k) if k>=20 else k*k/6)
                    return float((u*eta).sum())-normalizer
                correction=vmf(row['old_direction'],row['reverse_natural_parameter'])-vmf(row['new_direction'],row['forward_natural_parameter'])+math.log(row['forward_count']/row['reverse_count'])
                ratio=-float(new['potential_eV']-old['potential_eV'])/target.kT+correction
                assert abs(ratio-row['log_acceptance_ratio'])<1e-7;assert row['accepted']==(row['log_uniform']<min(0.,ratio));rotations+=1;continue
            std=row['proposal_std']
            z=(target.basis.T@old['positions']).flatten();y=(target.basis.T@new['positions']).flatten()
            correction=float(((y-row['forward_mean']).square().sum()-(z-row['reverse_mean']).square().sum())/(2*std**2))
            ratio=-float(new['potential_eV']-old['potential_eV'])/target.kT+correction
            assert abs(ratio-row['log_acceptance_ratio'])<1e-7;assert row['accepted']==(row['log_uniform']<min(0.,ratio));local+=1
        assert counts==chain['queries_per_parent']
    checked=audit_arithmetic(dict(saved,attempts=joints),target,models,protocol)
    for state in saved['states']:
        q=saved['query_trace'][state['query_batch']];j=state['query_row'];energy=.5*(float(q['raw_energy_eV'][j])+float(q['inverted_energy_eV'][j]))
        energy+=target.restraint/2*float(state['positions'].square().sum())
        assert abs(energy-float(state['potential_eV']))<1e-8
    return dict(independent_local_MH_checks=local,independent_rotation_MH_checks=rotations,independent_joint_MH_checks=checked['independent_MH_checks'],inverse_checks=checked['trained_and_control_inverse_checks'],all_physical_potentials_reconstructed=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','out']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['run','oracle-python','oracle-checkpoint']:p.add_argument('--'+name,type=Path)
    p.add_argument('--protocol',type=Path,default=Path('research/evidence/edit_mobility_chain_protocol_v1.json'))
    p.add_argument('--index',type=int,required=True);p.add_argument('--phase',choices=['evaluate','audit'],required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[2];pp=root/a.protocol
    _,protocol,physical,models,condition,sources=inputs(root,a.project,a.index,pp)
    a.out.mkdir(parents=True,exist_ok=True);output=a.out/'results.json'
    if output.exists():raise FileExistsError(output)
    report=dict(complete=False,index=a.index,phase=a.phase,protocol_sha256=sha(pp),condition=condition,new_raw_queries=0,scientific_submission_ready=False,scope=protocol['interpretation'])
    write(output,report);oracle=target=None;start=time.monotonic()
    try:
        if a.phase=='audit':
            producer=json.loads((a.run/'results.json').read_text());assert producer['complete'] and producer['protocol_sha256']==sha(pp) and sha(a.run/'trace.pt')==producer['trace_sha256']
            expected=torch.load(a.run/'trace.pt',map_location='cpu',weights_only=False);oracle=ReplayOracle(expected['query_trace'])
        else:
            assert sha(a.oracle_checkpoint)==physical['raw_oracle_sha256']
            oracle=EnergyOracle(a.oracle_python,root/'scripts/research/oracle_worker.py',a.oracle_checkpoint,numbers=condition['numbers'],charge=condition['charge'],spin_multiplicity=condition['spin_multiplicity'],device='cuda',batch_size=32)
            assert oracle.handshake['base_precision_dtype']=='torch.float32' and not oracle.handshake['tf32']
        target=ChemicalTarget(oracle,condition,physical['kT_eV'],physical['restraint_eV_A2']);actual,timing=run(target,models,sources,protocol,a.index)
        assert oracle.evaluated==sum(sum(c['queries_per_parent']) for c in actual['chains'])<=protocol['maximum_new_raw_queries']//len(protocol['condition_indices'])
        if a.phase=='audit':
            equal(actual,expected);assert oracle.index==len(oracle.queries) and oracle.evaluated==producer['new_raw_queries']
            report.update(complete=True,full_replay=True,source_results_sha256=sha(a.run/'results.json'),trace_sha256=producer['trace_sha256'],raw_queries_in_producer=oracle.evaluated,**audit_chains(actual,target,models,protocol))
        else:
            assert oracle.evaluated==oracle.requested_evaluations
            torch.save(actual,a.out/'trace.pt');compact=[]
            for c in actual['chains']:
                readouts={}
                for cap in protocol['readouts']:
                    values=[]
                    for j,parent in enumerate(c['parent_ids']):
                        hit=next((i for i,q in enumerate(c['query_count_history']) if q[j]==cap),None)
                        if hit is None:values.append(dict(parent=parent,reached=False));continue
                        sid=c['history_state_ids'][hit][j];state=actual['states'][sid];old=actual['states'][c['initial_state_ids'][j]]
                        visited={actual['states'][h[j]]['graph']['connectivity_smiles'] for h in c['history_state_ids'][:hit+1]}
                        values.append(dict(parent=parent,reached=True,state_id=sid,potential_change_eV=float(state['potential_eV']-old['potential_eV']),distinct_connectivities=len(visited)))
                    readouts[str(cap)]=values
                compact.append(dict(method=c['method'],replica=c['replica'],queries_per_parent=c['queries_per_parent'],cap_reached=c['cap_reached'],
                    attempted=len(c['attempts']),accepted=sum(r['accepted'] for r in c['attempts']),joint_accepted=sum(r['accepted'] for r in c['attempts'] if r['kind']=='joint'),readouts=readouts))
            report.update(complete=True,chains=compact,all_caps_reached=all(all(c['cap_reached']) for c in actual['chains']),method_seconds=timing,
                new_raw_queries=oracle.evaluated,requested_raw_queries=oracle.requested_evaluations,oracle_runtime=oracle.handshake,oracle_evaluation_seconds=oracle.evaluation_seconds,trace_sha256=sha(a.out/'trace.pt'))
        report['elapsed_seconds']=time.monotonic()-start;write(output,report);print(json.dumps({k:v for k,v in report.items() if k not in ['chains','condition','oracle_runtime']}))
    except Exception as exc:
        if target is not None:torch.save(dict(states=target.states,query_trace=target.query_trace),a.out/'failed_trace.pt')
        report.update(failure=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-start)
        if oracle is not None and a.phase=='evaluate':report.update(new_raw_queries=oracle.evaluated,requested_raw_queries=oracle.requested_evaluations)
        write(output,report);raise
    finally:
        if oracle is not None and a.phase=='evaluate':oracle.close()


if __name__=='__main__':main()
