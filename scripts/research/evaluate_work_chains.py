#!/usr/bin/env python3
"""Bounded full-chain comparison of physical, single and cooperative work moves."""
import argparse,json,math,time
from pathlib import Path
import torch
from cfm_mol.work_chain_kernel import WorkChainKernel,independently_normalized
from cfm_mol.chemical_sampler import ChemicalTarget
from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.edit_bridge_sampler import RootZeroBridgeField
from cfm_mol.conditional_molecular_proposal import invariant_jump_squared
from scripts.research.evaluate_edit_mobility_chains import run_chain,audit_chains as audit_legacy
from scripts.research.evaluate_cooperative_edits import load_models
from scripts.research.evaluate_edit_bridge import inputs
from scripts.research.audit_masked_angular import ReplayOracle,equal,sha
from scripts.research.evaluate_chemical_policy import write


def make_kernels(project,protocol):
    models=load_models(project,protocol);result={}
    for replica in protocol['replicas']:
        base=models[f'linear_s{replica}'][0]
        for method in protocol['methods']:
            common=dict(backbone=base,panel_size=protocol['panel_size'],policy_options=protocol['policy'],root_noise_options=protocol['bridge_by_method']['root_noise'])
            if method=='root_noise':kernel=WorkChainKernel('root_noise',**common)
            elif method=='single_linear':kernel=WorkChainKernel('single',**common)
            elif method=='panel_linear':kernel=WorkChainKernel('panel',**common)
            elif method=='panel_restraint':kernel=WorkChainKernel('panel',use_affinity=True,**common)
            else:
                variant={'panel_radial':'linear','panel_blind':'blind'}[method]
                kernel=WorkChainKernel('panel',interaction=models[f'{variant}_interaction_s{replica}'][1],use_affinity=True,**common)
            result[f'{method}_s{replica}']=kernel
    return result


def run(target,kernels,sources,protocol,index,progress=None):
    chains=[];timing={};loop_timing={}
    for arm in protocol['arm_order'][str(index)]:
        method,replica=arm['method'],arm['replica'];name=f'{method}_s{replica}';start=time.monotonic()
        chain,seconds=run_chain(target,None,sources,method,name,replica,index,protocol,joint_kernel=kernels[name])
        timing[name]=time.monotonic()-start;loop_timing[name]=seconds;chains.append(chain)
        print(json.dumps(dict(arm=name,condition=index,queries=chain['queries_per_parent'],all_caps_reached=all(chain['cap_reached']),seconds=timing[name])),flush=True)
        if progress is not None:progress(dict(chains=chains,states=target.states,query_trace=target.query_trace),timing,loop_timing)
    return dict(chains=chains,states=target.states,query_trace=target.query_trace),timing,loop_timing


def audit(data,target,protocol):
    legacy=[];normalized=normalizers=history_checks=0
    for chain in data['chains']:
        filtered=[];indices=list(chain['initial_state_ids']);counts=[2]*len(indices)
        assert chain['history_state_ids'][0]==indices and chain['query_count_history'][0]==counts
        by_step={}
        for row in chain['attempts']:
            by_step.setdefault(row['step'],[]).append(row)
            custom=row.get('kernel_type') in ('single','panel')
            # Legacy audit still counts every query and audits raw potentials,
            # local/rotation kernels and the physical root-noise map.
            filtered.append(dict(row,kind='custom',scored=False) if custom else row)
            if not custom or not row['scored']:continue
            for side in ['forward','reverse']:
                expected=independently_normalized(row[side],target.kT,protocol['policy']['uniform_fraction'])
                torch.testing.assert_close(torch.tensor(expected,dtype=torch.float64),row[side]['log_probability'],atol=1e-10,rtol=0)
                normalizers+=1
            old=data['states'][row['old_state_id']];new=data['states'][row['new_state_id']]
            forward=float(row['forward']['log_probability'][row['forward_selected']]);backward=float(row['reverse']['log_probability'][row['reverse_selected']])
            ratio=-float(new['potential_eV']-old['potential_eV'])/target.kT+row['log_volume']+backward-forward
            assert abs(ratio-row['log_acceptance_ratio'])<1e-7
            assert row['accepted']==(row['log_uniform']<min(0.,ratio));normalized+=1
        for step in range(len(chain['history_state_ids'])-1):
            attempts=by_step.get(step,[])
            assert len({r['parent_offset'] for r in attempts})==len(attempts)
            for row in attempts:
                j=row['parent_offset'];assert row['old_state_id']==indices[j]
                if row['accepted']:indices[j]=row['new_state_id']
                counts[j]+=row['raw_cost'];assert counts[j]<=protocol['query_cap_per_parent']
            assert chain['history_state_ids'][step+1]==indices and chain['query_count_history'][step+1]==counts
            history_checks+=1
        assert counts==chain['queries_per_parent'] and indices==chain['final_state_ids']
        legacy.append(dict(chain,attempts=filtered))
    checked=audit_legacy(dict(data,chains=legacy),target,{'root_noise':RootZeroBridgeField()},protocol)
    checked.update(independent_normalized_joint_MH_checks=normalized,independent_catalogue_normalizations=normalizers,
        independent_history_steps=history_checks,independent_joint_MH_checks=checked['independent_joint_MH_checks']+normalized)
    return checked


def readouts(data,protocol,numbers):
    result=[];numbers=torch.tensor(numbers,dtype=torch.long)
    for chain in data['chains']:
        reads={}
        for cap in protocol['readouts']:
            rows=[]
            for j,parent in enumerate(chain['parent_ids']):
                hit=next((step for step,counts in enumerate(chain['query_count_history']) if counts[j]==cap),None)
                if hit is None:rows.append(dict(parent=parent,reached=False));continue
                trajectory=[data['states'][ids[j]] for ids in chain['history_state_ids'][:hit+1]]
                old,new=trajectory[0],trajectory[-1];seen={old['graph']['connectivity_smiles']};last=old['graph']['connectivity_smiles'];changes=returns=0
                for state in trajectory[1:]:
                    identity=state['graph']['connectivity_smiles']
                    if identity!=last:
                        changes+=1;returns+=identity in seen;seen.add(identity);last=identity
                x=torch.stack([s['positions'] for s in trajectory])
                excursions=invariant_jump_squared(old['positions'][None].expand_as(x),x,numbers)
                events=[r for r in chain['attempts'] if r['parent_offset']==j and r['step']<hit]
                rows.append(dict(parent=parent,reached=True,state_id=new['state_id'],potential_change_eV=float(new['potential_eV']-old['potential_eV']),
                    distinct_connectivities=len(seen),connectivity_transitions=changes,connectivity_returns=returns,
                    final_typed_distance_change_A2=float(excursions[-1]),max_typed_distance_change_A2=float(excursions.max()),
                    accepted_uphill_moves=sum(r['accepted'] and float(data['states'][r['new_state_id']]['potential_eV']-data['states'][r['old_state_id']]['potential_eV'])>protocol['kT_eV'] for r in events),
                    accepted_joint_moves=sum(r['accepted'] for r in events if r['kind']=='joint'),
                    type_fallback_attempts=sum(r.get('type_fallback',False) for r in events)))
            reads[str(cap)]=rows
        result.append(dict(method=chain['method'],replica=chain['replica'],queries_per_parent=chain['queries_per_parent'],cap_reached=chain['cap_reached'],
            attempted=len(chain['attempts']),accepted=sum(r['accepted'] for r in chain['attempts']),
            joint_accepted=sum(r['accepted'] for r in chain['attempts'] if r['kind']=='joint'),readouts=reads))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','out','protocol']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['run','oracle-python','oracle-checkpoint']:p.add_argument('--'+name,type=Path)
    p.add_argument('--index',type=int,required=True);p.add_argument('--phase',choices=['evaluate','audit'],required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[2];pp=root/a.protocol
    _,protocol,physical,_,condition,sources=inputs(root,a.project,a.index,pp);kernels=make_kernels(a.project,protocol)
    a.out.mkdir(parents=True,exist_ok=True);output=a.out/'results.json'
    if output.exists():raise FileExistsError(output)
    report=dict(complete=False,index=a.index,phase=a.phase,protocol_sha256=sha(pp),new_raw_queries=0,scientific_submission_ready=False,scope=protocol['interpretation'])
    write(output,report);oracle=target=None;start=time.monotonic()
    try:
        if a.phase=='audit':
            producer=json.loads((a.run/'results.json').read_text());assert producer['complete'] and producer['protocol_sha256']==sha(pp) and sha(a.run/'trace.pt')==producer['trace_sha256']
            expected=torch.load(a.run/'trace.pt',map_location='cpu',weights_only=False);oracle=ReplayOracle(expected['query_trace'])
        else:
            assert sha(a.oracle_checkpoint)==physical['raw_oracle_sha256']
            oracle=EnergyOracle(a.oracle_python,root/'scripts/research/oracle_worker.py',a.oracle_checkpoint,numbers=condition['numbers'],charge=condition['charge'],spin_multiplicity=condition['spin_multiplicity'],device='cuda',batch_size=32)
            assert oracle.handshake['base_precision_dtype']=='torch.float32' and not oracle.handshake['tf32']
        target=ChemicalTarget(oracle,condition,physical['kT_eV'],physical['restraint_eV_A2'])
        def progress(data,timing,loop_timing):
            torch.save(data,a.out/'partial_trace.tmp');(a.out/'partial_trace.tmp').replace(a.out/'partial_trace.pt')
            write(output,dict(report,new_raw_queries=oracle.evaluated,completed_arms=len(data['chains']),method_seconds=timing,loop_seconds=loop_timing))
        actual,timing,loop_timing=run(target,kernels,sources,protocol,a.index,progress if a.phase=='evaluate' else None)
        assert oracle.evaluated==sum(sum(c['queries_per_parent']) for c in actual['chains'])<=protocol['maximum_new_raw_queries']//len(protocol['condition_indices'])
        checks=audit(actual,target,protocol);compact=readouts(actual,protocol,target.numbers)
        if a.phase=='audit':
            equal(actual,expected);assert oracle.index==len(oracle.queries) and oracle.evaluated==producer['new_raw_queries'];equal(compact,producer['chains'])
            report.update(complete=True,full_replay=True,source_results_sha256=sha(a.run/'results.json'),trace_sha256=producer['trace_sha256'],raw_queries_in_producer=oracle.evaluated,**checks)
        else:
            assert oracle.evaluated==oracle.requested_evaluations
            torch.save(actual,a.out/'trace.pt');report.update(complete=True,trace_sha256=sha(a.out/'trace.pt'),chains=compact,new_raw_queries=oracle.evaluated,
                requested_raw_queries=oracle.requested_evaluations,method_seconds=timing,loop_seconds=loop_timing,
                all_caps_reached=all(all(c['cap_reached']) for c in actual['chains']),oracle_evaluation_seconds=oracle.evaluation_seconds,**checks)
        report['elapsed_seconds']=time.monotonic()-start;write(output,report)
    except Exception as exc:
        if target is not None:torch.save(dict(states=target.states,query_trace=target.query_trace),a.out/'failed_trace.pt')
        report['failure']=f'{type(exc).__name__}: {exc}'
        if oracle is not None and a.phase=='evaluate':report['new_raw_queries']=oracle.evaluated
        write(output,report);raise
    finally:
        if oracle is not None and a.phase=='evaluate':oracle.close()


if __name__=='__main__':main()
