#!/usr/bin/env python3
"""Replay the complete 32-cap prefix, then continue every budget-exhausted arm."""
import argparse,json,time
from pathlib import Path
import torch
from cfm_mol.chemical_sampler import ChemicalTarget
from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.mobility_relaxation import relax_arms
from scripts.research.mobility_relaxation_pilot import inputs,run,independent_audit
from scripts.research.audit_masked_angular import ReplayOracle,equal,sha
from scripts.research.evaluate_chemical_policy import write


class OffsetOracle:
    def __init__(self,live,offset):self.live=live;self.offset=offset
    @property
    def evaluated(self):return self.offset+self.live.evaluated
    def evaluate_chunked(self,x,max_request):return self.live.evaluate_chunked(x,max_request=max_request)


def resume(target,original,options):
    prefix_count=len(target.query_trace);state_count=len(target.states)
    before=target.oracle.evaluated
    updated=relax_arms(target,original['pairs'],options,resume_arms=original['arms'])
    for old,new in zip(original['arms'],updated):
        equal(new['events'][:len(old['events'])],old['events'])
        equal(new['checkpoints'][:len(old['checkpoints'])],old['checkpoints'])
        if old['status']!='budget_exhausted':equal(new,old)
        assert new['evaluations']>=old['evaluations']
    assert target.oracle.evaluated-before==2*sum(n['evaluations']-o['evaluations'] for o,n in zip(original['arms'],updated))
    result=dict(original,arms=updated,states=target.states,query_trace=target.query_trace)
    return result,dict(prefix_query_batches=prefix_count,prefix_states=state_count,new_raw_queries=target.oracle.evaluated-before)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','out']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['run','oracle-python','oracle-checkpoint']:p.add_argument('--'+name,type=Path)
    p.add_argument('--index',type=int,required=True);p.add_argument('--phase',choices=['evaluate','audit'],required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[2]
    base_path,base,physical,condition,rows=inputs(root,a.project,a.index)
    pp=root/'research/evidence/mobility_relaxation_continuation_protocol_v1.json';protocol=json.loads(pp.read_text())
    assert protocol['frozen'] and sha(base_path)==protocol['base_protocol_sha256']
    for key,value in base['options'].items():
        if key not in ['max_evaluations','readouts']:assert protocol['options'][key]==value
    source=a.project/protocol['source_run']/f'condition_{a.index:02d}'
    source_header=json.loads((source/'results.json').read_text());frozen=protocol['sources'][str(a.index)]
    assert source_header['complete'] and sha(source/'results.json')==frozen['results_sha256']
    assert sha(source/'trace.pt')==source_header['trace_sha256']==frozen['trace_sha256']
    audit_path=a.project/protocol['source_audit']/f'condition_{a.index:02d}/results.json'
    old_audit=json.loads(audit_path.read_text());assert old_audit['complete'] and old_audit['full_replay'] and sha(audit_path)==frozen['audit_sha256']
    saved=torch.load(source/'trace.pt',map_location='cpu',weights_only=False)
    a.out.mkdir(parents=True,exist_ok=True);output=a.out/'results.json'
    if output.exists():raise FileExistsError(output)
    report=dict(complete=False,index=a.index,phase=a.phase,protocol_sha256=sha(pp),base_protocol_sha256=sha(base_path),condition=condition,
        selection=base['selections'][str(a.index)],inherited_raw_queries=source_header['new_raw_queries'],new_raw_queries=0,
        scientific_submission_ready=False,scope=protocol['interpretation'])
    write(output,report);live=target=None;start=time.monotonic()
    try:
        if a.phase=='audit':
            producer=json.loads((a.run/'results.json').read_text());assert producer['complete'] and producer['protocol_sha256']==sha(pp)
            assert sha(a.run/'trace.pt')==producer['trace_sha256']
            expected=torch.load(a.run/'trace.pt',map_location='cpu',weights_only=False)
            oracle=ReplayOracle(expected['query_trace'])
        else:oracle=ReplayOracle(saved['query_trace'])
        target=ChemicalTarget(oracle,condition,physical['kT_eV'],physical['restraint_eV_A2'])
        original=run(target,rows,base);equal(original,saved)
        assert oracle.index==len(saved['query_trace']) and oracle.evaluated==source_header['new_raw_queries']
        if a.phase=='evaluate':
            assert sha(a.oracle_checkpoint)==physical['raw_oracle_sha256']
            live=EnergyOracle(a.oracle_python,root/'scripts/research/oracle_worker.py',a.oracle_checkpoint,
                numbers=condition['numbers'],charge=condition['charge'],spin_multiplicity=condition['spin_multiplicity'],device='cuda',batch_size=base['oracle_batch_size'])
            assert live.handshake['base_precision_dtype']=='torch.float32' and not live.handshake['tf32']
            target.oracle=OffsetOracle(live,oracle.evaluated)
        actual,counts=resume(target,original,protocol['options'])
        equal(actual['query_trace'][:len(saved['query_trace'])],saved['query_trace'])
        equal(actual['states'][:len(saved['states'])],saved['states'])
        cost=target.oracle.evaluated
        assert counts['new_raw_queries']<=frozen['maximum_additional_raw_queries']
        if a.phase=='audit':
            equal(actual,expected);assert oracle.index==len(oracle.queries) and cost==producer['cumulative_raw_queries']
            assert counts['new_raw_queries']==producer['new_raw_queries']
            report.update(complete=True,source_results_sha256=sha(a.run/'results.json'),trace_sha256=producer['trace_sha256'],full_replay=True,
                raw_queries_in_producer=cost,new_queries_in_producer=counts['new_raw_queries'],original_prefix_replayed_and_preserved=True,
                **independent_audit(actual,target.restraint,protocol['options']))
        else:
            assert live.evaluated==live.requested_evaluations==counts['new_raw_queries']
            assert cost==source_header['new_raw_queries']+live.evaluated
            torch.save(actual,a.out/'trace.pt')
            compact=[{k:arm[k] for k in ['pair_id','parent','endpoint','mobility','evaluations','geometry_probes','accepted_steps','status',
                'best_projected_force_max_eV_A','final_projected_force_max_eV_A','best_potential_eV','final_potential_eV','initial_potential_eV','best_state_id','current_state_id']} for arm in actual['arms']]
            report.update(complete=True,arms=compact,pairs=actual['pairs'],initial_comparisons=actual['initial_comparisons'],
                initial_raw_queries=actual['initial_queries'],repeat_energy_error_eV=actual['repeat_energy_error_eV'],repeat_force_error_eV_A=actual['repeat_force_error_eV_A'],
                new_raw_queries=live.evaluated,requested_raw_queries=live.requested_evaluations,cumulative_raw_queries=cost,
                requested_cumulative_raw_queries=source_header['new_raw_queries']+live.requested_evaluations,
                trace_sha256=sha(a.out/'trace.pt'),oracle_runtime=live.handshake,oracle_evaluation_seconds=live.evaluation_seconds,
                original_prefix_replayed_and_preserved=True,continuation_counts=counts)
        report['elapsed_seconds']=time.monotonic()-start;write(output,report)
        print(json.dumps({k:v for k,v in report.items() if k not in ['arms','pairs','selection','condition','oracle_runtime','initial_comparisons']}))
    except Exception as exc:
        if target is not None:torch.save(dict(states=target.states,query_trace=target.query_trace),a.out/'failed_trace.pt')
        report.update(failure=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-start)
        if live is not None:report.update(new_raw_queries=live.evaluated,requested_raw_queries=live.requested_evaluations)
        write(output,report);raise
    finally:
        if live is not None:live.close()


if __name__=='__main__':main()
