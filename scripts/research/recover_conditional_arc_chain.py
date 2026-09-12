#!/usr/bin/env python3
"""Recover the verified chart failure using immutable cached oracle prefix data."""
import argparse
import copy
import json
from pathlib import Path
import time
import torch

from cfm_mol.chemical_sampler import ChemicalTarget
from cfm_mol.energy_oracle import EnergyOracle
from scripts.research.audit_masked_angular import ReplayOracle,sha
from scripts.research.evaluate_chemical_policy import write
from scripts.research.fresh_reuse import run_budget,parent_query_caps
from scripts.research.joint_arc_budget import inputs,load_scalar_model,diagnostics


class ReplayThenLiveOracle:
    """Reuse exactly matching completed requests before making any new request."""
    def __init__(self,live,queries,offset):
        self.live=live;self.replay=ReplayOracle(queries);self.replay.evaluated=offset
        self.evaluated=offset;self.requested_evaluations=offset
        self.cached_raw_calls=0;self.handshake=live.handshake

    def evaluate_chunked(self,positions,*,max_request=32):
        if self.replay.index<len(self.replay.queries):
            before=self.replay.evaluated
            result=self.replay.evaluate_chunked(positions,max_request)
            cost=self.replay.evaluated-before;self.cached_raw_calls+=cost
            self.evaluated+=cost;self.requested_evaluations+=cost
            return result
        before=self.live.evaluated;requested=self.live.requested_evaluations
        try:return self.live.evaluate_chunked(positions,max_request=max_request)
        finally:
            self.evaluated+=self.live.evaluated-before
            self.requested_evaluations+=self.live.requested_evaluations-requested


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','source','out','oracle-python','oracle-checkpoint']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();root=Path(__file__).resolve().parents[2]
    pp=root/'research/evidence/conditional_arc_chain_protocol_v1.json';index=5
    pp,protocol,header,warm,positions,ids,physical=inputs(root,args.project,index,pp)
    original=json.loads((args.source/'results.json').read_text())
    assert not original['complete'] and original['failure']=='ValueError: Orthogonal circle frame required'
    assert original['index']==index and original['protocol_sha256']==sha(pp) and original['parent_ids']==ids
    assert original['new_raw_queries']==original['requested_raw_queries']
    order=[(replica,method,f'{method}_s{replica}') for replica in protocol['replicas'] for method in protocol['methods']]
    assert len(original['arms'])==1 and original['arms'][0]['name']==order[0][2]
    partial_path=args.source/(order[1][2]+'_failed_trace.pt')
    partial=torch.load(partial_path,map_location='cpu',weights_only=False)
    offset=sum(r['raw_queries'] for r in original['arms'])
    assert partial['raw_query_offset']==offset==1750
    assert partial['query_trace'][-1]['raw_queries_after']==original['new_raw_queries']==2102
    assert sum(partial['query_count_history'][-1])==original['new_raw_queries']-offset
    args.out.mkdir(parents=True,exist_ok=True);output=args.out/'results.json'
    if output.exists():raise FileExistsError(output)
    report=copy.deepcopy(original);report.pop('failure')
    report.update(complete=False,new_raw_queries=offset,requested_raw_queries=offset,
        recovery=dict(source=str(args.source),source_results_sha256=sha(args.source/'results.json'),
            source_failed_trace_sha256=sha(partial_path),source_failure=original['failure'],
            previous_physical_calls=original['new_raw_queries'],reused_complete_arm_calls=offset,
            expected_cached_partial_calls=original['new_raw_queries']-offset,
            previous_attempt_elapsed_seconds=original['elapsed_seconds'],previous_oracle_runtime=original['oracle_runtime'],
            additional_physical_calls=0,
            accounting='The legacy new_raw_queries field is the combined logical-experiment total, including immutable previous calls reused once. additional_physical_calls is the genuinely new recovery cost. Previous failed-attempt time is retained separately; per-method timing is incomplete for the recovered partial arm.'))
    for row in report['arms']:
        path=args.source/row['file'];assert sha(path)==row['trace_sha256']
        (args.out/row['file']).symlink_to(path.resolve())
    write(output,report);live=oracle=target=None;start=time.monotonic();progress={};name='unstarted'
    try:
        assert sha(args.oracle_checkpoint)==physical['raw_oracle_sha256']
        live=EnergyOracle(args.oracle_python,root/'scripts/research/oracle_worker.py',args.oracle_checkpoint,
            numbers=header['condition']['numbers'],charge=header['condition']['charge'],spin_multiplicity=header['condition']['spin_multiplicity'],device='cuda',batch_size=32)
        assert live.handshake['base_precision_dtype']=='torch.float32' and not live.handshake['tf32']
        oracle=ReplayThenLiveOracle(live,partial['query_trace'],offset)
        report['oracle_runtime']=live.handshake
        report['recovery']['oracle_startup_seconds']=time.monotonic()-start
        for replica,method,name in order[len(original['arms']):]:
            tick=time.monotonic();model,digest=load_scalar_model(args.project,protocol,method,replica)
            load_seconds=time.monotonic()-tick;offset=oracle.evaluated
            target=ChemicalTarget(oracle,header['condition'],physical['kT_eV'],physical['restraint_eV_A2'])
            progress=dict(raw_query_offset=offset);tick=time.monotonic()
            run_budget(target,positions,ids,model,method,replica,index,protocol,protocol['shared'],progress)
            progress.update(states=target.states,query_trace=target.query_trace)
            assert oracle.replay.index==len(oracle.replay.queries)
            cost=oracle.evaluated-offset;assert cost==sum(progress['final_queries_per_parent'])
            path=args.out/f'{name}_trace.pt';torch.save(progress,path)
            row=dict(name=name,method=method,replica=replica,file=path.name,trace_sha256=sha(path),raw_query_offset=offset,
                raw_queries=cost,sampling_seconds=time.monotonic()-tick,model_sha256=digest,model_loading_seconds=load_seconds,
                **diagnostics(progress,protocol['readout_raw_queries']))
            report['arms'].append(row);report.update(new_raw_queries=oracle.evaluated,requested_raw_queries=oracle.requested_evaluations)
            report['recovery'].update(additional_physical_calls=live.evaluated,additional_requested_calls=live.requested_evaluations,
                cached_partial_calls=oracle.cached_raw_calls,all_cached_requests_replayed=oracle.replay.index==len(oracle.replay.queries))
            write(output,report);print(json.dumps(dict(arm=name,raw_queries=cost,cached_partial_calls=oracle.cached_raw_calls,additional_physical_calls=live.evaluated)),flush=True)
        total=sum(r['raw_queries'] for r in report['arms'])
        maximum=len(protocol['replicas'])*sum(sum(parent_query_caps(protocol,m,index,ids)) for m in protocol['methods'])
        assert total==oracle.evaluated==oracle.requested_evaluations<=maximum
        assert oracle.cached_raw_calls==352 and live.evaluated==live.requested_evaluations
        assert total==original['new_raw_queries']+live.evaluated
        report.update(complete=True,elapsed_seconds=time.monotonic()-start)
        write(output,report)
    except Exception as exc:
        if target is not None:
            progress.update(states=target.states,query_trace=target.query_trace)
            torch.save(progress,args.out/f'{name}_failed_trace.pt')
        report.update(failure=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-start)
        if oracle is not None:report.update(new_raw_queries=oracle.evaluated,requested_raw_queries=oracle.requested_evaluations)
        if live is not None:report['recovery'].update(additional_physical_calls=live.evaluated,additional_requested_calls=live.requested_evaluations)
        write(output,report);raise
    finally:
        if live is not None:live.close()


if __name__=='__main__':main()
