#!/usr/bin/env python3
"""Regenerate only a numerically affected control suffix under corrected arithmetic."""
import argparse
import copy
import json
from pathlib import Path
import time
import torch
from cfm_mol.chemical_sampler import ChemicalTarget
from cfm_mol.energy_oracle import EnergyOracle
from scripts.research.audit_masked_angular import equal,sha
from scripts.research.evaluate_chemical_policy import write
from scripts.research.fresh_reuse import run_budget
from scripts.research.joint_arc_budget import inputs,diagnostics
from scripts.research.recover_conditional_arc_chain import ReplayThenLiveOracle


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','out','oracle-python','oracle-checkpoint']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args();root=Path(__file__).resolve().parents[2]
    pp=root/'research/evidence/conditional_arc_chain_protocol_v1.json';index=5;method='arc_site';replica=0
    pp,protocol,header,warm,x,ids,physical=inputs(root,args.project,index,pp)
    source=args.project/'runs/conditional_arc_chain_recovery_v1'
    previous=json.loads((source/'results.json').read_text());assert previous['complete']
    old_row=previous['arms'][0];assert old_row['name']=='arc_site_s0' and old_row['raw_queries']==1750
    original=torch.load(source/old_row['file'],map_location='cpu',weights_only=False)
    failed_path=args.project/'runs/conditional_arc_chain_audit_v2/condition_05/arc_site_s0_failed_trace.pt'
    observed=torch.load(failed_path,map_location='cpu',weights_only=False)
    n=len(observed['query_trace']);equal(original['query_trace'][:n],observed['query_trace'])
    for key in ['rounds','transitions','history_state_ids','query_count_history']:
        equal(original[key][:len(observed[key])],observed[key])
    assert observed['query_trace'][-1]['raw_queries_after']==358 and len(observed['rounds'])==46
    assert original['transitions'][46][0]['rejection_reason']=='Orthogonal circle frame required'
    assert args.oracle_checkpoint is not None and sha(args.oracle_checkpoint)==physical['raw_oracle_sha256']
    if args.check_only:
        print('Validated original control,358-call prefix and1392-call suffix bound; no new physical queries');return
    args.out.mkdir(parents=True,exist_ok=True);output=args.out/'results.json'
    if output.exists():raise FileExistsError(output)
    report=copy.deepcopy(previous);report.update(complete=False)
    report['control_repair']=dict(source_results=str(source/'results.json'),source_results_sha256=sha(source/'results.json'),
        original_control_trace_sha256=old_row['trace_sha256'],audit_failed_prefix_sha256=sha(failed_path),
        reused_prefix_raw_calls=358,maximum_additional_physical_calls=1392,additional_physical_calls=0,
        reason='The old control rejected one reverse frame failure; the corrected implementation evaluates it. Rebuild its suffix so every retained arm replays under the corrected arithmetic. Preserve the discarded old suffix and charge its1392 extra historical calls separately.',
        previous_control_sampling_seconds=old_row['sampling_seconds'])
    for row in previous['arms'][1:]:
        path=source/row['file'];assert sha(path)==row['trace_sha256']
        (args.out/row['file']).symlink_to(path.resolve())
    write(output,report);live=oracle=target=None;progress=dict(raw_query_offset=0);start=time.monotonic()
    try:
        live=EnergyOracle(args.oracle_python,root/'scripts/research/oracle_worker.py',args.oracle_checkpoint,
            numbers=header['condition']['numbers'],charge=header['condition']['charge'],spin_multiplicity=header['condition']['spin_multiplicity'],device='cuda',batch_size=32)
        assert live.handshake['base_precision_dtype']=='torch.float32' and not live.handshake['tf32']
        oracle=ReplayThenLiveOracle(live,observed['query_trace'],0)
        target=ChemicalTarget(oracle,header['condition'],physical['kT_eV'],physical['restraint_eV_A2']);tick=time.monotonic()
        run_budget(target,x,ids,None,method,replica,index,protocol,protocol['shared'],progress)
        progress.update(states=target.states,query_trace=target.query_trace)
        assert oracle.evaluated==oracle.requested_evaluations==1750 and oracle.cached_raw_calls==358
        assert live.evaluated==live.requested_evaluations==1392
        path=args.out/'arc_site_s0_trace.pt';torch.save(progress,path)
        report['arms'][0]=dict(old_row,trace_sha256=sha(path),sampling_seconds=time.monotonic()-tick,
            **diagnostics(progress,protocol['readout_raw_queries']))
        report['control_repair'].update(additional_physical_calls=live.evaluated,additional_requested_calls=live.requested_evaluations,
            corrected_control_trace_sha256=sha(path),source_prefix_replayed=True,oracle_runtime=live.handshake,
            elapsed_seconds=time.monotonic()-start)
        report.update(complete=True,elapsed_seconds=previous['elapsed_seconds']+time.monotonic()-start)
        assert sum(r['raw_queries'] for r in report['arms'])==report['new_raw_queries']==5548
        write(output,report);print(json.dumps(dict(complete=True,additional_physical_calls=1392,logical_condition_queries=5548)))
    except Exception as exc:
        if target is not None:
            progress.update(states=target.states,query_trace=target.query_trace);torch.save(progress,args.out/'failed_control_trace.pt')
        report.update(failure=f'{type(exc).__name__}: {exc}')
        if live is not None:report['control_repair'].update(additional_physical_calls=live.evaluated,additional_requested_calls=live.requested_evaluations)
        write(output,report);raise
    finally:
        if live is not None:live.close()


if __name__=='__main__':main()
