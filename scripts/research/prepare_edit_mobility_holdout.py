#!/usr/bin/env python3
"""Prepare previously unused generated parents for evaluation only."""
import argparse,json
from pathlib import Path
import torch
from cfm_mol.chemical_sampler import ChemicalTarget
from cfm_mol.energy_oracle import EnergyOracle
from scripts.research.prepare_proposal_training_states import load_inputs
from scripts.research.fresh_reuse import prepare
from scripts.research.audit_masked_angular import ReplayOracle,equal,sha
from scripts.research.evaluate_chemical_policy import write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','out']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['run','oracle-python','oracle-checkpoint']:p.add_argument('--'+name,type=Path)
    p.add_argument('--index',type=int,required=True);p.add_argument('--phase',choices=['prepare','audit'],required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[2];pp=root/'research/evidence/edit_mobility_holdout_protocol_v1.json';protocol=json.loads(pp.read_text())
    assert protocol['frozen'] and a.index in protocol['condition_indices']
    old_path,old,source,data,physical=load_inputs(root,a.project,a.index)
    assert sha(old_path)==protocol['original_preparation_protocol_sha256']
    spec=protocol['sources'][str(a.index)];ids=spec['parent_ids'];assert len(ids)==3 and not set(ids)&set(old['parent_ids_by_condition'][str(a.index)])
    assert set(ids)<=set(source['supported_parent_ids']) and source['samples_sha256']==spec['samples_sha256']
    source=dict(source,supported_parent_ids=ids);local=dict(protocol,maximum_parents=3,minimum_parents=3,warm_seed=protocol['warm_seed']+100003*a.index)
    a.out.mkdir(parents=True,exist_ok=True);output=a.out/'results.json'
    if output.exists():raise FileExistsError(output)
    report=dict(complete=False,index=a.index,phase=a.phase,protocol_sha256=sha(pp),condition=data['condition'],parent_ids=ids,
        assigned_role='evaluation_only',original_source_stream='fresh_training',original_source_attempts=source['attempted'],original_supported_parents=spec['original_supported_parents'],
        new_raw_queries=0,scientific_submission_ready=False,scope=protocol['interpretation']);write(output,report);oracle=target=None;progress={}
    try:
        if a.phase=='audit':
            producer=json.loads((a.run/'results.json').read_text());assert producer['complete'] and producer['protocol_sha256']==sha(pp) and sha(a.run/'trace.pt')==producer['trace_sha256']
            expected=torch.load(a.run/'trace.pt',map_location='cpu',weights_only=False);oracle=ReplayOracle(expected['query_trace'])
        else:
            assert sha(a.oracle_checkpoint)==physical['raw_oracle_sha256']
            oracle=EnergyOracle(a.oracle_python,root/'scripts/research/oracle_worker.py',a.oracle_checkpoint,numbers=data['condition']['numbers'],charge=data['condition']['charge'],
                spin_multiplicity=data['condition']['spin_multiplicity'],device='cuda',batch_size=32)
            assert oracle.handshake['base_precision_dtype']=='torch.float32' and not oracle.handshake['tf32']
        target=ChemicalTarget(oracle,data['condition'],physical['kT_eV'],physical['restraint_eV_A2'])
        prepare(target,data,source,local,progress);progress.update(condition=data['condition'],states=target.states,query_trace=target.query_trace)
        assert oracle.evaluated==sum(progress['final_queries_per_parent'])<=protocol['maximum_queries_per_condition']
        if a.phase=='audit':
            equal(progress,expected);assert oracle.index==len(oracle.queries) and oracle.evaluated==producer['new_raw_queries']
            report.update(complete=True,full_replay=True,source_results_sha256=sha(a.run/'results.json'),trace_sha256=producer['trace_sha256'],raw_queries_in_producer=oracle.evaluated)
        else:
            assert oracle.evaluated==oracle.requested_evaluations
            torch.save(progress,a.out/'trace.pt');report.update(complete=True,trace_sha256=sha(a.out/'trace.pt'),new_raw_queries=oracle.evaluated,requested_raw_queries=oracle.requested_evaluations,
                final_state_ids=progress['history_state_ids'][-1],raw_queries_per_parent=progress['final_queries_per_parent'],oracle_runtime=oracle.handshake)
        write(output,report);print(json.dumps({k:v for k,v in report.items() if k not in ['condition','oracle_runtime']}))
    except Exception as exc:
        if target is not None:
            progress.update(states=target.states,query_trace=target.query_trace);torch.save(progress,a.out/'failed_trace.pt')
        report.update(failure=f'{type(exc).__name__}: {exc}')
        if oracle is not None and a.phase=='prepare':report.update(new_raw_queries=oracle.evaluated,requested_raw_queries=oracle.requested_evaluations)
        write(output,report);raise
    finally:
        if oracle is not None and a.phase=='prepare':oracle.close()


if __name__=='__main__':main()
