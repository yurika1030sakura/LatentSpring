#!/usr/bin/env python3
"""Bounded physical preparation of audited, disjoint proposal-training sources."""
import argparse
import json
from pathlib import Path

import torch

from cfm_mol.chemical_sampler import ChemicalTarget
from cfm_mol.energy_oracle import EnergyOracle
from scripts.research.audit_masked_angular import ReplayOracle, equal, sha
from scripts.research.evaluate_chemical_policy import write
from scripts.research.fresh_reuse import prepare


def load_inputs(root, project, index):
    pp=root/'research/evidence/proposal_training_preparation_protocol_v1.json'
    protocol=json.loads(pp.read_text())
    ap=project/protocol['source_audit'];audit=json.loads(ap.read_text())
    assert sha(ap)==protocol['source_audit_sha256'] and audit['complete']
    assert audit['all_conditions_retained'] and audit['all_chunk_rows_verified'] and audit['stream']=='fresh_training'
    assert audit['manifest_sha256']==protocol['manifest_sha256']
    assert audit['protocol_sha256']==protocol['source_protocol_sha256']
    row=audit['rows'][index];assert row['index']==index
    directory=project/protocol['source_run']/f'condition_{index:02d}'
    assert sha(directory/'results.json')==row['source_results_sha256']
    assert sha(directory/'samples.pt')==row['samples_sha256']
    source=json.loads((directory/'results.json').read_text())
    assert source['complete'] and source['physical_queries']==0 and not source['energy_labels_computed']
    data=torch.load(directory/'samples.pt',map_location='cpu',weights_only=False)
    assert data['stream']==source['stream']=='fresh_training'
    assert data['condition']==source['condition']==row['condition']
    assert not set(data)&{'energy_eV','work','log_q','importance_weights'}
    ids=row['supported_parent_ids'][:protocol['maximum_parents']]
    assert ids==protocol['parent_ids_by_condition'][str(index)]
    physical_path=root/'research/evidence/parity_training_protocol_v1.json'
    assert sha(physical_path)==protocol['physical_protocol_sha256']
    physical=json.loads(physical_path.read_text())
    local=dict(protocol,warm_seed=protocol['warm_seed']+100003*index)
    return pp,local,row,data,physical


def produce(args):
    root=Path(__file__).resolve().parents[2]
    pp,protocol,row,data,physical=load_inputs(root,args.project,args.index)
    ids=protocol['parent_ids_by_condition'][str(args.index)]
    args.out.mkdir(parents=True,exist_ok=True);output=args.out/'results.json'
    if output.exists():raise FileExistsError(output)
    result=dict(complete=False,index=args.index,protocol_sha256=sha(pp),condition=data['condition'],
        source_attempts=row['attempted'],source_supported=row['chemically_supported'],parent_ids=ids,
        source_results_sha256=row['source_results_sha256'],source_samples_sha256=row['samples_sha256'],
        source_audit_sha256=protocol['source_audit_sha256'],stream='fresh_training',new_raw_queries=0,
        scientific_submission_ready=False,model_fitted=False,reference_coordinates_loaded=False)
    write(output,result)
    if not ids:
        result.update(complete=True,zero_support=True,requested_raw_queries=0,
            scope='Frozen zero-support training composition retained; no replacement or oracle launch.')
        write(output,result);print(json.dumps(result));return
    assert args.oracle_python is not None and args.oracle_checkpoint is not None
    assert sha(args.oracle_checkpoint)==physical['raw_oracle_sha256']
    target=oracle=None;progress={}
    try:
        oracle=EnergyOracle(args.oracle_python,root/'scripts/research/oracle_worker.py',args.oracle_checkpoint,
            numbers=data['condition']['numbers'],charge=data['condition']['charge'],
            spin_multiplicity=data['condition']['spin_multiplicity'],device='cuda',batch_size=32)
        assert oracle.handshake['base_precision_dtype']=='torch.float32' and not oracle.handshake['tf32']
        target=ChemicalTarget(oracle,data['condition'],physical['kT_eV'],physical['restraint_eV_A2'])
        prepare(target,data,row,protocol,progress)
        progress.update(condition=data['condition'],states=target.states,query_trace=target.query_trace)
        torch.save(progress,args.out/'trace.pt')
        assert oracle.evaluated==oracle.requested_evaluations<=2*len(ids)*(protocol['warm_steps']+1)
        result.update(complete=True,zero_support=False,trace_sha256=sha(args.out/'trace.pt'),
            new_raw_queries=oracle.evaluated,requested_raw_queries=oracle.requested_evaluations,
            raw_queries_per_parent=progress['final_queries_per_parent'],oracle_runtime=oracle.handshake,
            scope='First supported TRAINING parents after fixed local preparation. No equilibrium qualification, fitting or learned-method benefit.')
        write(output,result);print(json.dumps({k:v for k,v in result.items() if k not in ['condition','oracle_runtime']}))
    except Exception as exc:
        if target is not None:
            progress.update(states=target.states,query_trace=target.query_trace)
            torch.save(progress,args.out/'failed_trace.pt')
        result.update(failure=f'{type(exc).__name__}: {exc}',new_raw_queries=oracle.evaluated if oracle else 0,
            requested_raw_queries=oracle.requested_evaluations if oracle else 0)
        write(output,result);raise
    finally:
        if oracle is not None:oracle.close()


def audit_all(args):
    root=Path(__file__).resolve().parents[2];rows=[];total=0
    for index in range(8):
        pp,protocol,source,data,physical=load_inputs(root,args.project,index)
        directory=args.run/f'condition_{index:02d}'
        report=json.loads((directory/'results.json').read_text())
        assert report['complete'] and report['protocol_sha256']==sha(pp)
        assert report['condition']==data['condition'] and report['stream']=='fresh_training'
        assert report['parent_ids']==protocol['parent_ids_by_condition'][str(index)]
        assert report['source_results_sha256']==source['source_results_sha256']
        assert report['source_samples_sha256']==source['samples_sha256']
        assert report['source_audit_sha256']==protocol['source_audit_sha256']
        assert report['source_attempts']==source['attempted'] and report['source_supported']==source['chemically_supported']
        assert not report['model_fitted'] and not report['reference_coordinates_loaded']
        assert report['new_raw_queries']==report['requested_raw_queries']
        ids=report['parent_ids']
        if not ids:
            assert report['zero_support'] and source['chemically_supported']==0 and report['new_raw_queries']==0
            assert not (directory/'trace.pt').exists()
        else:
            assert not report['zero_support'] and report['trace_sha256']==sha(directory/'trace.pt')
            saved=torch.load(directory/'trace.pt',map_location='cpu',weights_only=False)
            oracle=ReplayOracle(saved['query_trace'])
            target=ChemicalTarget(oracle,data['condition'],physical['kT_eV'],physical['restraint_eV_A2'])
            actual={};prepare(target,data,source,protocol,actual)
            actual.update(condition=data['condition'],states=target.states,query_trace=target.query_trace)
            equal(actual,saved)
            assert oracle.index==len(oracle.queries) and oracle.evaluated==report['new_raw_queries']
            independent=[2+2*sum(int(r[i]['valid']) for r in saved['transitions']) for i in range(len(ids))]
            assert independent==report['raw_queries_per_parent']==saved['final_queries_per_parent']
        total+=report['new_raw_queries']
        rows.append(dict(index=index,parent_ids=ids,zero_support=not ids,results_sha256=sha(directory/'results.json'),
            trace_sha256=report.get('trace_sha256'),raw_queries=report['new_raw_queries']))
    assert total<=protocol['maximum_total_new_raw_queries']
    result=dict(complete=True,protocol_sha256=sha(pp),rows=rows,raw_queries_in_preparation=total,
        new_physical_queries=0,all_eight_conditions_retained=True,full_producer_replay=True,
        all_random_streams_replayed=True,independent_parent_query_accounting=True,scientific_submission_ready=False)
    if args.out.exists():raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['prepare','audit'],required=True)
    for name in ['project','out']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['oracle-python','oracle-checkpoint','run']:p.add_argument('--'+name,type=Path)
    p.add_argument('--index',type=int,choices=range(8))
    args=p.parse_args()
    if args.phase=='prepare':
        if args.index is None:p.error('--index required for preparation')
        produce(args)
    else:
        if args.run is None:p.error('--run required for audit')
        audit_all(args)


if __name__=='__main__':main()
