#!/usr/bin/env python3
"""Score frozen first arc draws on FIT parents; no new neural model is trained."""
import argparse
import json
import math
from pathlib import Path
import torch

from cfm_mol.chemical_sampler import ChemicalTarget
from cfm_mol.energy_oracle import EnergyOracle
from scripts.research.audit_masked_angular import ReplayOracle,equal,sha
from scripts.research.evaluate_chemical_policy import write


def inputs(root,project,index):
    pp=root/'research/evidence/arc_oracle_feasibility_protocol_v1.json';protocol=json.loads(pp.read_text())
    ap=project/protocol['arc_audit'];audit=json.loads(ap.read_text())
    assert sha(ap)==protocol['arc_audit_sha256']
    assert audit['complete'] and audit['all_arc_and_fullsphere_draws_replayed'] and audit['all_graph_checks_repeated']
    assert audit['independent_first_draw_forward_reverse_density_checks']==3504
    assert audit['arc_results_sha256']==protocol['arc_results_sha256'] and audit['arc_trace_sha256']==protocol['arc_trace_sha256']
    directory=project/'runs/arc_teacher_support_v1'
    assert sha(directory/'results.json')==protocol['arc_results_sha256'] and sha(directory/'trace.pt')==protocol['arc_trace_sha256']
    arc=torch.load(directory/'trace.pt',map_location='cpu',weights_only=False)
    source_dir=project/f'runs/multicomposition_angular_probe_v1/condition_{index:02d}'
    source_report=json.loads((source_dir/'results.json').read_text())
    assert sha(source_dir/'results.json')==protocol['source_results_sha256'][str(index)]
    assert sha(source_dir/'trace.pt')==source_report['trace_sha256']
    source=torch.load(source_dir/'trace.pt',map_location='cpu',weights_only=False)
    selected=protocol['fit_context_ids'][str(index)]
    rows=[r for r in arc if r['condition']==index and r['context'] in selected and r['method'] in protocol['methods']]
    assert len(rows)==len(selected)*len(protocol['methods'])
    assert {r['context'] for r in rows}==set(selected)
    physical_path=root/'research/evidence/parity_training_protocol_v1.json'
    assert sha(physical_path)==protocol['physical_protocol_sha256']
    return pp,protocol,audit,source_report,source,rows,json.loads(physical_path.read_text())


def evaluate_rows(target,source,rows):
    candidates=[];metadata=[]
    for record in rows:
        context=source['contexts'][record['context']];center=source['states'][context['center_state_id']]
        sample=record['samples'][0]
        assert sample['draw']==0 and sample['valid'] and sample['direction'] is not None
        assert torch.isfinite(sample['forward_log_q']) and torch.isfinite(sample['reverse_log_q'])
        x=center['positions'];leaf,anchor=context['root'];radius=(x[leaf]-x[anchor]).norm()
        y=x.clone();y[leaf]=x[anchor]+radius*sample['direction'];y-=y.mean(0)
        state=target.coordinate_state(y)
        assert torch.equal(state['graph']['bond_orders'],center['graph']['bond_orders'])
        candidates.append(state)
        metadata.append(dict(context=record['context'],parent=context['parent_id'],kind=context['kind'],root=context['root'],
            method=record['method'],center_state_id=context['center_state_id'],old_potential_eV=float(center['potential_eV']),
            log_forward_q=float(sample['forward_log_q']),log_reverse_q=float(sample['reverse_log_q']),
            squared_COM_displacement_A2=float((y-x).square().sum())))
    target.evaluate(candidates,phase='frozen_arc_endpoints')
    for row,state in zip(metadata,candidates):
        delta=float(state['potential_eV'])-row['old_potential_eV']
        ratio=-delta/target.kT+row['log_reverse_q']-row['log_forward_q']
        acceptance=math.exp(min(0.,ratio))
        row.update(new_state_id=state['state_id'],new_potential_eV=float(state['potential_eV']),potential_difference_eV=delta,
            log_acceptance_ratio=ratio,expected_acceptance=acceptance,expected_potential_change_eV=acceptance*delta,
            expected_squared_COM_displacement_A2=acceptance*row['squared_COM_displacement_A2'])
    return metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','out']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['oracle-python','oracle-checkpoint','run']:p.add_argument('--'+name,type=Path)
    p.add_argument('--phase',choices=['evaluate','audit'],required=True)
    p.add_argument('--index',type=int,required=True)
    args=p.parse_args();root=Path(__file__).resolve().parents[2]
    pp,protocol,audit,header,source,rows,physical=inputs(root,args.project,args.index)
    assert args.index in protocol['condition_indices']
    if args.phase=='audit':
        report=json.loads((args.run/'results.json').read_text())
        assert report['complete'] and report['protocol_sha256']==sha(pp)
        assert report['arc_audit_sha256']==sha(args.project/protocol['arc_audit'])
        assert sha(args.run/'trace.pt')==report['trace_sha256']
        saved=torch.load(args.run/'trace.pt',map_location='cpu',weights_only=False)
        oracle=ReplayOracle(saved['query_trace']);target=ChemicalTarget(oracle,header['condition'],physical['kT_eV'],physical['restraint_eV_A2'])
        actual=evaluate_rows(target,source,rows)
        equal(actual,saved['rows']);equal(target.states,saved['states']);equal(target.query_trace,saved['query_trace'])
        for row in saved['rows']:
            context=source['contexts'][row['context']];old=source['states'][context['center_state_id']]
            new=saved['states'][row['new_state_id']];query=saved['query_trace'][new['query_batch']];j=new['query_row']
            raw_mean=(float(query['raw_energy_eV'][j])+float(query['inverted_energy_eV'][j]))/2
            delta=raw_mean-float(old['energy_eV'])+physical['restraint_eV_A2']/2*float(
                (new['positions'].square()-old['positions'].square()).sum())
            ratio=-delta/physical['kT_eV']+row['log_reverse_q']-row['log_forward_q']
            assert abs(ratio-row['log_acceptance_ratio'])<1e-7
            assert abs(math.exp(min(0.,ratio))*delta-row['expected_potential_change_eV'])<1e-8
        assert oracle.evaluated==report['new_raw_queries']==report['requested_raw_queries']==2*len(rows)
        assert oracle.index==len(oracle.queries)
        result=dict(complete=True,index=args.index,results_sha256=sha(args.run/'results.json'),trace_sha256=sha(args.run/'trace.pt'),
            full_replay=True,independent_raw_energy_MH_checks=len(rows),raw_queries=oracle.evaluated,new_physical_queries=0,rows=len(rows),scientific_submission_ready=False)
        if args.out.exists():raise FileExistsError(args.out)
        args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));return
    args.out.mkdir(parents=True,exist_ok=True);output=args.out/'results.json'
    if output.exists():raise FileExistsError(output)
    result=dict(complete=False,index=args.index,protocol_sha256=sha(pp),condition=header['condition'],arc_audit_sha256=sha(args.project/protocol['arc_audit']),
        source_results_sha256=sha(args.project/f'runs/multicomposition_angular_probe_v1/condition_{args.index:02d}/results.json'),
        fit_context_ids=protocol['fit_context_ids'][str(args.index)],new_raw_queries=0,new_model_fitted=False,scientific_submission_ready=False)
    write(output,result);oracle=target=None
    try:
        assert sha(args.oracle_checkpoint)==physical['raw_oracle_sha256']
        oracle=EnergyOracle(args.oracle_python,root/'scripts/research/oracle_worker.py',args.oracle_checkpoint,
            numbers=header['condition']['numbers'],charge=header['condition']['charge'],spin_multiplicity=header['condition']['spin_multiplicity'],device='cuda',batch_size=32)
        assert oracle.handshake['base_precision_dtype']=='torch.float32' and not oracle.handshake['tf32']
        target=ChemicalTarget(oracle,header['condition'],physical['kT_eV'],physical['restraint_eV_A2'])
        data=evaluate_rows(target,source,rows)
        torch.save(dict(rows=data,states=target.states,query_trace=target.query_trace),args.out/'trace.pt')
        assert oracle.evaluated==oracle.requested_evaluations==2*len(rows)
        result.update(complete=True,rows=data,trace_sha256=sha(args.out/'trace.pt'),new_raw_queries=oracle.evaluated,
            requested_raw_queries=oracle.requested_evaluations,oracle_runtime=oracle.handshake,
            scope='First frozen draw per FIT context and scoring rule. Expected one-step MH diagnostics with the per-context surrogate held fixed; not a trained learner, joint molecular sampler, cost-matched learning benefit or equilibrium result.')
        write(output,result);print(json.dumps(dict(index=args.index,endpoints=len(rows),new_raw_queries=oracle.evaluated)))
    except Exception as exc:
        if target is not None:torch.save(dict(states=target.states,query_trace=target.query_trace),args.out/'failed_trace.pt')
        result.update(failure=f'{type(exc).__name__}: {exc}',new_raw_queries=oracle.evaluated if oracle else 0,
            requested_raw_queries=oracle.requested_evaluations if oracle else 0);write(output,result);raise
    finally:
        if oracle is not None:oracle.close()


if __name__=='__main__':main()
