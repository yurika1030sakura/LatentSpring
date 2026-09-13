#!/usr/bin/env python3
"""Validate frozen feasible boundary steps with the actual declared potential."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from cfm_mol.chemical_sampler import ChemicalTarget
from cfm_mol.energy_oracle import EnergyOracle
from scripts.research.audit_masked_angular import ReplayOracle,equal,sha
from scripts.research.evaluate_chemical_policy import write


def run(target,source,rows,protocol):
    ids=sorted({row['state_id'] for row in rows});fresh=target.evaluate([target.coordinate_state(source['states'][sid]['positions']) for sid in ids],phase='fresh_sources')
    mapping=dict(zip(ids,fresh));repeat=target.evaluate([target.coordinate_state(s['positions']) for s in fresh],phase='repeat_sources')
    for sid in ids:torch.testing.assert_close(mapping[sid]['graph']['bond_orders'],source['states'][sid]['graph']['bond_orders'],atol=0,rtol=0)
    repeat_energy=max(abs(float(s['potential_eV']-r['potential_eV'])) for s,r in zip(fresh,repeat))
    repeat_force=max(float((s['force_eV_A']-r['force_eV_A']).abs().max()) for s,r in zip(fresh,repeat))
    cache_energy=max(abs(float(mapping[sid]['potential_eV']-source['states'][sid]['potential_eV'])) for sid in ids)
    cache_force=max(float((mapping[sid]['force_eV_A']-source['states'][sid]['force_eV_A']).abs().max()) for sid in ids)
    assert max(repeat_energy,cache_energy)<=protocol['energy_tolerance_eV']
    assert max(repeat_force,cache_force)<=protocol['force_tolerance_eV_A']
    candidates=[];indices=[]
    for index,row in enumerate(rows):
        if row['feasible'] is None:continue
        state=target.coordinate_state(torch.tensor(row['feasible']['positions'],dtype=torch.float64));old=mapping[row['state_id']]
        torch.testing.assert_close(state['graph']['bond_orders'],old['graph']['bond_orders'],atol=0,rtol=0)
        arm=next(v for v in source['arms'] if all(v[k]==row[k] for k in ['pair_id','endpoint','mobility']))
        change=state['positions']-old['positions'];basis=arm['basis']
        torch.testing.assert_close(change,basis@(basis.T@change),atol=1e-12,rtol=0)
        assert float(change.norm(dim=1).max())<=.05+1e-12
        predicted=float(((source['states'][row['state_id']]['force_eV_A']-target.restraint*old['positions'])*change).sum())
        assert abs(predicted-row['feasible']['force_predicted_decrease_eV'])<1e-10 and predicted>0
        candidates.append(state);indices.append(index)
    target.evaluate(candidates,phase='frozen_candidates');lookup=dict(zip(indices,candidates));result=[]
    for index,row in enumerate(rows):
        item={k:row[k] for k in ['pair_id','parent','endpoint','mobility','mode','state_id']};item['feasible']=index in lookup
        if index in lookup:
            candidate=lookup[index];old=mapping[row['state_id']]
            item.update(source_fresh_state_id=old['state_id'],candidate_state_id=candidate['state_id'],
                predicted_decrease_eV=row['feasible']['force_predicted_decrease_eV'],actual_decrease_eV=float(old['potential_eV']-candidate['potential_eV']),
                max_atom_step_A=row['feasible']['max_atom_step_A'])
        result.append(item)
    return dict(rows=result,states=target.states,query_trace=target.query_trace,repeat_energy_error_eV=repeat_energy,repeat_force_error_eV_A=repeat_force,
        cached_energy_error_eV=cache_energy,cached_force_error_eV_A=cache_force,initial_repeat_raw_queries=4*len(ids),candidate_raw_queries=2*len(candidates))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','out']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['run','oracle-python','oracle-checkpoint']:p.add_argument('--'+name,type=Path)
    p.add_argument('--index',type=int,required=True);p.add_argument('--phase',choices=['evaluate','audit'],required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[2];pp=root/'research/evidence/mobility_boundary_energy_protocol_v1.json';protocol=json.loads(pp.read_text())
    assert protocol['frozen'] and a.index in protocol['condition_indices']
    dp=root/protocol['directions'];assert sha(dp)==protocol['directions_sha256'];directions=json.loads(dp.read_text())
    rows=[v for v in directions['rows'] if v['index']==a.index];assert rows
    sp=a.project/protocol['source_run']/f'condition_{a.index:02d}';header=json.loads((sp/'results.json').read_text());frozen=protocol['sources'][str(a.index)]
    assert header['complete'] and sha(sp/'results.json')==frozen['results_sha256'] and sha(sp/'trace.pt')==frozen['trace_sha256']
    source=torch.load(sp/'trace.pt',map_location='cpu',weights_only=False)
    physical_path=root/protocol['physical_protocol'];assert sha(physical_path)==protocol['physical_protocol_sha256'];physical=json.loads(physical_path.read_text())
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
            oracle=EnergyOracle(a.oracle_python,root/'scripts/research/oracle_worker.py',a.oracle_checkpoint,
                numbers=header['condition']['numbers'],charge=header['condition']['charge'],spin_multiplicity=header['condition']['spin_multiplicity'],device='cuda',batch_size=1)
            assert oracle.handshake['base_precision_dtype']=='torch.float32' and not oracle.handshake['tf32']
        target=ChemicalTarget(oracle,header['condition'],physical['kT_eV'],physical['restraint_eV_A2']);actual=run(target,source,rows,protocol)
        assert oracle.evaluated==actual['initial_repeat_raw_queries']+actual['candidate_raw_queries']<=frozen['maximum_raw_queries']
        if a.phase=='audit':
            equal(actual,expected);assert oracle.index==len(oracle.queries) and oracle.evaluated==producer['new_raw_queries']
            for state in actual['states']:
                q=actual['query_trace'][state['query_batch']];j=state['query_row'];x=state['positions'].numpy()
                energy=(float(q['raw_energy_eV'][j])+float(q['inverted_energy_eV'][j]))/2+physical['restraint_eV_A2']/2*np.sum(x*x)
                force=(q['raw_force_eV_A'][j].numpy()-q['inverted_force_eV_A'][j].numpy())/2
                assert abs(energy-float(state['potential_eV']))<1e-8 and np.max(np.abs(force-state['force_eV_A'].numpy()))<1e-10
            report.update(complete=True,full_replay=True,source_results_sha256=sha(a.run/'results.json'),trace_sha256=producer['trace_sha256'],
                independently_reconstructed_physical_states=len(actual['states']),raw_queries_in_producer=oracle.evaluated)
        else:
            assert oracle.evaluated==oracle.requested_evaluations
            torch.save(actual,a.out/'trace.pt');report.update(complete=True,rows=actual['rows'],initial_repeat_raw_queries=actual['initial_repeat_raw_queries'],candidate_raw_queries=actual['candidate_raw_queries'],
                new_raw_queries=oracle.evaluated,requested_raw_queries=oracle.requested_evaluations,trace_sha256=sha(a.out/'trace.pt'),oracle_runtime=oracle.handshake,
                oracle_evaluation_seconds=oracle.evaluation_seconds,numerical_consistency={k:actual[k] for k in ['repeat_energy_error_eV','repeat_force_error_eV_A','cached_energy_error_eV','cached_force_error_eV_A']})
        report['elapsed_seconds']=time.monotonic()-start;write(output,report)
        print(json.dumps({k:v for k,v in report.items() if k not in ['rows','oracle_runtime']}))
    except Exception as exc:
        if target is not None:torch.save(dict(states=target.states,query_trace=target.query_trace),a.out/'failed_trace.pt')
        report.update(failure=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-start)
        if oracle is not None and a.phase=='evaluate':report.update(new_raw_queries=oracle.evaluated,requested_raw_queries=oracle.requested_evaluations)
        write(output,report);raise
    finally:
        if oracle is not None and a.phase=='evaluate':oracle.close()


if __name__=='__main__':main()
