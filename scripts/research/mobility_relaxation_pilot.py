#!/usr/bin/env python3
"""Paired source/destination mobility diagnostic with explicit physical costs."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from cfm_mol.chemical_sampler import ChemicalTarget
from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.mobility_relaxation import relax_arms
from scripts.research.audit_masked_angular import ReplayOracle,equal,sha
from scripts.research.evaluate_chemical_policy import write


def inputs(root,project,index):
    pp=root/'research/evidence/mobility_relaxation_protocol_v1.json';protocol=json.loads(pp.read_text())
    assert protocol['frozen'] and index in protocol['condition_indices']
    assert sha(root/'research/evidence/mobility_relaxation_settings_v1.json')==protocol['settings_sha256']
    directory=project/protocol['data_run'];header=json.loads((directory/'results.json').read_text())
    assert header['complete'] and sha(directory/'results.json')==protocol['data_results_sha256']
    assert sha(directory/'data.pt')==header['data_sha256']==protocol['data_sha256']
    data=torch.load(directory/'data.pt',map_location='cpu',weights_only=False)
    eligible=sorted({r['parent'] for r in data if r['index']==index and r['role']=='fit'})
    selected=sorted(eligible,key=lambda pid:hashlib.sha256(f"{protocol['selection_seed']}|parent|{index}|{pid}".encode()).hexdigest())[:protocol['parents_per_condition']]
    assert selected==[s['parent'] for s in protocol['selections'][str(index)]]
    rows=[]
    for spec in protocol['selections'][str(index)]:
        choices=[(i,r) for i,r in enumerate(data) if r['index']==index and r['parent']==spec['parent'] and r['role']=='fit' and r['valid']]
        offset,row=min(choices,key=lambda item:hashlib.sha256(f"{protocol['selection_seed']}|pair|{index}|{spec['parent']}|{item[1]['replica']}|{item[1]['step']}".encode()).hexdigest())
        assert offset==spec['data_index'] and row['replica']==spec['replica'] and row['step']==spec['step'] and list(row['action'])==spec['action']
        rows.append((spec,row))
    physical_path=root/protocol['physical_protocol'];assert sha(physical_path)==protocol['physical_protocol_sha256']
    physical=json.loads(physical_path.read_text());z=rows[0][1]['numbers'];e=rows[0][1]['electronic']
    assert float(e[2])==physical['kT_eV']
    for spec,row in rows:
        torch.testing.assert_close(row['numbers'],z,atol=0,rtol=0);torch.testing.assert_close(row['electronic'],e,atol=0,rtol=0)
    condition=dict(numbers=z.tolist(),charge=int(e[0]),spin_multiplicity=int(e[1]))
    return pp,protocol,physical,condition,rows


def run(target,rows,protocol):
    assert protocol['fresh_initial_repeats']==2
    initial=[]
    for spec,row in rows:
        for key,graph in [('x','bonds'),('y','new_bonds')]:
            state=target.coordinate_state(row[key]);torch.testing.assert_close(state['graph']['bond_orders'],row[graph],atol=0,rtol=0);initial.append(state)
    target.evaluate(initial,phase='fresh_initial')
    repeats=target.evaluate([target.coordinate_state(s['positions']) for s in initial],phase='initial_repeat')
    repeat_energy=max(abs(float(a['potential_eV']-b['potential_eV'])) for a,b in zip(initial,repeats))
    repeat_force=max(float((a['force_eV_A']-b['force_eV_A']).abs().max()) for a,b in zip(initial,repeats))
    assert repeat_energy<=protocol['repeat_energy_tolerance_eV'] and repeat_force<=protocol['repeat_force_tolerance_eV_A']
    pairs=[];initial_comparisons=[]
    for offset,(spec,row) in enumerate(rows):
        source,destination=initial[2*offset:2*offset+2]
        gap=float(destination['potential_eV']-source['potential_eV']);gap_error=abs(gap+float(row['reward_eV']))
        force_error=max(float((source['force_eV_A']-row['source_force_eV_A']).abs().max()),float((destination['force_eV_A']-row['candidate_force_eV_A']).abs().max()))
        assert gap_error<=protocol['archived_gap_tolerance_eV'] and force_error<=protocol['archived_force_tolerance_eV_A']
        pair=dict(pair_id=spec['pair_id'],parent=spec['parent'],roots=list(row['action'][:2]),action=list(row['action']),
            connectivity_changed=row['connectivity_changed'],source_state_id=source['state_id'],destination_state_id=destination['state_id'])
        pairs.append(pair);initial_comparisons.append(dict(pair_id=spec['pair_id'],gap_eV=gap,archived_gap_error_eV=gap_error,archived_force_error_eV_A=force_error))
    initial_queries=target.oracle.evaluated
    assert initial_queries==8*len(rows)
    arms=relax_arms(target,pairs,protocol['options'])
    assert target.oracle.evaluated==initial_queries+2*sum(a['evaluations'] for a in arms)
    return dict(pairs=pairs,arms=arms,states=target.states,query_trace=target.query_trace,initial_queries=initial_queries,
        repeat_energy_error_eV=repeat_energy,repeat_force_error_eV_A=repeat_force,initial_comparisons=initial_comparisons)


def independent_audit(saved,restraint,options):
    checked=0;maximum_threshold_error=0.
    for state in saved['states']:
        q=saved['query_trace'][state['query_batch']];j=state['query_row'];x=state['positions'].numpy()
        force=(q['raw_force_eV_A'][j].numpy()-q['inverted_force_eV_A'][j].numpy())/2
        energy=(float(q['raw_energy_eV'][j])+float(q['inverted_energy_eV'][j]))/2+restraint/2*np.sum(x*x)
        assert abs(energy-float(state['potential_eV']))<1e-8
        assert np.max(np.abs(force-state['force_eV_A'].numpy()))<1e-10
    for arm in saved['arms']:
        initial=saved['states'][arm['initial_state_id']];x0=initial['positions'].numpy();n=len(x0);basis=arm['basis'].numpy()
        assert np.max(np.abs(basis.T@basis-np.eye(basis.shape[1])))<1e-12
        assert np.max(np.abs(basis.sum(0)))<1e-12
        pair=next(p for p in saved['pairs'] if p['pair_id']==arm['pair_id']);roots=pair['roots']
        if arm['mobility']=='collective':projection=np.eye(n)-np.ones((n,n))/n
        else:
            v=np.eye(n)[:,roots]-1/n;projection=v@np.linalg.solve(v.T@v,v.T)
        assert np.max(np.abs(projection-basis@basis.T))<1e-12
        queried=0
        for event in arm['events']:
            if not event['queried']:continue
            queried+=1;state=saved['states'][event['state_id']];old=saved['states'][event['origin_state_id']]
            x=state['positions'].numpy();old_x=old['positions'].numpy();force=old['force_eV_A'].numpy()-restraint*old_x
            threshold=float(old['potential_eV'])-options['armijo_c1']*float(np.sum(force*(x-old_x)))+options['energy_noise_tolerance_eV']
            error=abs(threshold-event['armijo_threshold_eV']);assert error<1e-8;maximum_threshold_error=max(maximum_threshold_error,error)
            assert event['accepted']==(float(state['potential_eV'])<=threshold)
            assert np.max(np.linalg.norm(x-old_x,axis=1))<=options['max_atom_step_A']+1e-9
            assert np.max(np.abs(x.mean(0)))<1e-9
            if arm['mobility']=='roots':
                passive=[i for i in range(n) if i not in roots]
                assert np.max(np.abs((x[passive]-x[passive[0]])-(x0[passive]-x0[passive[0]])))<1e-9
            torch.testing.assert_close(state['graph']['bond_orders'],arm['frozen_bonds'],atol=0,rtol=0);checked+=1
        assert queried==arm['evaluations']<=options['max_evaluations']
        best=saved['states'][arm['best_state_id']];projected=projection@(best['force_eV_A'].numpy()-restraint*best['positions'].numpy())
        assert abs(float(np.linalg.norm(projected,axis=1).max())-arm['best_projected_force_max_eV_A'])<1e-8
        feasible=[initial]+[saved['states'][e['state_id']] for e in arm['events'] if e['queried']]
        assert abs(min(float(s['potential_eV']) for s in feasible)-arm['best_potential_eV'])<1e-8
        if arm['status'].startswith('converged'):assert arm['final_projected_force_max_eV_A']<=options['force_tolerance_eV_A']+1e-9
    return dict(queried_trials_checked=checked,maximum_independent_armijo_threshold_error_eV=maximum_threshold_error,
        all_graphs_and_passive_constraints_checked=True,all_raw_potentials_and_forces_reconstructed=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','out']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['run','oracle-python','oracle-checkpoint']:p.add_argument('--'+name,type=Path)
    p.add_argument('--index',type=int,required=True);p.add_argument('--phase',choices=['evaluate','audit'],required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[2]
    pp,protocol,physical,condition,rows=inputs(root,a.project,a.index)
    a.out.mkdir(parents=True,exist_ok=True);output=a.out/'results.json'
    if output.exists():raise FileExistsError(output)
    report=dict(complete=False,index=a.index,phase=a.phase,protocol_sha256=sha(pp),condition=condition,selection=protocol['selections'][str(a.index)],
        new_raw_queries=0,scientific_submission_ready=False,scope=protocol['interpretation'])
    write(output,report);oracle=target=None;start=time.monotonic()
    try:
        if a.phase=='audit':
            producer=json.loads((a.run/'results.json').read_text());assert producer['complete'] and producer['protocol_sha256']==sha(pp) and sha(a.run/'trace.pt')==producer['trace_sha256']
            saved=torch.load(a.run/'trace.pt',map_location='cpu',weights_only=False);oracle=ReplayOracle(saved['query_trace'])
        else:
            assert sha(a.oracle_checkpoint)==physical['raw_oracle_sha256']
            oracle=EnergyOracle(a.oracle_python,root/'scripts/research/oracle_worker.py',a.oracle_checkpoint,
                numbers=condition['numbers'],charge=condition['charge'],spin_multiplicity=condition['spin_multiplicity'],device='cuda',batch_size=protocol['oracle_batch_size'])
            assert oracle.handshake['base_precision_dtype']=='torch.float32' and not oracle.handshake['tf32']
        target=ChemicalTarget(oracle,condition,physical['kT_eV'],physical['restraint_eV_A2'])
        actual=run(target,rows,protocol)
        if a.phase=='audit':
            equal(actual,saved);assert oracle.index==len(oracle.queries) and oracle.evaluated==producer['new_raw_queries']
            report.update(complete=True,source_results_sha256=sha(a.run/'results.json'),trace_sha256=producer['trace_sha256'],full_replay=True,
                raw_queries_in_producer=oracle.evaluated,**independent_audit(saved,target.restraint,protocol['options']))
        else:
            assert oracle.evaluated==oracle.requested_evaluations<=protocol['maximum_total_new_raw_calls']//4
            torch.save(actual,a.out/'trace.pt')
            compact=[{k:arm[k] for k in ['pair_id','parent','endpoint','mobility','evaluations','geometry_probes','accepted_steps','status',
                'best_projected_force_max_eV_A','final_projected_force_max_eV_A','best_potential_eV','final_potential_eV','initial_potential_eV','best_state_id','current_state_id']} for arm in actual['arms']]
            report.update(complete=True,arms=compact,pairs=actual['pairs'],initial_comparisons=actual['initial_comparisons'],
                initial_raw_queries=actual['initial_queries'],repeat_energy_error_eV=actual['repeat_energy_error_eV'],repeat_force_error_eV_A=actual['repeat_force_error_eV_A'],
                new_raw_queries=oracle.evaluated,requested_raw_queries=oracle.requested_evaluations,trace_sha256=sha(a.out/'trace.pt'),oracle_runtime=oracle.handshake,
                oracle_evaluation_seconds=oracle.evaluation_seconds)
        report['elapsed_seconds']=time.monotonic()-start;write(output,report)
        print(json.dumps({k:v for k,v in report.items() if k not in ['arms','pairs','selection','condition','oracle_runtime','initial_comparisons']}))
    except Exception as exc:
        if target is not None:torch.save(dict(states=target.states,query_trace=target.query_trace),a.out/'failed_trace.pt')
        report.update(failure=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-start)
        if oracle is not None and a.phase=='evaluate':report.update(new_raw_queries=oracle.evaluated,requested_raw_queries=oracle.requested_evaluations)
        write(output,report);raise
    finally:
        if oracle is not None and a.phase=='evaluate':oracle.close()


if __name__=='__main__':main()
