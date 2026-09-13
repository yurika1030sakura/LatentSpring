#!/usr/bin/env python3
"""Bounded FIT-only four-state experiment for interactions between chemical edits."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import torch
from cfm_mol.chemical_edit_interaction import independent_actions, four_positions, four_graphs, mixed_difference, restraint_interaction
from cfm_mol.chemical_sampler import ChemicalTarget
from cfm_mol.energy_oracle import EnergyOracle
from scripts.research.audit_masked_angular import ReplayOracle, equal, sha
from scripts.research.evaluate_chemical_policy import write


def build(root, project, out):
    original_path=project/'runs/chemical_work_catalogue_plan_v1/plan.pt'
    original_protocol=json.loads((root/'research/evidence/chemical_work_catalogue_protocol_v1.json').read_text())
    assert sha(original_path)==original_protocol['plan_sha256']
    original=torch.load(original_path,map_location='cpu',weights_only=False)
    audit_summary=json.loads((root/'research/evidence/chemical_work_catalogue_summary_v1.json').read_text())
    physical=json.loads((root/original_protocol['physical_protocol']).read_text())
    assert sha(root/original_protocol['physical_protocol'])==original_protocol['physical_protocol_sha256']
    sources=[];pairs=[];provenance={}
    for index in [1,2,3,5]:
        path=project/f'runs/chemical_work_catalogue_v1/condition_{index:02d}/trace.pt'
        assert sha(path)==audit_summary['sources'][str(index)]['trace_sha256']
        data=torch.load(path,map_location='cpu',weights_only=False)
        provenance[str(index)]=dict(trace=str(path.relative_to(project)),trace_sha256=sha(path))
        for spec in (s for s in original['sources'] if s['index']==index):
            assert spec['fit_only']
            rows=[r for r in data['rows'] if r['source_id']==spec['source_id'] and r['valid']]
            old=data['states'][rows[0]['source_state_id']]
            source=dict(source_id=spec['source_id'],index=index,parent=spec['parent'],fit_only=True,
                        condition=spec['condition'],state=old)
            target=ChemicalTarget(None,spec['condition'],physical['kT_eV'],physical['restraint_eV_A2'])
            candidates=[(a,b) for a,b in itertools.combinations(rows,2) if independent_actions(a['action'],b['action'])]
            def order(pair):
                key=json.dumps([28703,spec['source_id'],pair[0]['action'],pair[1]['action']])
                return hashlib.sha256(key.encode()).hexdigest()
            candidates.sort(key=order)
            source.update(eligible_pairs=len(candidates),selected_pairs=min(32,len(candidates)))
            sources.append(source)
            for ra,rb in candidates[:32]:
                a,b=tuple(ra['action']),tuple(rb['action'])
                corners,volume,inverse=four_positions(old['positions'],target.radii,a,b)
                graphs=four_graphs(old['graph']['bond_orders'],a,b)
                sa=data['states'][ra['candidate_state_id']];sb=data['states'][rb['candidate_state_id']]
                for j,state in [(1,sa),(2,sb)]:
                    torch.testing.assert_close(corners[j],state['positions'],atol=1e-10,rtol=0)
                    assert torch.equal(graphs[j],state['graph']['bond_orders'])
                item=dict(source_id=spec['source_id'],index=index,parent=spec['parent'],actions=[a,b],inverse_actions=inverse,
                    positions=corners,log_volume=float(volume),cached_states=[old,sa,sb],valid=False)
                try:
                    end=target.coordinate_state(corners[3])
                    if not torch.equal(end['graph']['bond_orders'],graphs[3]):raise ValueError('Combined endpoint graph differs')
                    inverse_square,jrev,_=four_positions(corners[3],target.radii,*inverse)
                    torch.testing.assert_close(inverse_square[3],old['positions'],atol=1e-9,rtol=0)
                    assert abs(float(volume+jrev))<1e-10
                    item.update(valid=True,candidate=end)
                except ValueError as exc:item['failure']=str(exc)
                pairs.append(item)
    assert len(sources)==36 and all(s['fit_only'] for s in sources)
    counts={}
    for index in [1,2,3,5]:
        ss=[s for s in sources if s['index']==index];pp=[p for p in pairs if p['index']==index]
        counts[str(index)]=dict(parents=len(ss),eligible_pairs=sum(s['eligible_pairs'] for s in ss),selected_pairs=len(pp),
            valid=sum(p['valid'] for p in pp),invalid=sum(not p['valid'] for p in pp),
            source_checks=sum(s['selected_pairs']>0 for s in ss))
        counts[str(index)]['raw_queries']=2*(counts[str(index)]['source_checks']+counts[str(index)]['valid'])
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);torch.save(dict(sources=sources,pairs=pairs),out/'plan.pt')
    write(out/'results.json',dict(complete=True,plan_sha256=sha(out/'plan.pt'),counts=counts,
        maximum_new_raw_queries=sum(c['raw_queries'] for c in counts.values()),old_catalogue=provenance,
        physical_protocol=original_protocol['physical_protocol'],physical_protocol_sha256=original_protocol['physical_protocol_sha256'],
        scope='All36 FIT parents, hash-first at most32 independent pairs per parent before combined validity/energy. Both single endpoints must be valid. Zero-eligible parents and combined failures retained. Mechanism data, not a sampler.'))


def score(target, sources, pairs):
    active_sources=[s for s in sources if s['selected_pairs']>0]
    initial=[target.coordinate_state(s['state']['positions']) for s in active_sources]
    target.evaluate(initial,phase='check_cached_sources')
    for state,source in zip(initial,active_sources):
        assert abs(float(state['potential_eV']-source['state']['potential_eV']))<1e-4
        equal(state['graph'],source['state']['graph'])
    pending=[]
    for pair in pairs:
        if not pair['valid']:continue
        candidate=target.coordinate_state(pair['positions'][3]);equal(candidate,pair['candidate']);pending.append(candidate)
    target.evaluate(pending,phase='combined_endpoints')
    rows=[];offset=0
    for pair in pairs:
        row={key:pair[key] for key in ('source_id','parent','actions','inverse_actions','valid','log_volume')}
        if not pair['valid']:
            row['failure']=pair['failure'];rows.append(row);continue
        end=pending[offset];offset+=1
        energies=torch.tensor([float(s['energy_eV']) for s in pair['cached_states']]+[float(end['energy_eV'])],dtype=torch.float64)
        potentials=torch.tensor([float(s['potential_eV']) for s in pair['cached_states']]+[float(end['potential_eV'])],dtype=torch.float64)
        interaction=float(mixed_difference(energies));known=float(restraint_interaction(pair['positions'],target.restraint))
        assert abs(float(mixed_difference(potentials))-interaction-known)<1e-7
        wa,wb=float(potentials[1]-potentials[0]),float(potentials[2]-potentials[0])
        joint=float(potentials[3]-potentials[0])
        row.update(candidate_state_id=end['state_id'],electronic_energies_eV=energies.tolist(),potentials_eV=potentials.tolist(),
            electronic_interaction_eV=interaction,restraint_interaction_eV=known,
            work_a_eV=wa,work_b_eV=wb,joint_work_eV=joint,oracle_additive_joint_work_eV=wa+wb+known,
            electronic_sign_flip=(joint<0)!=(wa+wb+known<0),both_singles_uphill_joint_downhill=wa>0 and wb>0 and joint<0,
            joint_better_than_either_single=joint<min(wa,wb),
            scope='Oracle-additive is a diagnostic with true single energies, not a deployable zero-query control.')
        rows.append(row)
    return dict(rows=rows,states=target.states,query_trace=target.query_trace)


def audit_arithmetic(actual,pairs,target):
    checked=0
    for row,pair in zip(actual['rows'],pairs):
        if not row['valid']:continue
        state=actual['states'][row['candidate_state_id']];q=actual['query_trace'][state['query_batch']];j=state['query_row']
        raw=(float(q['raw_energy_eV'][j])+float(q['inverted_energy_eV'][j]))/2
        e0,ea,eb=[float(s['energy_eV']) for s in pair['cached_states']]
        assert abs(raw-ea-eb+e0-row['electronic_interaction_eV'])<1e-7
        old=pair['positions'][0];da=pair['positions'][1]-old;db=pair['positions'][2]-old
        assert abs(target.restraint*float((da*db).sum())-row['restraint_interaction_eV'])<1e-9
        checked+=1
    return checked


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('project','out'):parser.add_argument('--'+name,type=Path,required=True)
    for name in ('run','oracle-python','oracle-checkpoint'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--index',type=int);parser.add_argument('--phase',choices=['build','score','audit'],required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[2]
    if args.phase=='build':build(root,args.project,args.out);return
    pp=root/'research/evidence/chemical_edit_interaction_protocol_v1.json';protocol=json.loads(pp.read_text());assert protocol['frozen']
    plan_path=args.project/protocol['plan'];assert sha(plan_path)==protocol['plan_sha256']
    plan=torch.load(plan_path,map_location='cpu',weights_only=False)
    sources=[s for s in plan['sources'] if s['index']==args.index];pairs=[p for p in plan['pairs'] if p['index']==args.index]
    assert all(s['fit_only'] for s in sources)
    physical_path=root/protocol['physical_protocol'];assert sha(physical_path)==protocol['physical_protocol_sha256']
    physical=json.loads(physical_path.read_text());condition=sources[0]['condition']
    args.out.mkdir(parents=True,exist_ok=True);output=args.out/'results.json'
    if output.exists():raise FileExistsError(output)
    report=dict(complete=False,index=args.index,phase=args.phase,protocol_sha256=sha(pp),new_raw_queries=0,scientific_submission_ready=False)
    write(output,report);oracle=target=None
    try:
        if args.phase=='audit':
            producer=json.loads((args.run/'results.json').read_text())
            assert producer['complete'] and producer['protocol_sha256']==sha(pp) and sha(args.run/'trace.pt')==producer['trace_sha256']
            expected=torch.load(args.run/'trace.pt',map_location='cpu',weights_only=False);oracle=ReplayOracle(expected['query_trace'])
        else:
            assert sha(args.oracle_checkpoint)==physical['raw_oracle_sha256']
            oracle=EnergyOracle(args.oracle_python,root/'scripts/research/oracle_worker.py',args.oracle_checkpoint,
                numbers=condition['numbers'],charge=condition['charge'],spin_multiplicity=condition['spin_multiplicity'],device='cuda',batch_size=32)
            assert oracle.handshake['base_precision_dtype']=='torch.float32' and not oracle.handshake['tf32']
        target=ChemicalTarget(oracle,condition,physical['kT_eV'],physical['restraint_eV_A2']);actual=score(target,sources,pairs)
        assert oracle.evaluated==protocol['counts'][str(args.index)]['raw_queries']
        checks=audit_arithmetic(actual,pairs,target)
        if args.phase=='audit':
            equal(actual,expected);assert oracle.index==len(oracle.queries)
            report.update(complete=True,full_replay=True,source_results_sha256=sha(args.run/'results.json'),trace_sha256=producer['trace_sha256'],
                producer_raw_queries=oracle.evaluated,independent_four_state_checks=checks)
        else:
            assert oracle.evaluated==oracle.requested_evaluations
            torch.save(actual,args.out/'trace.pt');report.update(complete=True,trace_sha256=sha(args.out/'trace.pt'),rows=actual['rows'],
                new_raw_queries=oracle.evaluated,independent_four_state_checks=checks)
        write(output,report);print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
    except Exception as exc:
        if target is not None:torch.save(dict(states=target.states,query_trace=target.query_trace),args.out/'failed_trace.pt')
        report['failure']=f'{type(exc).__name__}: {exc}'
        if oracle is not None and args.phase=='score':report['new_raw_queries']=oracle.evaluated
        write(output,report);raise
    finally:
        if oracle is not None and args.phase=='score':oracle.close()


if __name__=='__main__':main()
