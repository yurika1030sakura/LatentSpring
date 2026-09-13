import torch
import copy
import pytest
from cfm_mol.mobility_relaxation import mobility_basis,relax_arms


def options(**changes):
    result=dict(force_tolerance_eV_A=1e-8,initial_inverse_hessian_A2_eV=.01,max_atom_step_A=.05,
        minimum_atom_step_A=1e-10,max_backtracks=24,armijo_c1=1e-4,energy_noise_tolerance_eV=0.,
        curvature_relative_floor=1e-10,history_size=5,max_evaluations=32,readouts=[8,16,32])
    result.update(changes);return result


class QuadraticTarget:
    restraint=0.
    def __init__(self,centre,limit=None):
        self.centre=centre;self.limit=limit;self.states=[]
        self.oracle=type('Counter',(),dict(evaluated=0))()
    def coordinate_state(self,x):
        if self.limit is not None and float(x[0,0])>self.limit:raise ValueError('Outside toy domain')
        return dict(positions=x.clone(),graph=dict(bond_orders=torch.zeros(len(x),len(x),dtype=x.dtype)))
    def evaluate(self,states,phase):
        for state in states:
            delta=state['positions']-self.centre
            state.update(force_eV_A=-delta,potential_eV=.5*delta.square().sum(),state_id=len(self.states))
            self.states.append(state)
        self.oracle.evaluated+=2*len(states);return states


def test_orthonormal_root_chart_preserves_passive_relative_positions():
    basis=mobility_basis(5,[0,3],'roots')
    torch.testing.assert_close(basis.T@basis,torch.eye(2,dtype=basis.dtype),atol=1e-14,rtol=0)
    torch.testing.assert_close(basis.sum(0),torch.zeros(2,dtype=basis.dtype),atol=1e-14,rtol=0)
    change=basis@torch.tensor([[.1,.2,.3],[-.2,.1,-.3]],dtype=basis.dtype)
    torch.testing.assert_close(change[[1,2,4]],change[[1,1,1]],atol=1e-14,rtol=0)


def test_root_and_collective_relaxation_match_quadratic_constrained_minima():
    x=torch.tensor([[.15,.02,0],[-.02,.10,0],[-.03,-.05,.04],[-.1,-.07,-.04]],dtype=torch.float64)
    target=QuadraticTarget(torch.zeros_like(x));initial=target.evaluate([target.coordinate_state(x),target.coordinate_state(x*.8)],phase='initial')
    pairs=[dict(pair_id=0,parent=0,roots=[0,1],source_state_id=initial[0]['state_id'],destination_state_id=initial[1]['state_id'])]
    before=target.oracle.evaluated;arms=relax_arms(target,pairs,options())
    assert target.oracle.evaluated-before==2*sum(a['evaluations'] for a in arms)
    for arm in arms:
        origin=target.states[arm['initial_state_id']]['positions'];basis=arm['basis']
        optimum=origin-basis@(basis.T@origin)
        best=target.states[arm['best_state_id']]['positions']
        torch.testing.assert_close(best,optimum,atol=1e-7,rtol=0)
        assert arm['status'].startswith('converged') and arm['evaluations']<=32
        assert arm['best_potential_eV']<=arm['initial_potential_eV']
        if arm['mobility']=='roots':
            torch.testing.assert_close(best[2]-best[3],origin[2]-origin[3],atol=1e-13,rtol=0)


def test_geometry_backtracking_is_retained_without_oracle_charges():
    x=torch.zeros(4,3,dtype=torch.float64);centre=x.clone();centre[0,0]=.3;centre[1:,0]=-.1
    target=QuadraticTarget(centre,limit=.03);initial=target.evaluate([target.coordinate_state(x),target.coordinate_state(x)],phase='initial')
    pairs=[dict(pair_id=0,parent=0,roots=[0,1],source_state_id=0,destination_state_id=1)]
    before=target.oracle.evaluated;arms=relax_arms(target,pairs,options(max_evaluations=8,force_tolerance_eV_A=1e-5))
    assert any(not e['queried'] for arm in arms for e in arm['events'])
    assert target.oracle.evaluated-before==2*sum(e['queried'] for arm in arms for e in arm['events'])
    assert all(arm['evaluations']<=8 for arm in arms)
    for arm in arms:
        for event in arm['events']:
            if event['queried']:assert float(event['positions'][0,0])<=.03


def test_real_graph_optimizer_replays_all_queries_and_independent_checks():
    from cfm_mol.chemical_sampler import ChemicalTarget
    from scripts.research.mobility_relaxation_pilot import run,independent_audit
    from scripts.research.audit_masked_angular import ReplayOracle,equal
    from test_bounded_action_geometry import pair_record
    row=pair_record();row['source_force_eV_A']=-.02*row['x'];row['candidate_force_eV_A']=-.02*row['y']
    row['reward_eV']=-.06*(row['y'].square().sum()-row['x'].square().sum())
    class Oracle:
        evaluated=0
        def evaluate_chunked(self,p,max_request):
            self.evaluated+=len(p);return .01*p.square().sum((1,2)),-.02*p
    def target(oracle):return ChemicalTarget(oracle,dict(numbers=row['numbers'].tolist(),charge=0,spin_multiplicity=1),.026,.1)
    protocol=dict(options=options(max_evaluations=4,readouts=[2,4],force_tolerance_eV_A=.05),fresh_initial_repeats=2,
        repeat_energy_tolerance_eV=1e-10,repeat_force_tolerance_eV_A=1e-10,archived_gap_tolerance_eV=1e-10,archived_force_tolerance_eV_A=1e-10)
    rows=[(dict(pair_id=0,parent=0),row)]
    first=target(Oracle());saved=run(first,rows,protocol)
    replay=ReplayOracle(saved['query_trace']);actual=run(target(replay),rows,protocol)
    equal(actual,saved);assert replay.index==len(replay.queries)
    checks=independent_audit(saved,.1,protocol['options'])
    assert checks['queried_trials_checked']==sum(a['evaluations'] for a in saved['arms'])>0
    from scripts.research.continue_mobility_relaxation import resume
    frozen=copy.deepcopy(saved);extended_options=dict(protocol['options'],max_evaluations=8,readouts=[2,4,8])
    extended,counts=resume(first,saved,extended_options);assert counts['new_raw_queries']>0
    replay=ReplayOracle(extended['query_trace']);replay_target=target(replay)
    prefix=run(replay_target,rows,protocol);equal(prefix,frozen)
    replayed,replayed_counts=resume(replay_target,prefix,extended_options)
    equal(replayed,extended);equal(replayed_counts,counts)
    assert replay.index==len(replay.queries)
    independent_audit(extended,.1,extended_options)


def test_resume_reaches_same_quadratic_points_without_repeating_prefix_queries():
    from scripts.research.audit_masked_angular import equal
    x=torch.tensor([[.15,.02,0],[-.02,.10,0],[-.03,-.05,.04],[-.1,-.07,-.04]],dtype=torch.float64)
    def setup():
        target=QuadraticTarget(torch.zeros_like(x));target.evaluate([target.coordinate_state(x),target.coordinate_state(x*.8)],phase='initial')
        return target,[dict(pair_id=0,parent=0,roots=[0,1],source_state_id=0,destination_state_id=1)]
    full,pairs=setup();reference=relax_arms(full,pairs,options())
    target,pairs=setup();prefix=relax_arms(target,pairs,options(max_evaluations=2,readouts=[]));snapshot=copy.deepcopy(prefix)
    before=target.oracle.evaluated;states=copy.deepcopy(target.states)
    continued=relax_arms(target,pairs,options(),resume_arms=prefix)
    equal(prefix,snapshot);equal(target.states[:len(states)],states)
    assert target.oracle.evaluated-before==2*sum(a['evaluations']-b['evaluations'] for a,b in zip(continued,prefix))
    for a,b in zip(continued,reference):
        assert a['status']==b['status'] and a['evaluations']==b['evaluations']
        torch.testing.assert_close(target.states[a['best_state_id']]['positions'],full.states[b['best_state_id']]['positions'],atol=1e-12,rtol=0)


def test_resume_preserves_terminal_boundary_stops_and_rejects_same_cap():
    from scripts.research.audit_masked_angular import equal
    x=torch.zeros(4,3,dtype=torch.float64);centre=x.clone();centre[0,0]=.3;centre[1:,0]=-.1
    target=QuadraticTarget(centre,limit=.03);target.evaluate([target.coordinate_state(x),target.coordinate_state(x)],phase='initial')
    pairs=[dict(pair_id=0,parent=0,roots=[0,1],source_state_id=0,destination_state_id=1)]
    prefix=relax_arms(target,pairs,options(max_evaluations=128));assert all(a['status'] in ['step_too_small','line_search_blocked'] for a in prefix)
    before=target.oracle.evaluated;continued=relax_arms(target,pairs,options(max_evaluations=256),resume_arms=prefix)
    equal(continued,prefix);assert target.oracle.evaluated==before
    capped=copy.deepcopy(prefix);capped[0]['status']='budget_exhausted'
    with pytest.raises(ValueError,match='cap must increase'):
        relax_arms(target,pairs,options(max_evaluations=capped[0]['evaluations']),resume_arms=capped)
