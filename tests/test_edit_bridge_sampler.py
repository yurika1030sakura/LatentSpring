import torch
from cfm_mol.chemical_sampler import ChemicalTarget
from cfm_mol.edit_conditioned_bridge import edit_bridge,EditBridgeField,center
from cfm_mol.edit_bridge_sampler import ZeroBridgeField,AnalyticBridgeField
from scripts.research.evaluate_edit_bridge import run,audit_arithmetic
from scripts.research.audit_masked_angular import ReplayOracle,equal
from test_bounded_action_geometry import pair_record


def test_analytic_bridge_reversal_for_changed_molecular_graph():
    r=pair_record();field=AnalyticBridgeField();x=r['x'];torch.manual_seed(28201);p=center(torch.randn_like(x))
    field_args=(r['bonds'],r['new_bonds'],r['numbers'],r['electronic'],r['action'])
    torch.testing.assert_close(field(x,*field_args,.25),field(x,r['new_bonds'],r['bonds'],r['numbers'],r['electronic'],r['inverse_action'],.75),atol=1e-12,rtol=0)
    y,q,j=edit_bridge(x,p,*field_args,field,steps_per_side=2,drift_step=.02)
    xx,pp,jj=edit_bridge(y,q,r['new_bonds'],r['bonds'],r['numbers'],r['electronic'],r['inverse_action'],field,steps_per_side=2,drift_step=.02)
    torch.testing.assert_close(xx,x,atol=1e-10,rtol=0);torch.testing.assert_close(pp,p,atol=1e-10,rtol=0);torch.testing.assert_close(jj,-j)


def test_complete_molecular_caller_replays_costs_failures_and_all_MH_terms():
    r=pair_record();condition=dict(numbers=r['numbers'].tolist(),charge=0,spin_multiplicity=1)
    class Oracle:
        evaluated=0
        def evaluate_chunked(self,x,max_request):
            self.evaluated+=len(x);return .01*x.square().sum((1,2)),-.02*x
    class BadField(ZeroBridgeField):
        def forward(self,x,*args):return 100000*x
    def target(oracle):return ChemicalTarget(oracle,condition,.026,.1)
    warm=target(Oracle());states=warm.evaluate([warm.coordinate_state(r['x']) for _ in range(3)],phase='warm')
    sources=[(dict(parent=i),s) for i,s in enumerate(states)]
    models=dict(zero_bridge=ZeroBridgeField(),analytic_bridge=AnalyticBridgeField(),learned=EditBridgeField(hidden=8,radial=6,layers=1).double(),physical_arc=None,bad_bridge=BadField())
    protocol=dict(seed=28211,trials_per_source=3,method_order={'1':list(models)},bridge=dict(steps_per_side=1,kick_step=.2,drift_step=.005),
        arc_options=dict(radial_width=.05,max_segment_width=.05,margin=1e-8,chart_tolerance=1e-10),source_energy_tolerance_eV=1e-10,source_force_tolerance_eV_A=1e-10,inverse_checks_per_method_condition=2)
    physical=target(Oracle());saved,_=run(physical,models,sources,protocol,1)
    assert len(saved['attempts'])==45 and saved['initial_raw_queries']==6
    assert all(not row['scored'] and row['raw_cost']==0 for row in saved['attempts'] if row['method']=='bad_bridge')
    assert physical.oracle.evaluated==6+2*sum(row['scored'] for row in saved['attempts'])
    assert sum(row['scored'] for row in saved['attempts'])>0
    replay=ReplayOracle(saved['query_trace']);rt=target(replay);actual,_=run(rt,models,sources,protocol,1)
    equal(actual,saved);assert replay.index==len(replay.queries)
    check=audit_arithmetic(actual,rt,models,protocol)
    assert check['independent_MH_checks']==sum(row['scored'] for row in saved['attempts']) and check['trained_and_control_inverse_checks']>0
