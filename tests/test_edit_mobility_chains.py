import torch
from cfm_mol.chemical_sampler import ChemicalTarget
from cfm_mol.edit_mobility import MobilizedEditField
from cfm_mol.edit_bridge_sampler import ZeroBridgeField,RootZeroBridgeField
from scripts.research.evaluate_edit_mobility_chains import run,audit_chains
from scripts.research.audit_masked_angular import ReplayOracle,equal
from test_bounded_action_geometry import pair_record


def test_strict_budget_mixed_chains_replay_states_costs_and_kernel_ratios():
    r=pair_record();condition=dict(numbers=r['numbers'].tolist(),charge=0,spin_multiplicity=1)
    class Oracle:
        evaluated=0
        def evaluate_chunked(self,x,max_request):
            self.evaluated+=len(x);return .01*x.square().sum((1,2)),-.02*x
    def target(oracle):return ChemicalTarget(oracle,condition,.026,.1)
    warm=target(Oracle());states=warm.evaluate([warm.coordinate_state(r['x']) for _ in range(3)],phase='warm');sources=[(dict(parent=i),s) for i,s in enumerate(states)]
    model=MobilizedEditField(dict(hidden=8,radial=6,layers=1,force_scale=40.,edit_conditioned=True,roots_only=False),dict(variant='collective',hidden=8,rank=2)).double()
    models=dict(physical_arc=None,mobility_collective_s0=model,bare_edit=ZeroBridgeField(),root_noise=RootZeroBridgeField())
    protocol=dict(arm_order={'1':[dict(method=m,replica=0) for m in ['physical_arc','collective','bare_edit','root_noise']]},query_cap_per_parent=20,maximum_microsteps=128,
        evaluation_seeds=[28401,28402],schedule=['local','force_rotation','joint','local'],local_scales=[.1,.03,.01],
        bridge=dict(steps_per_side=1,kick_step=.2,drift_step=.005),bridge_by_method=dict(bare_edit=dict(steps_per_side=0),root_noise=dict(drift_step=.003)),arc_options=dict(radial_width=.05,max_segment_width=.05,margin=1e-8,chart_tolerance=1e-10),source_energy_tolerance_eV=1e-10,inverse_checks_per_method_condition=2)
    physical=target(Oracle());saved,_=run(physical,models,sources,protocol,1)
    assert all(c['queries_per_parent']==[20,20,20] and all(c['cap_reached']) for c in saved['chains'])
    assert physical.oracle.evaluated==240
    replay=ReplayOracle(saved['query_trace']);rt=target(replay);actual,_=run(rt,models,sources,protocol,1);equal(actual,saved)
    assert replay.index==len(replay.queries)
    checks=audit_chains(actual,rt,models,protocol)
    assert checks['independent_local_MH_checks']+checks['independent_rotation_MH_checks']+checks['independent_joint_MH_checks']==108
