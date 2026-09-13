import torch
from cfm_mol.chemical_sampler import ChemicalTarget
from cfm_mol.chemical_work import LinearBondWork
from cfm_mol.interaction_work_model import InteractionWorkModel
from cfm_mol.work_chain_kernel import WorkChainKernel
from scripts.research.evaluate_work_chains import run,audit,readouts
from scripts.research.audit_masked_angular import ReplayOracle,equal
from test_bounded_action_geometry import pair_record


def test_normalized_work_chains_budget_history_and_full_replay():
    row=pair_record();condition=dict(numbers=row['numbers'].tolist(),charge=0,spin_multiplicity=1)
    class Oracle:
        evaluated=0
        def evaluate_chunked(self,x,max_request):
            self.evaluated+=len(x);return .01*x.square().sum((1,2)),-.02*x
    def target(oracle):return ChemicalTarget(oracle,condition,.026,.1)
    warm=target(Oracle());initial=warm.evaluate([warm.coordinate_state(row['x']) for _ in range(3)],phase='warm');sources=[(dict(parent=i),s) for i,s in enumerate(initial)]
    elements=sorted(set(condition['numbers']));base=LinearBondWork(elements).double()
    interaction=InteractionWorkModel(elements,hidden=8,radial=6,linear=True).double()
    torch.manual_seed(28801)
    with torch.no_grad():base.coefficients.normal_(0,.05);interaction.coefficients.normal_(0,.1)
    options=dict(steps_per_side=2,kick_step=.2,drift_step=.01)
    kernels={'root_noise_s0':WorkChainKernel('root_noise',root_noise_options=options),
        'single_linear_s0':WorkChainKernel('single',base),
        'panel_radial_s0':WorkChainKernel('panel',base,interaction,use_affinity=True,panel_size=4)}
    protocol=dict(arm_order={'1':[dict(method=m,replica=0) for m in ['root_noise','single_linear','panel_radial']]},query_cap_per_parent=24,maximum_microsteps=128,
        evaluation_seeds=[28801,28802],schedule=['local','force_rotation','joint','local'],local_scales=[.1,.03,.01],
        bridge=options,bridge_by_method=dict(root_noise=options),source_energy_tolerance_eV=1e-10,inverse_checks_per_method_condition=2,
        policy=dict(uniform_fraction=.1),readouts=[16,24],kT_eV=.026)
    actual_target=target(Oracle());data,_,_=run(actual_target,kernels,sources,protocol,1)
    assert actual_target.oracle.evaluated==216 and all(c['queries_per_parent']==[24]*3 for c in data['chains'])
    replay=ReplayOracle(data['query_trace']);rt=target(replay);again,_,_=run(rt,kernels,sources,protocol,1)
    equal(data,again);assert replay.index==len(replay.queries)
    checks=audit(data,rt,protocol);assert checks['independent_normalized_joint_MH_checks']>0 and checks['independent_catalogue_normalizations']>0
    reads=readouts(data,protocol,condition['numbers'])
    assert all(r['reached'] and r['distinct_connectivities']>=1 for c in reads for r in c['readouts']['24'])
