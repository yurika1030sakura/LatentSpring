import torch
from cfm_mol.chemical_sampler import ChemicalTarget
from cfm_mol.edit_bridge_sampler import ZeroBridgeField,propose_edit
from cfm_mol.joint_chemical_geometry import distinct_anchor_actions
from scripts.research.chemical_work_catalogue import score
from scripts.research.audit_masked_angular import ReplayOracle,equal
from test_bounded_action_geometry import pair_record


def test_complete_action_catalogue_replays_work_and_failed_action_denominators():
    r=pair_record();condition=dict(numbers=r['numbers'].tolist(),charge=0,spin_multiplicity=1)
    class Oracle:
        evaluated=0
        def evaluate_chunked(self,x,max_request):
            self.evaluated+=len(x);return .01*x.square().sum((1,2)),-.02*x
    def target(oracle):return ChemicalTarget(oracle,condition,.026,.1)
    warm=target(Oracle());old=warm.evaluate([warm.coordinate_state(r['x'])],phase='source')[0];sources=[dict(source_id=0,state=old)]
    catalogue=[]
    for action in distinct_anchor_actions(condition['numbers'],old['graph']['bond_orders']):
        new,record=propose_edit(warm,old,action,torch.zeros_like(r['x']),0,0,method='bare_edit',field=ZeroBridgeField(),bridge_options=dict(steps_per_side=0,kick_step=.2,drift_step=.02),arc_options={})
        catalogue.append(dict(source_id=0,parent=0,action=action,candidate=new))
    physical=target(Oracle());saved=score(physical,sources,catalogue)
    assert len(saved['rows'])==len(catalogue) and physical.oracle.evaluated==2+2*sum(row['valid'] for row in saved['rows'])
    for row in saved['rows']:
        if not row['valid']:continue
        x=saved['states'][row['source_state_id']]['positions'];y=saved['states'][row['candidate_state_id']]['positions']
        assert abs(row['potential_change_eV']-float(.06*(y.square().sum()-x.square().sum())))<1e-12
    replay=ReplayOracle(saved['query_trace']);actual=score(target(replay),sources,catalogue);equal(actual,saved)
    assert replay.index==len(replay.queries)
