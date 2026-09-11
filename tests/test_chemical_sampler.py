import torch
from cfm_mol.chemical_sampler import ChemicalTarget,policy_log_probabilities
from cfm_mol.chemical_policy import ChemicalMovePolicy


class HarmonicOracle:
    evaluated=0
    def evaluate_chunked(self,x,max_request):
        self.evaluated+=len(x)
        return .5*x.square().sum((1,2)),-x


def test_trace_reconstructs_raw_target_and_policy_corrected_acceptance():
    target=ChemicalTarget(HarmonicOracle(),dict(numbers=[6,1,1,1,9],charge=0,spin_multiplicity=1),.5,.1)
    x=torch.tensor([[0.,0.,0.],[1.,1.,1.],[1.,-1.,-1.],[-1.,1.,-1.],[-1.,-1.,1.]],dtype=torch.float64)
    x[1:4]*=1.09/3**.5;x[4]*=1.35/3**.5;x-=x.mean(0)
    states=target.evaluate([target.coordinate_state(x)],phase='initial')
    assert len(states[0]['actions'])==3
    policy=ChemicalMovePolicy().double()
    with torch.no_grad():
        policy.family_head.weight.normal_();policy.action_head[-1].weight.normal_()
    rng=torch.Generator().manual_seed(31)
    for _ in range(12):
        old=states[0]
        states,records=target.transition(states,policy=policy,generator=rng,proposal_std=.02,phase='test')
        row=records[0]
        if row['valid']:
            new=target.states[row['new_state_id']]
            forward=policy_log_probabilities([old],target.numbers,target.kT,policy)[0,row['action_index']]
            reverse=policy_log_probabilities([new],target.numbers,target.kT,policy)[0,row['reverse_action_index']]
            torch.testing.assert_close(torch.tensor(row['log_acceptance_ratio'],dtype=x.dtype),row['base_log_ratio']+reverse-forward)
        assert row['accepted']==(row['valid'] and row['log_uniform']<min(0,row['log_acceptance_ratio']))
        assert states[0]['state_id']==(row['new_state_id'] if row['accepted'] else row['old_state_id'])
    assert target.oracle.evaluated==sum(2*len(q['positions']) for q in target.query_trace)
    for q in target.query_trace:
        torch.testing.assert_close(q['raw_force_eV_A'],-q['positions'])
        torch.testing.assert_close(q['inverted_force_eV_A'],q['positions'])
        torch.testing.assert_close(q['raw_energy_eV'],.5*q['positions'].square().sum((1,2)))
