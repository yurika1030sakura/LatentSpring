import torch
from cfm_mol.physical_endpoint_relaxation import relax_endpoint


def harmonic():
    target=torch.tensor([[-.5,.3,.1],[.5,-.3,-.1]],dtype=torch.float64)
    k=torch.tensor([2.,8.,20.],dtype=torch.float64)
    def evaluate(x):
        d=x-target
        return float((d.square()*k).sum()/2),-d*k
    return target,evaluate


def test_relaxation_reaches_anisotropic_target_with_capped_centered_steps():
    target,evaluate=harmonic();x=torch.tensor([[-1.,-.2,.5],[1.,.2,-.5]],dtype=torch.float64)
    result=relax_endpoint(x,evaluate,lambda x:True,max_atom_step_A=.15,
        force_tolerance_eV_A=1e-4,energy_tolerance_eV=0.)
    assert result['converged']
    torch.testing.assert_close(result['final']['positions'],target,atol=1e-4,rtol=0)
    states=[result['initial']]+[result['queries'][i] for i in result['accepted_query_indices']]
    for previous,current in zip(states,states[1:]):
        assert current['energy_eV']<=previous['energy_eV']
        assert (current['positions']-previous['positions']).norm(dim=-1).max()<=.15000001
        assert current['positions'].mean(0).abs().max()<1e-12


def test_hard_support_rejections_do_not_query_or_drop_the_reference():
    _,evaluate=harmonic();x=torch.tensor([[-1.,0,0],[1.,0,0]],dtype=torch.float64)
    result=relax_endpoint(x,evaluate,lambda y:torch.equal(x,y),backtracking_steps=4)
    assert result['status']=='line_search_stalled' and result['evaluations']==1
    assert len(result['events'])==4 and not result['accepted_steps']
    assert torch.equal(result['final']['positions'],x)


def test_query_budget_includes_initial_state_and_keeps_unconverged_status():
    _,evaluate=harmonic();x=torch.tensor([[-1.,0,0],[1.,0,0]],dtype=torch.float64)
    result=relax_endpoint(x,evaluate,lambda y:True,max_evaluations=1)
    assert result['status']=='query_limit' and result['evaluations']==1
    assert not result['converged'] and torch.equal(result['final']['positions'],x)
