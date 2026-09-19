import torch
import pytest
from cfm_mol.trajectory_physical_teacher import candidates,targets,velocity_target


def test_linear_energy_work_is_constant_and_antithetic_moment_is_force_shift():
    rng=torch.Generator().manual_seed(2)
    anchor=torch.randn(2,5,3,dtype=torch.float64,generator=rng);anchor-=anchor.mean(1,keepdim=True)
    force=torch.randn(2,5,3,dtype=torch.float64,generator=rng);force-=force.mean(1,keepdim=True)
    source,proposal,sigma,shift=candidates(anchor,force,kT=.025,particles=8,max_sigma=.03,max_shift=.1,seed=9)
    torch.testing.assert_close(proposal.mean(1)-anchor,shift,atol=1e-14,rtol=0)
    torch.testing.assert_close((source-anchor[:,None]).mean(-2),torch.zeros(2,8,3,dtype=anchor.dtype),atol=1e-14,rtol=0)
    energy=-(force*anchor).sum((-1,-2));proposal_energy=-(force[:,None]*proposal).sum((-1,-2))
    result=targets(anchor,source,proposal,sigma,shift,energy,proposal_energy,kT=.025,radius=1.)
    torch.testing.assert_close(result['weights'],torch.full((2,8),1/8,dtype=anchor.dtype),atol=1e-12,rtol=0)
    torch.testing.assert_close(result['work_shift'],shift,atol=1e-12,rtol=0)
    assert (shift.norm(dim=(-1,-2))<=.1+1e-12).all()


def test_quadratic_tilt_mean_and_empty_support_fallback():
    anchor=torch.tensor([[[.2,0.,0.],[-.2,0.,0.]]],dtype=torch.float64);kT=.5;spring=2.;force=-spring*anchor
    source,proposal,sigma,shift=candidates(anchor,force,kT=kT,particles=40000,max_sigma=.1,max_shift=.1,seed=52)
    energy=.5*spring*anchor.square().sum((-1,-2));ep=.5*spring*proposal.square().sum((-1,-2))
    result=targets(anchor,source,proposal,sigma,shift,energy,ep,kT=kT,radius=10.)
    expected=anchor/(1+spring*sigma[:,None,None].square()/kT)-anchor
    torch.testing.assert_close(result['work_shift'],expected,atol=2e-4,rtol=0)
    bad=targets(anchor,source,proposal,sigma,shift,energy,ep,kT=kT,radius=1e-12)
    assert not bad['eligible'].any() and bad['weights'].count_nonzero()==0
    torch.testing.assert_close(bad['work_shift'],shift)


def test_velocity_cap_preserves_center_and_has_declared_bound():
    delta=torch.tensor([[[10.,2.,3.],[-4.,-5.,9.],[1.,7.,-2.]]],dtype=torch.float64);t=torch.tensor([.9],dtype=torch.float64)
    velocity,scale=velocity_target(delta,t,velocity_scale=2.,gate_power=2)
    assert scale.item()<1 and velocity.norm(dim=-1).max()<=2*.9**2+1e-12
    torch.testing.assert_close(velocity.sum(1),torch.zeros(1,3,dtype=delta.dtype),atol=1e-12,rtol=0)
    with pytest.raises(ValueError):velocity_target(delta,torch.ones(1),velocity_scale=2.,gate_power=2)


def test_recording_cannot_change_sampler_and_captures_actual_states():
    from types import SimpleNamespace
    from test_clamped_density import graph_batch
    from cfm_mol.clamped_density import sample_clamped_flow,center_by_graph
    class Field(torch.nn.Module):
        def forward(self,graph,t,**kwargs):return {'x':-.4*graph.ndata['x_t']}
    model=SimpleNamespace(vector_field=Field());g,nbi,uem=graph_batch((3,),dtype=torch.float64)
    x0=g.ndata['x_1_true'].clone();rows=[]
    def observe(step,x,v,t):
        rows.append((step,x.clone(),v.clone(),t.clone()));x.fill_(100);v.zero_();t.zero_()
    plain=sample_clamped_flow(model,g,nbi,uem,x0=x0,n_ode_steps=4,terminal_time=1.,parameterization='velocity')
    recorded=sample_clamped_flow(model,g,nbi,uem,x0=x0,n_ode_steps=4,terminal_time=1.,parameterization='velocity',midpoint_observer=observe)
    torch.testing.assert_close(recorded,plain,atol=0,rtol=0)
    torch.testing.assert_close(rows[0][1],.95*center_by_graph(x0,nbi,1),atol=1e-14,rtol=0)
    assert [r[0] for r in rows]==list(range(4))
    for step,x,v,t in rows:
        torch.testing.assert_close(v,-.4*x,atol=1e-14,rtol=0);assert t.item()==(step+.5)/4
