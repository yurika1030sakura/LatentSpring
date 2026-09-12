import torch

from cfm_mol.conditional_angular_probe import angular_probes, angular_force, identify_parameter
from cfm_mol.local_site_guide import confinement_parameter
from cfm_mol.masked_angular_guide import masked_angular_context


def test_probe_context_is_fixed_and_recovers_energy_gradient_and_finite_work():
    rng=torch.Generator().manual_seed(25209)
    x=torch.randn(7,3,generator=rng,dtype=torch.float64);x-=x.mean(0)
    root=(0,1);eta=torch.tensor([12.,-29.,101.],dtype=torch.float64);kT=.026;gamma=.1
    probes,_=angular_probes(x,root,generator=rng)
    masked,radius,_=masked_angular_context(x[None],torch.tensor([root]))
    expected=eta+confinement_parameter(masked,radius,torch.tensor([kT],dtype=torch.float64),gamma)[0]
    def state(y):
        z=y.clone().requires_grad_();v=z[0]-z[1];u=v/v.norm();raw=-kT*(eta*u).sum()
        force=-torch.autograd.grad(raw,z)[0]
        return dict(positions=y,force_eV_A=force,potential_eV=raw.detach()+gamma/2*y.square().sum())
    center=state(x);fit=[center];checks=[]
    for row in probes:
        m,r,_=masked_angular_context(row['positions'][None],torch.tensor([root]))
        torch.testing.assert_close(m,masked,atol=1e-12,rtol=0)
        torch.testing.assert_close(r,radius,atol=1e-12,rtol=0)
        (fit if row['role']=='fit' else checks).append(state(row['positions']))
    values=[angular_force(s,root,kT,gamma) for s in fit]
    fitted=identify_parameter(torch.stack([v[0] for v in values]),torch.stack([v[1] for v in values]))
    assert fitted['full_rank']
    actual=torch.tensor(fitted['fitted_parameter'],dtype=torch.float64)
    torch.testing.assert_close(actual,expected,atol=1e-7,rtol=1e-9)
    u0=values[0][0]
    for s in checks:
        u,score=angular_force(s,root,kT,gamma)
        torch.testing.assert_close(actual-(actual*u).sum()*u,score,atol=1e-7,rtol=1e-9)
        torch.testing.assert_close(-actual@(u-u0),(s['potential_eV']-center['potential_eV'])/kT,atol=1e-7,rtol=1e-9)


def test_one_direction_does_not_identify_concentration():
    u=torch.tensor([[0.,0.,1.]],dtype=torch.float64).expand(5,-1)
    result=identify_parameter(u,torch.tensor([[2.,3.,0.]],dtype=torch.float64).expand_as(u))
    assert not result['full_rank'] and 'fitted_parameter' not in result


def test_local_fit_does_not_claim_an_arbitrary_angular_potential_is_vmf():
    x=torch.tensor([[1.,0.,0.],[0.,0.,0.],[-1.,1.,0.]],dtype=torch.float64);x-=x.mean(0)
    probes,_=angular_probes(x,(0,1),generator=torch.Generator().manual_seed(25219))
    def values(y):
        z=y.clone().requires_grad_();v=z[0]-z[1];u=v/v.norm()
        energy=-u[0]+20*u[1]**4+7*u[2]**2
        force=-torch.autograd.grad(energy,z)[0]
        return angular_force(dict(positions=y,force_eV_A=force),(0,1),1.,0.)
    fit=[values(x)]+[values(p['positions']) for p in probes if p['role']=='fit']
    result=identify_parameter(torch.stack([r[0] for r in fit]),torch.stack([r[1] for r in fit]))
    eta=torch.tensor(result['fitted_parameter'],dtype=torch.float64)
    errors=[]
    for p in probes:
        if p['role']=='check':
            u,s=values(p['positions']);errors.append(float((eta-(eta*u).sum()*u-s).square().sum()))
    assert max(errors)>.01
