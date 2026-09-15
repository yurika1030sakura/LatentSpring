import torch
from cfm_mol.escorted_thermal_teacher import escorted_candidates,local_work_log_weights


def test_constant_force_escort_has_constant_complete_work():
    a=torch.zeros(2,3,3,dtype=torch.float64);f=torch.tensor([[[1.,0.,0.],[-1.,0.,0.],[0.,0.,0.]]],dtype=a.dtype).repeat(2,1,1)
    source,y,sigma,shift=escorted_candidates(a,f,kT=.3,particles=64,max_sigma=.2,max_shift=10.,seed=37101)
    energy=-(y*f[:,None]).sum((-1,-2));valid=torch.ones_like(energy,dtype=torch.bool)
    logw=local_work_log_weights(a,source,y,torch.zeros(2,dtype=a.dtype),energy,sigma,.3,valid)
    expected=.5*sigma.square()*f.square().sum((1,2))/.3**2
    torch.testing.assert_close(logw,expected[:,None].expand_as(logw),atol=1e-12,rtol=1e-12)
    incomplete=-energy/.3
    assert incomplete.std()>0.1
    valid[0,0]=False
    assert torch.isneginf(local_work_log_weights(a,source,y,torch.zeros(2,dtype=a.dtype),energy,sigma,.3,valid)[0,0])


def test_quadratic_normalizer_and_weighted_mean_match_analytic_local_target():
    a=torch.tensor([[[.3,0.,0.],[-.3,0.,0.],[0.,0.,0.]]],dtype=torch.float64);k=2.;kt=.5
    x,y,sigma,shift=escorted_candidates(a,-k*a,kT=kt,particles=100000,max_sigma=.2,max_shift=10.,seed=37102)
    ea=.5*k*a.square().sum((1,2));ey=.5*k*y.square().sum((-1,-2));logw=local_work_log_weights(a,x,y,ea,ey,sigma,kt,torch.ones_like(ey,dtype=torch.bool))
    r=1+k*sigma.square()/kt;dim=6
    exact_logz=-dim/2*r.log()+.5*(k/kt)**2*sigma.square()*a.square().sum((1,2))/r
    actual=torch.logsumexp(logw,-1)-torch.tensor(float(logw.shape[-1])).log()
    torch.testing.assert_close(actual,exact_logz,atol=.006,rtol=0)
    mean=(logw.softmax(-1)[:,:,None,None]*y).sum(1)
    torch.testing.assert_close(mean,a/r[:,None,None],atol=.002,rtol=0)
