import math
import torch
from cfm_mol.angular_envelope import envelope_parameters,logcosh
from cfm_mol.spherical_proposal import vmf_log_prob
from cfm_mol.masked_angular_guide import angular_log_score
from cfm_mol.masked_angular_sampler import capped_directions


def test_envelope_bound_and_accepted_density_ratio_for_linear_quadratic_scores():
    torch.manual_seed(135);n=4096
    eta=torch.randn(n,3,dtype=torch.float64)*10;a=torch.randn(n,3,3,dtype=eta.dtype)*10
    a=.5*(a+a.transpose(1,2));p=envelope_parameters(eta,a)
    u=torch.randn_like(eta);u=u/u.norm(dim=1,keepdim=True)
    rejection=torch.einsum('bi,bij,bj->b',u,a,u)-logcosh(p['gap']*(u*p['axis']).sum(1))-p['bound']
    assert float(rejection.max())<1e-10
    component=torch.stack([vmf_log_prob(u,p['parameters'][:,i]) for i in [0,1]],1)
    base=(p['log_weights']+component).logsumexp(1)
    target=angular_log_score(u,eta,a)
    torch.testing.assert_close(base+rejection-target,-p['log_base_partition']-p['bound'],atol=1e-10,rtol=0)
    q=torch.linalg.qr(torch.randn(3,3,dtype=eta.dtype))[0];q[:,0]*=-1
    rotated=envelope_parameters(eta@q,q.T@a@q)
    torch.testing.assert_close(rotated['bound'],p['bound'],atol=1e-10,rtol=0)
    torch.testing.assert_close(rotated['log_base_partition'],p['log_base_partition'],atol=1e-10,rtol=0)


def test_vector_case_has_no_score_rejection_and_sharp_tensor_envelope_is_effective():
    n=4096;eta=torch.tensor([0.,0.,60.],dtype=torch.float64).expand(n,-1);zero=torch.zeros(n,3,3,dtype=eta.dtype)
    g=torch.Generator().manual_seed(136);valid=lambda u,index:torch.ones(len(u),dtype=torch.bool)
    _,success,trace=capped_directions(eta,zero,torch.full((n,),60.,dtype=eta.dtype),valid,max_trials=1,generator=g,proposal='vmf_envelope')
    assert success.all() and trace[0]['score_passed'].all()
    a=torch.diag(torch.tensor([-20.,-20.,40.],dtype=eta.dtype))[None].expand(n,-1,-1)
    y,success,trace=capped_directions(torch.zeros_like(eta),a,torch.full((n,),40.,dtype=eta.dtype),valid,max_trials=16,generator=g,proposal='vmf_envelope')
    assert int(success.sum())>4080
    assert abs(float(y[success,2].mean()))<.04 and float(y[success,2].square().mean())>.95
    assert sum(len(t['indices']) for t in trace)<4*n
