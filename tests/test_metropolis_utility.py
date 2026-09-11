import math
import torch
from cfm_mol.metropolis_utility import mh_utility_score_loss


def normal_log(x,mean,std):return -.5*((x-mean)/std).square()-std.log()-.5*math.log(2*math.pi)


def test_hard_support_scale_gradient_matches_boundary_derivative_and_baseline():
    log_scale=torch.tensor(math.log(.7),dtype=torch.float64,requires_grad=True)
    noise=torch.randn(262144,dtype=torch.float64,generator=torch.Generator().manual_seed(9991))
    y=(log_scale.exp()*noise).detach()
    logq=normal_log(y,0.,log_scale.exp())
    ratio=torch.where(y.abs()<1,torch.zeros_like(y),torch.full_like(y,-torch.inf))
    expected=2/.7*math.exp(-.5/(.7**2))/math.sqrt(2*math.pi)
    for baseline in [0.,.37]:
        loss=mh_utility_score_loss(logq,logq,ratio,torch.ones_like(y),baseline=baseline)
        gradient=torch.autograd.grad(loss,log_scale,retain_graph=True)[0]
        assert abs(float(gradient)-expected)<.006
    naive=-((log_scale.exp()*noise).abs()<1).double().mean()+0*log_scale
    assert float(torch.autograd.grad(naive,log_scale)[0])==0


def test_forward_reverse_score_selection_matches_asymmetric_exact_acceptance():
    drift=torch.tensor(.3,dtype=torch.float64,requires_grad=True);std=torch.tensor(.7,dtype=torch.float64)
    x=.4;noise=torch.randn(262144,dtype=torch.float64,generator=torch.Generator().manual_seed(9993))
    y=(x+drift+std*noise).detach()
    forward=normal_log(y,x+drift,std);reverse=normal_log(torch.full_like(y,x),y+drift,std)
    ratio=torch.where(y.abs()<1,reverse-forward,torch.full_like(y,-torch.inf))
    loss=mh_utility_score_loss(forward,reverse,ratio,torch.ones_like(y))
    gradient=float(torch.autograd.grad(loss,drift)[0])
    phi=lambda v:math.exp(-v*v/2)/math.sqrt(2*math.pi)
    expected=-(phi((1+x+.3)/.7)+phi((1-x+.3)/.7)-2*phi(.3/.7))/.7
    assert abs(gradient-expected)<.012
