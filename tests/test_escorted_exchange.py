import math
import torch
from cfm_mol.escorted_exchange import escorted_path,gaussian_path_logs,reverse_augmented_vertices
from cfm_mol.tempered_smc import DensityValue


def harmonic(x,**kwargs):return DensityValue(-.5*x.square().sum(-1),-x)


def test_augmented_path_involution_jacobian_and_probability_flow():
    torch.manual_seed(111)
    matrix=torch.tensor([[1.2,.3],[0.,.7]],dtype=torch.float64);inverse=torch.linalg.inv(matrix)
    logj=torch.linalg.slogdet(matrix)[1];k=2
    free=torch.randn(2*k+1,2,dtype=torch.float64)
    forward=lambda v:reverse_augmented_vertices(v,k,lambda x:x@matrix.T)
    reverse=lambda v:reverse_augmented_vertices(v,k,lambda x:x@inverse.T)
    torch.testing.assert_close(reverse(forward(free)),free,atol=1e-12,rtol=0)
    jac=torch.autograd.functional.jacobian(lambda v:forward(v.reshape(5,2)).flatten(),free.flatten())
    torch.testing.assert_close(torch.linalg.slogdet(jac)[1],logj,atol=1e-12,rtol=0)
    def path_ratio(v,a):
        states=list(v.unbind(0));states.insert(k+1,v[k]@a.T)
        lp=torch.zeros((),dtype=v.dtype)
        for i in range(len(states)-1):
            if i==k:continue
            x,y=states[i][None],states[i+1][None]
            f,r,_,_=gaussian_path_logs(x,y,-x,-y,std=.3,max_score_norm=2.)
            lp+=(r-f)[0]
        return -.5*(states[-1].square().sum()-states[0].square().sum())+lp+torch.linalg.slogdet(a)[1]
    r=path_ratio(free,matrix);rr=path_ratio(forward(free),inverse)
    torch.testing.assert_close(rr,-r,atol=1e-11,rtol=0)


def test_zero_path_reduces_to_map_and_saved_gaussians_reverse():
    x=torch.tensor([[-.7,.5],[.4,1.2]],dtype=torch.float64);g=torch.Generator().manual_seed(112)
    mapping=lambda v:(1.3*v,torch.full((len(v),),2*math.log(1.3),dtype=v.dtype))
    y,v,p=escorted_path(x,harmonic,mapping,steps_per_side=0,std=.2,max_score_norm=3.,generator=g)
    assert p['evaluated_vertices']==2 and len(p['edges'])==0
    torch.testing.assert_close(p['smooth_log_acceptance_ratio'],harmonic(y).log_value-harmonic(x).log_value+p['log_volume'])
    y,v,p=escorted_path(x,harmonic,mapping,steps_per_side=3,std=.2,max_score_norm=3.,generator=g,initial_value=harmonic(x))
    assert p['evaluated_vertices']==7 and len(p['edges'])==6
    total=torch.zeros(len(x),dtype=x.dtype)
    for edge in p['edges']:
        i,j=edge['source'],edge['destination'];u,w=p['vertices'][i],p['vertices'][j]
        # Independent direct Gaussian expression, including the normalizer.
        f=-.5*((w-edge['forward_mean'])/.2).square().sum(-1)-2*math.log(.2*math.sqrt(2*math.pi))
        r=-.5*((u-edge['reverse_mean'])/.2).square().sum(-1)-2*math.log(.2*math.sqrt(2*math.pi))
        torch.testing.assert_close(f,edge['log_forward']);torch.testing.assert_close(r,edge['log_reverse'])
        torch.testing.assert_close(w,edge['forward_mean']+.2*edge['noise'])
        total+=r-f
    torch.testing.assert_close(total,p['path_log_ratio'])


def test_corrected_path_preserves_known_half_normal_despite_outside_intermediates():
    g=torch.Generator().manual_seed(113)
    x=torch.randn(8192,1,dtype=torch.float64,generator=g).abs()
    invalid=0;outside_intermediate=0
    for _ in range(16):
        forward_scale=torch.full((len(x),),1.4,dtype=x.dtype)
        factors=torch.where(torch.rand(len(x),generator=g)<.5,forward_scale,forward_scale.reciprocal())
        def mapping(v):return v*factors[:,None],factors.log()
        y,_,path=escorted_path(x,harmonic,mapping,steps_per_side=2,std=.35,max_score_norm=3.,generator=g,initial_value=harmonic(x))
        valid=y[:,0]>0;invalid+=int((~valid).sum())
        outside_intermediate+=int((path['vertices'][1:-1,:,0]<0).any(0).sum())
        take=valid&(torch.rand(len(x),dtype=x.dtype,generator=g).log()<path['smooth_log_acceptance_ratio'].clamp_max(0))
        x=torch.where(take[:,None],y,x)
    assert invalid>0 and outside_intermediate>0
    assert abs(float(x.mean())-math.sqrt(2/math.pi))<.03
    assert abs(float(x.square().mean())-1)<.05
