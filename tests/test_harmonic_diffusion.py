import json
from pathlib import Path
import numpy as np
import pytest
import torch
from cfm_mol import harmonic_diffusion as hd,matched_egnn as base,connectivity_feedback as feedback
from cfm_mol.physical_connection import PhysicalConnection
from cfm_mol.matched_physical_connection import PhysicalFieldTransform


def model_for(kind):
    torch.set_num_threads(2)
    spec=json.loads(Path(f'research/evidence/matched_generators_{kind}_s0_v1.json').read_text())
    model=feedback.install(base.initialize(spec,'cpu')).eval()
    return model,spec


def test_exact_existing_harmonic_source_and_tree_precision():
    z=[6,6,8,1,1];noise=hd.HarmonicNoise();old=base.HarmonicSource()
    factor,edges,std=noise.tree(z,np.random.default_rng(91001))
    rng=np.random.default_rng(91001);factor,edges,std=noise.tree(z,rng)
    x=factor@rng.normal(size=(len(z)-1,3))
    np.testing.assert_allclose(x,old.sample(z,np.random.default_rng(91001)).numpy(),atol=1e-12,rtol=1e-12)
    incidence=np.zeros((len(z)-1,len(z)))
    for k,(i,j) in enumerate(edges):incidence[k,i]=1/std[k];incidence[k,j]=-1/std[k]
    covariance=factor@factor.T;precision=incidence.T@incidence
    np.testing.assert_allclose(covariance@precision,np.eye(len(z))-np.ones((len(z),len(z)))/len(z),atol=1e-12,rtol=1e-12)
    c=noise.batch([z],91001,device='cpu',dtype=torch.float64)
    error=torch.randn((1,len(z),3),dtype=torch.float64)
    expected=torch.einsum('bik,ij,bjk->',error,torch.from_numpy(precision),error)/(3*(len(z)-1))
    torch.testing.assert_close(hd.mahalanobis_loss(error,c),expected,atol=1e-12,rtol=1e-12)


@pytest.mark.parametrize('kind',['edm','gaga'])
def test_initial_covariance_matches_declared_diffusion_prior(kind):
    model,spec=model_for(kind);model.double();noise=hd.HarmonicNoise();n=4
    one=noise.batch([[6,8,1,1]],92001,device='cpu',dtype=torch.float64)
    component={k:v.expand((12000,)+v.shape[1:]) if k!='context' else v for k,v in one.items()}
    x=hd.initial_state(model,spec,component,torch.Generator().manual_seed(92002))
    covariance=torch.einsum('bik,bjk->ij',x,x)/(len(x)*3)
    expected=one['covariance'][0]
    if kind=='gaga':
        t=x.new_tensor([[spec['gaga_max_t']/model.T]]);gamma=model.gamma(t)
        expected=model.alpha(gamma,x)[0,0,0]**2*spec['data_variance_per_dof']*(torch.eye(n)-torch.ones(n,n)/n)+model.sigma(gamma,x)[0,0,0]**2*expected
    torch.testing.assert_close(covariance,expected,atol=.03,rtol=.06)
    assert x.mean(1).abs().max()<1e-12


@pytest.mark.parametrize('kind',['edm','gaga'])
def test_conditioned_prediction_symmetry_and_trainable_loss(kind):
    model,spec=model_for(kind);noise=hd.HarmonicNoise();z=torch.tensor([[6,6,8,1,1]])
    x=base.center(torch.randn(1,5,3));t=torch.tensor([[.31]])
    c=noise.batch(z.tolist(),93001,device='cpu',dtype=torch.float32)
    y=hd.prediction(model,x,t,z,spec,c)
    permutation=torch.tensor([3,2,0,4,1]);rotation=torch.tensor([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]])
    c2=dict(c,context=c['context'].reshape(1,5,5)[:,permutation][:,:,permutation].reshape(-1,1))
    transformed=hd.prediction(model,x[:,permutation]@rotation,t,z[:,permutation],spec,c2)
    torch.testing.assert_close(transformed,y[:,permutation]@rotation,atol=3e-6,rtol=3e-5)
    objective=hd.loss(model,x,z,spec,noise,93002);objective.backward()
    gradients=[p.grad for p in model.parameters() if p.grad is not None]
    assert torch.isfinite(objective) and gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(g.abs().sum() for g in gradients)>0


@pytest.mark.parametrize('kind',['edm','gaga'])
def test_zero_head_preserves_structured_sampler_and_tree(kind):
    model,spec=model_for(kind);noise=hd.HarmonicNoise();head=PhysicalConnection(spec['atomic_numbers'],velocity_scale=2.)
    before=base.state_hash(model);transform=PhysicalFieldTransform(model,spec,head,4.,strength_limit=4.)
    plain=hd.sample(model,[6,6,8,1,1],spec,noise,94001,2,8)
    corrected=hd.sample(model,[6,6,8,1,1],spec,noise,94001,2,8,field_transform=transform)
    for x,y in zip(plain[:2],corrected[:2]):torch.testing.assert_close(x,y,atol=0,rtol=0)
    for key in plain[2]:torch.testing.assert_close(plain[2][key],corrected[2][key],atol=0,rtol=0)
    assert before==base.state_hash(model) and transform.calls==8
    assert torch.isfinite(corrected[0]).all()


def test_colored_reverse_posterior_matches_gaussian_conditioning():
    model,spec=model_for('edm');model.double();noise=hd.HarmonicNoise()
    c=noise.batch([[6,6,8,1,1]],95001,device='cpu',dtype=torch.float64)
    x0=base.center(torch.randn(1,5,3,dtype=torch.float64));rng=torch.Generator().manual_seed(95002)
    s,t=x0.new_tensor([[.3]]),x0.new_tensor([[.7]])
    gs,gt=model.gamma(s),model.gamma(t);at=model.alpha(gt,x0);as_=model.alpha(gs,x0)
    st,ss=model.sigma(gt,x0),model.sigma(gs,x0)
    eta=hd.draw_noise(c,rng);xt=at*x0+st*eta
    variance,std,ratio=model.sigma_and_alpha_t_given_s(gt,gs,x0)
    actual=xt/ratio-variance/(ratio*st)*eta
    expected=as_*x0+ratio*ss.square()/st.square()*(xt-at*x0)
    torch.testing.assert_close(actual,expected,atol=1e-9,rtol=1e-8)
    torch.testing.assert_close((std*ss/st).square(),ss.square()-ratio.square()*ss.pow(4)/st.square(),atol=1e-9,rtol=1e-8)
