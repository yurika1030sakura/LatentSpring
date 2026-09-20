import json
from pathlib import Path
import torch
from cfm_mol import matched_egnn as base
from cfm_mol.hydrogen_completion import detached_hydrogens,training_example,loss,complete


def specification():
    s=json.loads(Path('research/evidence/matched_generators_gaga_s0_v1.json').read_text())
    s['upstream_args']['nf']=32;s['upstream_args']['n_layers']=2
    return dict(network_spec=s,noise_std_A=.4)


def test_training_paths_fix_heavy_relative_geometry_and_preserve_hydrogen_set():
    clean=base.center(torch.randn(3,8,3,dtype=torch.float64));z=torch.tensor([[6,6,8,1,1,1,1,1]]*3)
    x,t,v=training_example(clean,z,seed=63111)
    torch.testing.assert_close(x[:,:3]-x[:,:1],clean[:,:3]-clean[:,:1],atol=1e-12,rtol=0)
    assert torch.equal(v[:,:3],torch.zeros_like(v[:,:3]))
    # At t=1 the implied endpoint has exactly the original unlabeled H set,
    # while preserving the heavy anchor frame (up to one common translation).
    endpoint=x+(1-t[...,None])*v
    for b in range(3):
        align=clean[b,0]-endpoint[b,0];dist=torch.cdist(endpoint[b,3:]+align,clean[b,3:])
        assert (dist.min(0).values<1e-7).all() and (dist.min(1).values<1e-7).all()


def test_completion_gradient_scope_and_equivariance_with_an_actual_egnn():
    torch.set_num_threads(2);spec=specification();model=base.initialize(spec['network_spec'],'cpu').double()
    z=torch.tensor([[6,8,1,1,1]]*2)
    x=torch.tensor([[[0.,0.,0.],[1.4,0.,0.],[-1.,0.,0.],[0.,1.,0.],[1.4,1.,0.]],
                    [[0.,0.,0.],[1.4,0.,0.],[-2.1,0.,0.],[0.,1.,0.],[1.4,1.,0.]]],dtype=torch.float64)
    assert detached_hydrogens(x,z).sum().item()==1
    objective=loss(model,base.center(x),z,spec,seed=63112);objective.backward()
    assert torch.isfinite(objective) and any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())
    q=torch.linalg.qr(torch.randn(3,3,dtype=x.dtype))[0];permutation=torch.tensor([4,2,0,3,1]);shift=torch.tensor([2.,3.,1.],dtype=x.dtype)
    y,info=complete(model,x,z,spec,steps=4);yr,ir=complete(model,x[:,permutation]@q+shift,z[:,permutation],spec,steps=4)
    assert info['network_calls']==ir['network_calls']==8 and info['network_example_calls']==8
    torch.testing.assert_close(y[0],x[0],atol=0,rtol=0)
    torch.testing.assert_close(yr[0],x[0,permutation]@q+shift,atol=0,rtol=0)
    torch.testing.assert_close(yr[1],y[1,permutation]@q+shift,atol=1e-8,rtol=1e-7)
    torch.testing.assert_close(y.mean(1),x.mean(1),atol=1e-12,rtol=0)
    torch.testing.assert_close(y[1,1]-y[1,0],x[1,1]-x[1,0],atol=1e-12,rtol=0)
