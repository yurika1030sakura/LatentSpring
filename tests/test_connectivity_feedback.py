import copy
import json
from pathlib import Path
import torch

from cfm_mol import matched_egnn as base
from cfm_mol.connectivity_feedback import install,GeometryContext,tree_marginals,prediction,use_context


def test_tree_marginals_match_enumerated_three_node_law():
    w=torch.tensor([[[0.,2.,3.],[2.,0.,5.],[3.,5.,0.]]],dtype=torch.float64)
    p=tree_marginals(w)[0]
    # Three trees have weights6,10,15; each edge appears in two trees.
    expected=torch.tensor([[0.,16/31,21/31],[16/31,0.,25/31],[21/31,25/31,0.]],dtype=torch.float64)
    torch.testing.assert_close(p,expected,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(p.sum()/2,torch.tensor(2.,dtype=p.dtype))


def test_distance_context_preserves_original_egnn_and_ema_copy():
    spec=json.loads(Path('research/evidence/matched_generators_harmonic_fm_s0_v1.json').read_text())
    model=install(base.initialize(spec,'cpu'));source=base.HarmonicSource();context=GeometryContext(source,'distance')
    x=base.center(torch.randn(2,5,3));z=torch.tensor([[6,6,8,1,1]]*2);t=torch.tensor([[.3],[.7]])
    old=base.vector(model,x,t,z,spec)
    with use_context(model,context(x,z)):new=base.vector(model,x,t,z,spec)
    torch.testing.assert_close(old,new,atol=0,rtol=0)
    ema=copy.deepcopy(model);changed=context(x+.1*torch.randn_like(x),z)
    with use_context(ema,changed):other=base.vector(ema,x,t,z,spec)
    assert not torch.equal(other,old)
    torch.testing.assert_close(base.vector(model,x,t,z,spec),old,atol=0,rtol=0)
    assert all(getattr(b,'_geometry_context',None) is None for b in model.dynamics.egnn.modules())


def test_tree_feedback_velocity_is_rotation_and_permutation_equivariant():
    torch.manual_seed(42991)
    spec=json.loads(Path('research/evidence/matched_generators_harmonic_fm_s0_v1.json').read_text())
    model=install(base.initialize(spec,'cpu'));source=base.HarmonicSource();context=GeometryContext(source,'tree')
    x=base.center(torch.randn(2,5,3));z=torch.tensor([[6,6,8,1,1]]*2);t=torch.tensor([[.3],[.7]])
    r=torch.linalg.qr(torch.randn(3,3))[0];perm=torch.tensor([4,2,0,3,1])
    y=prediction(model,x,t,z,spec,context)
    yr=prediction(model,x[:,perm]@r,t,z[:,perm],spec,context)
    torch.testing.assert_close(yr,y[:,perm]@r,atol=3e-6,rtol=3e-5)


def test_two_pass_objective_has_finite_gradients_and_sampler_counts_both_passes():
    from cfm_mol.connectivity_feedback import loss,sample
    spec=json.loads(Path('research/evidence/matched_generators_harmonic_fm_s0_v1.json').read_text())
    spec['two_pass']=True
    clean=base.center(torch.randn(2,5,3));z=torch.tensor([[6,6,8,1,1]]*2)
    source=base.HarmonicSource()
    for kind in ['distance','tree']:
        model=install(base.initialize(spec,'cpu'))
        context=GeometryContext(source,kind)
        objective=loss(model,clean,z,spec,source,context,43217)
        objective.backward()
        gradients=[p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.square().sum()) for g in gradients)>0
        calls=[]
        handle=model.dynamics.egnn.register_forward_hook(lambda *_:calls.append(1))
        result,_=sample(model,z[0].tolist(),spec,source,context,43219,2,128)
        handle.remove()
        assert len(calls)==128
        assert result.shape==(2,5,3) and torch.isfinite(result).all()
        torch.testing.assert_close(result.mean(1),torch.zeros(2,3),atol=1e-6,rtol=0)
