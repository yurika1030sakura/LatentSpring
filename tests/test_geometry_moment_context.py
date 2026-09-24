import json
from pathlib import Path
import torch
from cfm_mol import matched_egnn as base, connectivity_feedback as feedback
from cfm_mol.geometry_moment_context import GeometryMomentContext, local_moments


def test_second_moment_distinguishes_plane_and_tetrahedron_at_equal_radius():
    planar = torch.tensor([[0.,0.,0.],[1.,0.,0.],[-1.,0.,0.],[0.,1.,0.],[0.,-1.,0.]],dtype=torch.float64)
    tetra = torch.tensor([[0.,0.,0.],[1.,1.,1.],[1.,-1.,-1.],[-1.,1.,-1.],[-1.,-1.,1.]],dtype=torch.float64)
    tetra[1:] /= 3 ** .5
    x = torch.stack([planar, tetra]) * 1.4
    radii = torch.full((2,5), .76, dtype=x.dtype)
    values = local_moments(x, radii)
    torch.testing.assert_close(values[0][0,0],values[0][1,0],atol=1e-12,rtol=0)
    determinants = torch.linalg.det(values[6][:,0])
    assert abs(determinants[0]) < 1e-12 and determinants[1] > .03


def test_zero_initialization_and_learned_context_symmetries():
    torch.manual_seed(75101)
    module = GeometryMomentContext([1,6,8]).double()
    x = torch.randn(2,5,3,dtype=torch.float64)
    z = torch.tensor([[6,6,8,1,1]]*2)
    exact = (x[:,:,None]-x[:,None,:]).square().sum(-1).reshape(-1,1)
    torch.testing.assert_close(module(x,z),exact,atol=0,rtol=0)
    with torch.no_grad():module.network[-1].weight.normal_(std=.1)
    before = module(x,z).reshape(2,5,5)
    q,_ = torch.linalg.qr(torch.randn(3,3,dtype=x.dtype))
    permutation = torch.tensor([4,1,3,0,2])
    after = module(x[:,permutation]@q+5,z[:,permutation]).reshape(2,5,5)
    torch.testing.assert_close(after,before[:,permutation][:,:,permutation],atol=1e-10,rtol=1e-10)
    torch.testing.assert_close(before,before.transpose(1,2),atol=1e-12,rtol=1e-12)
    assert before.diagonal(dim1=1,dim2=2).eq(0).all()


def test_context_has_real_backbone_gradient_and_preserves_initial_predictions():
    torch.set_num_threads(1)
    spec=json.loads(Path('research/evidence/gaga_feedback_distance_s0_v1.json').read_text())
    model=base.initialize(spec,'cpu').eval().requires_grad_(False);feedback.install(model)
    source=base.HarmonicSource();old=feedback.GeometryContext(source,'distance')
    module=GeometryMomentContext(spec['atomic_numbers'])
    x=base.center(torch.randn(1,5,3));z=torch.tensor([[6,6,8,1,1]]);t=x.new_full((1,1),.75)
    original=feedback.prediction(model,x,t,z,spec,old)
    new=feedback.prediction(model,x,t,z,spec,module)
    torch.testing.assert_close(new,original,atol=0,rtol=0)
    (new-torch.randn_like(new)).square().mean().backward()
    grad=module.network[-1].weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0
    assert all(p.grad is None for p in model.parameters())


def test_coincident_atoms_do_not_break_context_or_parameter_gradients():
    module=GeometryMomentContext([1,6]).double()
    with torch.no_grad():module.network[-1].weight.normal_(std=.1)
    x=torch.zeros(1,4,3,dtype=torch.float64);z=torch.tensor([[6,1,1,1]])
    value=module(x,z);assert torch.isfinite(value).all()
    value.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None)


def test_radial_control_matches_parameter_count_and_identity_initialization():
    torch.manual_seed(75102)
    moment=GeometryMomentContext([1,6,8]);radial=GeometryMomentContext([1,6,8],mode='radial')
    radial.load_state_dict(moment.state_dict())
    x=torch.randn(2,5,3);z=torch.tensor([[6,6,8,1,1]]*2)
    assert sum(p.numel() for p in moment.parameters())==sum(p.numel() for p in radial.parameters())
    torch.testing.assert_close(moment(x,z),radial(x,z),atol=0,rtol=0)
    with torch.no_grad():
        moment.network[-1].weight.normal_(std=.1);radial.load_state_dict(moment.state_dict())
    assert not torch.equal(moment(x,z),radial(x,z))
    assert torch.isfinite(radial(x,z)).all()
