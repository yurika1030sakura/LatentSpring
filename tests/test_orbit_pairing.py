import torch
from cfm_mol.orbit_pairing import haar_rotation, proper_alignment, orbit_pair, path_cost


def centered(x):
    return x-x.mean(-2, keepdim=True)


def test_proper_alignment_and_chirality():
    g=torch.Generator().manual_seed(331)
    x=centered(torch.randn(7,3,generator=g,dtype=torch.float64))
    rotation=haar_rotation(x,g)
    fitted=proper_alignment(x,x@rotation)
    torch.testing.assert_close(fitted,rotation,atol=1e-12,rtol=0)
    mirrored=x.clone();mirrored[:,0]*=-1
    constrained=proper_alignment(x,mirrored)
    assert torch.linalg.det(constrained)>.999999
    assert (x@constrained-mirrored).norm()>.1


def test_pairing_preserves_orbits_and_steric_selection_cost():
    g=torch.Generator().manual_seed(772)
    source=centered(torch.randn(12,3,generator=g,dtype=torch.float64))
    target=centered(torch.randn(12,3,generator=g,dtype=torch.float64))
    radii=torch.full((12,),.7,dtype=torch.float64)
    outputs={}
    for mode in ['independent','rotation','steric']:
        a,b,record=orbit_pair(source,target,radii,mode=mode,generator=torch.Generator().manual_seed(813))
        for old,new in [(source,a),(target,b)]:
            torch.testing.assert_close(old@old.T,new@new.T,atol=1e-11,rtol=0)
            torch.testing.assert_close(new.mean(0),torch.zeros(3,dtype=new.dtype),atol=1e-12,rtol=0)
        outputs[mode]=(a,b,record)
    torch.testing.assert_close(outputs['independent'][1],outputs['steric'][1],atol=0,rtol=0)
    assert outputs['rotation'][2]['displacement_per_atom']<=outputs['independent'][2]['displacement_per_atom']+1e-12
    assert outputs['steric'][2]['selected_cost']<=outputs['steric'][2]['standard_cost']+1e-12


def test_haar_restores_non_gaussian_orientation_selection():
    # Aligning a random first vector to +x is a deliberately severe gauge bias.
    # Haar augmentation must restore both first moments and intrinsic covariance.
    g=torch.Generator().manual_seed(791)
    n=3;draws=4096
    z=centered(torch.randn(draws,n,3,generator=g,dtype=torch.float64))
    target=centered(torch.tensor([[1.,0.,0.],[-.3,.8,0.],[-.7,-.8,0.]],dtype=torch.float64))
    rotations=torch.stack([proper_alignment(x,target) for x in z])
    aligned=z@rotations
    augmentation=torch.stack([haar_rotation(z,g) for _ in range(draws)])
    result=aligned@augmentation
    torch.testing.assert_close(result@result.transpose(-1,-2),z@z.transpose(-1,-2),atol=1e-11,rtol=0)
    assert result.mean(0).abs().max()<.05
    flat=result.reshape(draws,-1)
    covariance=flat.T@flat/draws
    expected=torch.kron(torch.eye(n,dtype=z.dtype)-torch.ones(n,n,dtype=z.dtype)/n,torch.eye(3,dtype=z.dtype))
    assert (covariance-expected).abs().max()<.06
    assert aligned.mean(0).abs().max()>.3


def test_one_atom_and_invalid_mode():
    import pytest
    x=torch.zeros(1,3,dtype=torch.float64)
    a,b,_=orbit_pair(x,x,torch.ones(1),mode='steric',generator=torch.Generator().manual_seed(1))
    assert not a.any() and not b.any()
    with pytest.raises(ValueError):
        orbit_pair(x,x,torch.ones(1),mode='unknown',generator=torch.Generator().manual_seed(1))


def test_coupled_fm_path_preserves_terminal_semantics_and_input_graph():
    from test_clamped_density import graph_batch, Schedule
    from cfm_mol.clamped_fm import clamped_fm_path
    graph,nbi,_=graph_batch();original=graph.ndata['x_1_true'].clone()
    for mode in ['independent','rotation','steric','typed_rotation']:
        xt,t,head,info=clamped_fm_path(graph,nbi,Schedule(2),terminal_time=1.,parameterization='displacement',
            generator=torch.Generator().manual_seed(18),pairing=mode,pairing_radii=torch.ones(len(original)),
            pairing_generator=torch.Generator().manual_seed(19))
        torch.testing.assert_close(head-xt,2*t[nbi,None]*(info['x1']-info['x0']),atol=1e-12,rtol=0)
        torch.testing.assert_close(graph.ndata['x_1_true'],original,atol=0,rtol=0)
        assert len(info['pairing_records'])==2


def test_typed_matching_preserves_conditioned_shape_and_reduces_displacement():
    from cfm_mol.orbit_pairing import typed_orbit_pair
    g=torch.Generator().manual_seed(923)
    source=centered(torch.randn(9,3,generator=g,dtype=torch.float64))
    target=centered(torch.randn(9,3,generator=g,dtype=torch.float64))
    groups=torch.tensor([0,0,0,0,1,1,1,2,2])
    radii=torch.tensor([.7]*4+[.4]*3+[1.]*2,dtype=source.dtype)
    a,b,r=typed_orbit_pair(source,target,radii,groups,generator=g)
    for old,new in [(source,a),(target,b)]:
        for group in groups.unique():
            idx=groups==group
            torch.testing.assert_close(old[idx].square().sum(-1).sort().values,new[idx].square().sum(-1).sort().values,atol=1e-11,rtol=0)
        torch.testing.assert_close(torch.pdist(old).sort().values,torch.pdist(new).sort().values,atol=1e-11,rtol=0)
    reference=source@proper_alignment(source,target)
    assert r['displacement_per_atom']<=float((reference-target).square().sum(-1).mean())+1e-12
