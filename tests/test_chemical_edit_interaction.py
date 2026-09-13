import torch
import pytest
from cfm_mol.chemical_moves import exchange_terminal_sites, covalent_radii
from cfm_mol.chemical_edit_interaction import four_positions, mixed_difference, restraint_interaction


def example():
    torch.manual_seed(28701)
    x=torch.randn(8,3,dtype=torch.float64);x-=x.mean(0)
    radii=covalent_radii([1,17,1,9,6,7,8,16])
    return x,radii,(0,1,4,5),(2,3,6,7)


def test_commutation_inverse_intrinsic_jacobian():
    x,r,a,b=example();corners,j,(ia,ib)=four_positions(x,r,a,b)
    undo,jrev,_=four_positions(corners[3],r,ia,ib)
    torch.testing.assert_close(undo[3],x,atol=1e-12,rtol=0)
    torch.testing.assert_close(j+jrev,torch.tensor(0.,dtype=x.dtype),atol=1e-12,rtol=0)
    basis=torch.linalg.qr(torch.cat([torch.eye(7,dtype=x.dtype),-torch.ones(1,7,dtype=x.dtype)]),mode='reduced')[0]
    def transform(z):
        y,_,_=exchange_terminal_sites(basis@z.reshape(7,3),r,a)
        y,_,_=exchange_terminal_sites(y,r,b)
        return (basis.T@y).flatten()
    jac=torch.autograd.functional.jacobian(transform,(basis.T@x).flatten())
    torch.testing.assert_close(torch.linalg.slogdet(jac)[1],j,atol=1e-10,rtol=0)


def test_interaction_sign_and_exact_restraint_cross_term():
    x,r,a,b=example();corners,_,(ia,ib)=four_positions(x,r,a,b)
    # A coupled physical toy; an additive single-edit oracle is insufficient.
    energy=lambda p: .7*(p[0]-p[2]).square().sum()+.3*(p[1]-p[3]).square().sum()
    val=mixed_difference(torch.stack([energy(p) for p in corners]))
    reverse_a,_,_=four_positions(corners[1],r,ia,b)
    reverse_b,_,_=four_positions(corners[2],r,a,ib)
    for square in [reverse_a,reverse_b]:
        torch.testing.assert_close(mixed_difference(torch.stack([energy(p) for p in square])),-val)
    expected=.1*((corners[1]-x)*(corners[2]-x)).sum()
    torch.testing.assert_close(restraint_interaction(corners),expected,atol=1e-12,rtol=0)
    assert abs(float(val))>.01


def test_reject_overlapping_edits():
    x,r,a,b=example()
    with pytest.raises(ValueError):four_positions(x,r,a,(0,3,6,7))


@pytest.mark.parametrize('kind',['linear','blind','environment'])
def test_interaction_representation_four_state_symmetries(kind):
    from cfm_mol.chemical_edit_interaction import four_graphs
    from cfm_mol.interaction_work_model import InteractionWorkModel
    x,r,a,b=example();corners,_,(ia,ib)=four_positions(x,r,a,b)
    numbers=torch.tensor([1,17,1,9,6,7,8,16]);electronic=torch.tensor([[0.,1.,.026]],dtype=x.dtype)
    graph=torch.zeros(8,8,dtype=x.dtype)
    for leaf,anchor in [(0,4),(1,5),(2,6),(3,7),(4,5),(5,6),(6,7)]:graph[leaf,anchor]=graph[anchor,leaf]=1.
    graphs=four_graphs(graph,a,b);actions=torch.tensor([[a,b]])
    m=InteractionWorkModel(sorted(set(numbers.tolist())),hidden=8,radial=6,linear=kind=='linear',environment=kind=='environment').double()
    with torch.no_grad():
        if kind=='linear':m.coefficients.normal_()
        else:m.coefficient_head[-1].weight.normal_(0,.2)
    value=m(corners[None],graphs[None],numbers,electronic,actions)
    for order,aa in [([1,0,3,2],[ia,b]),([2,3,0,1],[a,ib])]:
        torch.testing.assert_close(m(corners[order][None],graphs[order][None],numbers,electronic,torch.tensor([aa])),-value,atol=1e-10,rtol=0)
    torch.testing.assert_close(m(corners[[0,2,1,3]][None],graphs[[0,2,1,3]][None],numbers,electronic,actions.flip(1)),value,atol=1e-10,rtol=0)
    q=torch.linalg.qr(torch.randn(3,3,dtype=x.dtype))[0]
    torch.testing.assert_close(m((corners@q+3)[None],graphs[None],numbers,electronic,actions),value,atol=1e-10,rtol=0)
    perm=torch.tensor([4,2,7,0,6,3,5,1]);inv=perm.argsort()
    torch.testing.assert_close(m(corners[:,perm][None],graphs[:,perm][:,:,perm][None],numbers[perm],electronic,inv[actions]),value,atol=1e-10,rtol=0)
    value.sum().backward();assert any(p.grad is not None and bool((p.grad!=0).any()) for p in m.parameters())
