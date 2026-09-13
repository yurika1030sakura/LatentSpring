import torch
import pytest
from cfm_mol.edit_conditioned_bridge import EditBridgeField,edit_bridge,augmented_log_ratio,center
from cfm_mol.nonequilibrium import centered_orthonormal_basis
from test_bounded_action_geometry import pair_record


def example(roots_only=False):
    row=pair_record();torch.manual_seed(28101);model=EditBridgeField(hidden=8,radial=6,layers=1,roots_only=roots_only).double()
    with torch.no_grad():model.head.weight.normal_(0,.1);model.head.bias.fill_(.03)
    p=center(torch.randn(row['x'].shape,dtype=torch.float64))
    return row,model,p


def test_graph_time_reversal_and_joint_atom_O3_equivariance():
    r,m,p=example();x=r['x'];b=r['bonds'];c=r['new_bonds'];z=r['numbers'];e=r['electronic'];a=r['action']
    f=m(x,b,c,z,e,a,.2)
    torch.testing.assert_close(f,m(x,c,b,z,e,r['inverse_action'],.8),atol=1e-12,rtol=0)
    rotation=torch.linalg.qr(torch.randn(3,3,dtype=x.dtype))[0];rotation[:,0]*=-1
    torch.testing.assert_close(m(x@rotation,b,c,z,e,a,.2),f@rotation,atol=1e-10,rtol=0)
    perm=torch.randperm(len(x));inv=torch.argsort(perm);ap=tuple(int(inv[v]) for v in a)
    torch.testing.assert_close(m(x[perm],b[perm][:,perm],c[perm][:,perm],z[perm],e,ap,.2),f[perm],atol=1e-10,rtol=0)


@pytest.mark.parametrize('roots_only',[False,True])
def test_full_edit_bridge_inverse_intrinsic_volume_and_MH_ratio_reversal(roots_only):
    r,m,p=example(roots_only);x=r['x'];b=r['bonds'];c=r['new_bonds'];z=r['numbers'];e=r['electronic'];a=r['action'];ar=r['inverse_action']
    call=lambda u,v,g,h,action:edit_bridge(u,v,g,h,z,e,action,m,steps_per_side=1)
    y,q,j=call(x,p,b,c,a);xx,pp,jj=call(y,q,c,b,ar)
    if roots_only:
        passive=[i for i in range(len(x)) if i not in a[:2]]
        torch.testing.assert_close(y[passive]-y[passive[0]],x[passive]-x[passive[0]],atol=1e-12,rtol=0)
    torch.testing.assert_close(xx,x,atol=1e-11,rtol=0);torch.testing.assert_close(pp,p,atol=1e-11,rtol=0);torch.testing.assert_close(jj,-j)
    basis=centered_orthonormal_basis(len(x));dim=3*(len(x)-1);free=torch.cat([(basis.T@x).flatten(),(basis.T@p).flatten()])
    def transform(free):
        y,q,_=call(basis@free[:dim].reshape(-1,3),basis@free[dim:].reshape(-1,3),b,c,a)
        return torch.cat([(basis.T@y).flatten(),(basis.T@q).flatten()])
    jac=torch.autograd.functional.jacobian(transform,free)
    torch.testing.assert_close(torch.linalg.slogdet(jac)[1],j,atol=1e-10,rtol=0)
    ratio=augmented_log_ratio(x.square().sum(),y.square().sum(),p,q,.026,j,x.new_tensor(.3))
    reverse=augmented_log_ratio(y.square().sum(),xx.square().sum(),q,pp,.026,jj,x.new_tensor(-.3))
    torch.testing.assert_close(reverse,-ratio,atol=1e-9,rtol=0)


def test_zero_length_bridge_and_nonzero_parameter_gradient():
    from cfm_mol.chemical_moves import exchange_terminal_sites
    r,m,p=example();x=r['x'];args=(r['bonds'],r['new_bonds'],r['numbers'],r['electronic'],r['action'],m)
    y,q,j=edit_bridge(x,p,*args,steps_per_side=0)
    expected,volume,_=exchange_terminal_sites(x,r['radii'],r['action'])
    torch.testing.assert_close(y,expected);torch.testing.assert_close(q,-p);torch.testing.assert_close(j,volume)
    y,q,j=edit_bridge(x,p,*args,steps_per_side=2)
    loss=(y-r['y']).square().sum()+.01*q.square().sum();loss.backward()
    assert torch.isfinite(m.head.weight.grad).all() and float(m.head.weight.grad.abs().max())>1e-6
