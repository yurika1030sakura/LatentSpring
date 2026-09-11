import torch
from cfm_mol.chemical_moves import exchange_terminal_sites,covalent_radii
from cfm_mol.nonequilibrium import centered_orthonormal_basis


def test_terminal_exchange_inverse_intrinsic_volume_and_atom_symmetries():
    numbers=torch.tensor([6,8,1,9]);radii=covalent_radii(numbers)
    x=torch.tensor([[-1.,0.,0.],[1.,0.,0.],[-1.,1.05,.1],[1.,1.3,.2]],dtype=torch.float64)
    x=x-x.mean(0);action=(2,3,0,1)
    y,volume,inverse_action=exchange_terminal_sites(x,radii,action)
    recovered,reverse_volume,_=exchange_terminal_sites(y,radii,inverse_action)
    torch.testing.assert_close(recovered,x,atol=1e-12,rtol=0)
    torch.testing.assert_close(volume+reverse_volume,torch.zeros_like(volume),atol=1e-12,rtol=0)
    basis=centered_orthonormal_basis(4);z=(basis.T@x).flatten().requires_grad_()
    def transform(z):return (basis.T@exchange_terminal_sites(basis@z.reshape(3,3),radii,action)[0]).flatten()
    jac=torch.autograd.functional.jacobian(transform,z)
    torch.testing.assert_close(torch.linalg.slogdet(jac)[1],volume,atol=1e-12,rtol=0)
    rotation=torch.linalg.qr(torch.randn(3,3,dtype=torch.float64))[0]
    transformed,_,_=exchange_terminal_sites(-x@rotation,radii,action)
    torch.testing.assert_close(transformed,-y@rotation,atol=1e-12,rtol=0)
    order=torch.tensor([3,0,2,1]);lookup=torch.argsort(order)
    mapped_action=tuple(int(lookup[i]) for i in action)
    permuted,_,_=exchange_terminal_sites(x[order],radii[order],mapped_action)
    torch.testing.assert_close(permuted,y[order],atol=1e-12,rtol=0)
