import torch
from cfm_mol.geometric_domain import connected_nonoverlapping


def test_domain_rejects_fragments_and_overlap_and_preserves_atom_symmetries():
    x=torch.tensor([[[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]],
                    [[0.,0.,0.],[1.,0.,0.],[5.,0.,0.]],
                    [[0.,0.,0.],[.1,0.,0.],[1.,0.,0.]]],dtype=torch.float64)
    radii=torch.full((3,),.5,dtype=torch.float64)
    valid=connected_nonoverlapping(x,radii)
    assert valid.tolist()==[True,False,False]
    order=torch.tensor([2,0,1]);rotation=torch.linalg.qr(torch.randn(3,3,dtype=torch.float64))[0]
    assert torch.equal(connected_nonoverlapping(-x[:,order]@rotation+3,radii[order]),valid)
