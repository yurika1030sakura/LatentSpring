"""Neighbourhood selection must preserve parent labels and energy precision."""
import torch
import pytest
from cfm_mol.perturbation_loader import PerturbationLoader


def test_subset_keeps_geometry_energy_alignment_and_float64(tmp_path):
    positions=[];types=[];charges=[];nia=[];energies=[];groups=[];offset=0
    for parent,n_atoms in enumerate([2,3]):
        for k in range(3):
            positions.append(torch.full((n_atoms,3),parent*10.+k))
            types.append(torch.arange(n_atoms)%2);charges.append(torch.zeros(n_atoms,dtype=torch.long))
            nia.append([offset,offset+n_atoms]);offset+=n_atoms
            energies.append(2.**30+parent+k*.125);groups.append(parent)
    path=tmp_path/'shard.pt'
    torch.save(dict(positions=torch.cat(positions),atom_types=torch.cat(types),
        atom_charges=torch.cat(charges),node_idx_array=torch.tensor(nia),
        energies=torch.tensor(energies,dtype=torch.float64),group_id=torch.tensor(groups),K=3),path)
    loader=PerturbationLoader(path,n_atom_types=2,b_parents=2,device='cpu',perturbation_indices=[2,0])
    loader._order=[0,1]
    g,e,pid,nbi,_=loader.next_batch()
    assert g.batch_num_nodes().tolist()==[2,2,3,3]
    assert pid.tolist()==[0,0,1,1]
    assert e.dtype==torch.float64 and (e[0]-e[1]).item()==.25
    assert e.tolist()==[energies[i] for i in [2,0,5,3]]
    for i,expected in enumerate([2.,0.,12.,10.]):
        assert (g.ndata['x_1_true'][nbi==i]==expected).all()
    with pytest.raises(ValueError,match='distinct valid'):
        PerturbationLoader(path,n_atom_types=2,perturbation_indices=[1,1],device='cpu')
