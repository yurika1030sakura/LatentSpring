import pytest
import torch
from cfm_mol.chemical_moves import infer_chemical_graph
from cfm_mol.chemical_support_v2 import infer_chemical_graph_v2,check_electron_sector,ChemicalValidatorError


def test_methyl_radical_is_not_rejected_as_an_incorrect_total_charge():
    x=torch.tensor([[0.,0.,0.],[1.09,0.,0.],[-.545,.944,0.],[-.545,-.944,0.]],dtype=torch.float64)
    with pytest.raises(ValueError):infer_chemical_graph(x,[6,1,1,1],0)
    result=infer_chemical_graph_v2(x,[6,1,1,1],0,2)
    assert result['assignment_mode']=='neutral_radical'
    assert result['electron_count']==9 and sum(result['radical_electrons'])==1
    assert result['connectivity_smiles']=='[CH3]'
    assert not result['quantum_spin_certified']
    q=torch.linalg.qr(torch.randn(3,3,dtype=x.dtype))[0];perm=torch.tensor([3,0,2,1])
    transformed=infer_chemical_graph_v2((x@q)[perm],[1,6,1,1],0,2)
    assert transformed['connectivity_smiles']==result['connectivity_smiles']
    torch.testing.assert_close(transformed['bond_orders'],result['bond_orders'][perm][:,perm])


def test_closed_shell_ion_is_preserved_and_electronic_labels_are_checked():
    x=torch.tensor([[0.,0.,0.],[1.,1.,1.],[1.,-1.,-1.],[-1.,1.,-1.],[-1.,-1.,1.]],dtype=torch.float64)
    x[1:]*=1.03/3**.5
    old=infer_chemical_graph(x,[7,1,1,1,1],1)
    new=infer_chemical_graph_v2(x,[7,1,1,1,1],1,1)
    assert new['assignment_mode']=='charged' and sum(new['formal_charges'])==1
    torch.testing.assert_close(new['bond_orders'],old['bond_orders'])
    assert check_electron_sector([8,8],0,3)==16  # O2 triplet need not have RDKit radical flags.
    with pytest.raises(ValueError,match='parity'):check_electron_sector([6,1,1,1],0,1)
    with pytest.raises(ValueError):check_electron_sector([1],2,1)


def test_validator_crash_is_distinct_from_chemical_rejection(monkeypatch):
    def unsupported(*args):raise IndexError('unordered_map::at')
    monkeypatch.setattr('cfm_mol.chemical_support_v2.infer_chemical_graph',unsupported)
    with pytest.raises(ChemicalValidatorError,match='IndexError'):
        infer_chemical_graph_v2(torch.zeros(2,3),[6,6],0,1)
