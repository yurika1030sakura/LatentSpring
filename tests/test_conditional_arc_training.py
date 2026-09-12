import copy
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from cfm_mol.chemical_moves import infer_chemical_graph
from cfm_mol.conditional_arc_energy import ConditionalArcEnergy
from scripts.research.train_conditional_arc_energy import objective


def test_withheld_labels_cannot_change_training_loss_or_parameter_gradient():
    mol=Chem.AddHs(Chem.MolFromSmiles('FCCF'));assert AllChem.EmbedMolecule(mol,randomSeed=25531)==0
    x=torch.tensor(mol.GetConformer().GetPositions(),dtype=torch.float64);x-=x.mean(0)
    z=torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()]);bonds=infer_chemical_graph(x,z.tolist(),0)['bond_orders']
    directions=torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],dtype=x.dtype)
    row=dict(role='fit',positions=x,numbers=z,bonds=bonds,electronic=x.new_tensor([0.,1.,.026]),root=torch.tensor([0,1]),
        directions=directions,work_eV=x.new_tensor([0.,.2,.5]),angular_energy_gradients_eV=x.new_tensor([[0.,.2,0.],[.3,0.,0.],[0.,0.,0.]]),
        training_mask=torch.tensor([True,True,False]))
    altered=copy.deepcopy(row);altered['work_eV'][-1]=1e8;altered['angular_energy_gradients_eV'][-1]=1e8
    model=ConditionalArcEnergy().double()
    for weight in [0.,.1]:
        loss,_=objective(model,[row],weight,.1)
        expected=torch.autograd.grad(loss,model.query_head[-1].weight)[0]
        changed,_=objective(model,[altered],weight,.1)
        observed=torch.autograd.grad(changed,model.query_head[-1].weight)[0]
        torch.testing.assert_close(loss,changed,atol=0,rtol=0)
        torch.testing.assert_close(expected,observed,atol=0,rtol=0)
        assert torch.isfinite(expected).all() and expected.norm()>0
