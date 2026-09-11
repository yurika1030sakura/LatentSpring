"""Auditable chemical graph perception and reversible terminal-site exchanges.

Graph perception is an explicit algorithmic validity criterion, not a quantum
certificate. Exchanges are Monte Carlo proposals, not reaction trajectories.
"""
import math
import torch
from rdkit import Chem,rdBase
from rdkit.Chem import rdDetermineBonds
from rdkit.Geometry import Point3D
from cfm_mol.geometric_domain import connected_nonoverlapping


def covalent_radii(numbers,*,dtype=torch.float64,device=None):
    table=Chem.GetPeriodicTable()
    return torch.tensor([table.GetRcovalent(int(z)) for z in numbers],dtype=dtype,device=device)


def infer_chemical_graph(x,numbers,charge,*,require_connected=True):
    if x.shape!=(len(numbers),3) or not torch.isfinite(x).all():raise ValueError('Invalid coordinate matrix')
    radii=covalent_radii(numbers,dtype=x.dtype,device=x.device)
    if require_connected and not bool(connected_nonoverlapping(x[None],radii)[0]):
        raise ValueError('Outside the connected/no-overlap geometric domain')
    builder=Chem.RWMol()
    for z in numbers:
        atom=Chem.Atom(int(z));atom.SetNoImplicit(True);builder.AddAtom(atom)
    molecule=builder.GetMol();conformer=Chem.Conformer(len(numbers))
    for i,xyz in enumerate(x.detach().cpu().tolist()):conformer.SetAtomPosition(i,Point3D(*xyz))
    molecule.AddConformer(conformer,assignId=True)
    with rdBase.BlockLogs():
        rdDetermineBonds.DetermineBonds(molecule,charge=int(charge),covFactor=1.25,
            allowChargedFragments=True,embedChiral=False,useVdw=True)
        Chem.SanitizeMol(molecule)
    if Chem.GetFormalCharge(molecule)!=charge:raise ValueError('Inferred charge differs from the condition')
    if require_connected and len(Chem.GetMolFrags(molecule))!=1:raise ValueError('Inferred molecule is disconnected')
    bonds=torch.zeros(len(numbers),len(numbers),dtype=x.dtype,device=x.device)
    for bond in molecule.GetBonds():
        i,j=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx();bonds[i,j]=bonds[j,i]=bond.GetBondTypeAsDouble()
    # Do not mistake non-tetrahedral coordinate-derived tags for different
    # constitutional identities. Stereo assessment remains separate.
    copy=Chem.Mol(molecule);Chem.RemoveStereochemistry(copy)
    with rdBase.BlockLogs():copy=Chem.RemoveHs(copy)
    return dict(bond_orders=bonds,connectivity_smiles=Chem.MolToSmiles(copy,isomericSmiles=False),
        formal_charges=[a.GetFormalCharge() for a in molecule.GetAtoms()],
        radical_electrons=[a.GetNumRadicalElectrons() for a in molecule.GetAtoms()])


def terminal_exchange_actions(numbers,bond_orders):
    adjacency=bond_orders>0;degrees=adjacency.sum(1)
    allowed={1,9,17,35,53};leaves=[i for i,z in enumerate(numbers) if int(z) in allowed and int(degrees[i])==1]
    result=[]
    for at,i in enumerate(leaves):
        k=int(adjacency[i].nonzero()[0,0])
        for j in leaves[at+1:]:
            l=int(adjacency[j].nonzero()[0,0])
            if int(numbers[i])==int(numbers[j]) or k in (i,j) or l in (i,j):continue
            if float(bond_orders[i,k])!=1. or float(bond_orders[j,l])!=1.:continue
            result.append((i,j,k,l))
    return result


def exchange_terminal_sites(x,radii,action):
    """Swap anchor-relative leaf vectors, rescaling by covalent-radius ratios.

    Passive anchors may coincide. The map is translation equivariant before
    centering, so its intrinsic absolute determinant is (scale_i*scale_j)^3.
    The inverse action has the two anchor indices exchanged.
    """
    i,j,k,l=action
    if i==j or k in (i,j) or l in (i,j):raise ValueError('Terminal atoms require passive anchors')
    si=(radii[i]+radii[l])/(radii[j]+radii[l])
    sj=(radii[j]+radii[k])/(radii[i]+radii[k])
    y=x.clone();y[i]=x[l]+si*(x[j]-x[l]);y[j]=x[k]+sj*(x[i]-x[k])
    y=y-y.mean(0,keepdim=True)
    return y,3*(si.log()+sj.log()),(i,j,l,k)
