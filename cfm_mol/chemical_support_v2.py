"""Explicit experimental extension of chemical support to neutral radicals.

This defines a NEW support, never the target of the preserved v1 trajectories.
RDKit graph/radical labels do not certify quantum spin or energy-model accuracy.
"""
import numbers as numeric
import torch
from rdkit import Chem,rdBase
from rdkit.Chem import rdDetermineBonds
from rdkit.Geometry import Point3D
from cfm_mol.chemical_moves import infer_chemical_graph,covalent_radii
from cfm_mol.geometric_domain import connected_nonoverlapping


class ChemicalValidatorError(RuntimeError):
    """Graph validator unavailable/failed; not a chemical-invalidity verdict."""


def check_electron_sector(numbers,charge,spin_multiplicity):
    if (not isinstance(charge,numeric.Integral) or not isinstance(spin_multiplicity,numeric.Integral)
            or spin_multiplicity<1 or not numbers
            or any(not isinstance(z,numeric.Integral) or not 1<=z<=118 for z in numbers)):
        raise ValueError('Integer atomic identities, charge and positive spin multiplicity required')
    electrons=sum(numbers)-charge;unpaired=spin_multiplicity-1
    if electrons<unpaired or (electrons-unpaired)%2:
        raise ValueError('Electron count and spin multiplicity have incompatible parity or magnitude')
    return electrons


def infer_chemical_graph_v2(x,numbers,charge,spin_multiplicity):
    """Charged-mode v1 first, neutral radical-mode fallback on a fresh molecule.

    Nonzero-charge radical combinations and unsupported metals remain limited
    by the installed RDKit implementation. No false all-OMol coverage claim.
    Reference: https://www.rdkit.org/docs/source/rdkit.Chem.rdDetermineBonds.html
    """
    electrons=check_electron_sector(numbers,charge,spin_multiplicity)
    try:
        graph=infer_chemical_graph(x,numbers,charge)
        return dict(**graph,support_version=2,assignment_mode='charged',electron_count=electrons,
            spin_multiplicity=spin_multiplicity,quantum_spin_certified=False)
    except (IndexError,RuntimeError) as exc:
        raise ChemicalValidatorError(f'{type(exc).__name__}: {exc}') from exc
    except ValueError as exc:
        charged_failure=str(exc)
    if charge!=0:raise ValueError('Charged assignment rejected; ionic-radical fallback is not qualified: '+charged_failure)
    if x.shape!=(len(numbers),3) or not torch.isfinite(x).all():raise ValueError('Invalid coordinate matrix')
    if not bool(connected_nonoverlapping(x[None],covalent_radii(numbers,dtype=x.dtype,device=x.device))[0]):
        raise ValueError('Outside the connected/no-overlap geometric domain')
    builder=Chem.RWMol()
    for z in numbers:
        atom=Chem.Atom(int(z));atom.SetNoImplicit(True);builder.AddAtom(atom)
    molecule=builder.GetMol();conformer=Chem.Conformer(len(numbers))
    for i,xyz in enumerate(x.detach().cpu().tolist()):conformer.SetAtomPosition(i,Point3D(*xyz))
    molecule.AddConformer(conformer,assignId=True)
    try:
        with rdBase.BlockLogs():
            rdDetermineBonds.DetermineBonds(molecule,charge=0,covFactor=1.25,
                allowChargedFragments=False,embedChiral=False,useVdw=True)
            Chem.SanitizeMol(molecule)
    except (IndexError,RuntimeError) as exc:
        raise ChemicalValidatorError(f'Radical mode {type(exc).__name__}: {exc}') from exc
    if Chem.GetFormalCharge(molecule)!=charge or len(Chem.GetMolFrags(molecule))!=1:
        raise ValueError('Radical assignment violates total charge or connectivity')
    bonds=torch.zeros(len(numbers),len(numbers),dtype=x.dtype,device=x.device)
    for bond in molecule.GetBonds():
        i,j=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx();bonds[i,j]=bonds[j,i]=bond.GetBondTypeAsDouble()
    copy=Chem.Mol(molecule);Chem.RemoveStereochemistry(copy)
    with rdBase.BlockLogs():copy=Chem.RemoveHs(copy)
    return dict(bond_orders=bonds,connectivity_smiles=Chem.MolToSmiles(copy,isomericSmiles=False),
        formal_charges=[a.GetFormalCharge() for a in molecule.GetAtoms()],
        radical_electrons=[a.GetNumRadicalElectrons() for a in molecule.GetAtoms()],
        support_version=2,assignment_mode='neutral_radical',electron_count=electrons,
        spin_multiplicity=spin_multiplicity,quantum_spin_certified=False,
        charged_assignment_rejection=charged_failure)
