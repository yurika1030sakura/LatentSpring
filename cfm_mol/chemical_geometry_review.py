"""Additional chemical geometry diagnostics; original benchmark scores stay separate."""
import numpy as np
from rdkit import Chem,rdBase
from rdkit.Chem import rdDetermineBonds
from posebusters.modules.distance_geometry import check_geometry
from posebusters.modules.flatness import check_flatness,flat_rings,flat_bonds,nonflat


def molecule_from_coordinates(positions,numbers,charge=0):
    x=np.asarray(positions,dtype=float);builder=Chem.RWMol()
    for z in numbers:
        atom=Chem.Atom(int(z));atom.SetNoImplicit(True);builder.AddAtom(atom)
    mol=builder.GetMol();conf=Chem.Conformer(len(numbers))
    for i,p in enumerate(x):conf.SetAtomPosition(i,p.tolist())
    mol.AddConformer(conf,assignId=True)
    with rdBase.BlockLogs():
        rdDetermineBonds.DetermineBonds(mol,charge=int(charge),covFactor=1.25,allowChargedFragments=True,embedChiral=False,useVdw=True)
        Chem.SanitizeMol(mol)
    if len(Chem.GetMolFrags(mol))!=1 or Chem.GetFormalCharge(mol)!=charge:raise ValueError('Connectivity or charge mismatch')
    np.testing.assert_array_equal(mol.GetConformer().GetPositions(),x)
    return mol


def contact_diagnostics(positions,numbers):
    x=np.asarray(positions,dtype=float);z=np.asarray(numbers);r=np.array([Chem.GetPeriodicTable().GetRcovalent(int(v)) for v in z])
    d=np.linalg.norm(x[:,None]-x[None],axis=-1)/(r[:,None]+r[None]);np.fill_diagonal(d,np.inf)
    adjacency=d<=1.25
    def components(indices):
        unseen=set(map(int,indices));out=[]
        while unseen:
            stack=[unseen.pop()];part=[]
            while stack:
                i=stack.pop();part.append(i);neighbors=set(np.flatnonzero(adjacency[i]))&unseen
                unseen-=neighbors;stack.extend(neighbors)
            out.append(part)
        return out
    parts=components(range(len(z)));heavy=components(np.flatnonzero(z!=1))
    return dict(contact_components=len(parts),heavy_components=len(heavy),detached_hydrogen_atoms=int(sum(z[i]==1 and not any(adjacency[i]&(z!=1)) for i in range(len(z)))),
        severe_overlap=bool(np.any(d<.6)),hydrogen_only_fragmentation=bool(len(parts)>1 and len(heavy)==1))


def geometry_diagnostics(mol):
    with rdBase.BlockLogs():
        g=check_geometry(mol,threshold_bad_bond_length=.25,threshold_bad_angle=.25,threshold_clash=.3)['results']
        f=check_flatness(mol,threshold_flatness=.25,flat_systems=flat_rings|flat_bonds)['results']
        nf=check_flatness(mol,threshold_flatness=.05,flat_systems=nonflat,check_nonflat=True)['results']
    flags={k:bool(g[k]) if np.isfinite(g[k]) else False for k in ['bond_lengths_within_bounds','bond_angles_within_bounds','no_internal_clash']}
    flags['planar_groups_pass']=bool(f['flatness_passes']) if np.isfinite(f['flatness_passes']) else False
    flags['nonaromatic_rings_nonflat']=bool(nf['flatness_passes']) if np.isfinite(nf['flatness_passes']) else False
    radicals=sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms());rings=list(mol.GetRingInfo().AtomRings())
    small_sp_rings=sum(len(r)<8 and any(mol.GetAtomWithIdx(i).GetHybridization()==Chem.HybridizationType.SP for i in r) for r in rings)
    copy=Chem.Mol(mol);Chem.RemoveStereochemistry(copy)
    with rdBase.BlockLogs():copy=Chem.RemoveHs(copy)
    return dict(**flags,geometry_subset_pass=bool(all(flags.values())),closed_shell_geometry_pass=bool(all(flags.values()) and radicals==0),
        radical_electrons=radicals,charged_atoms=sum(a.GetFormalCharge()!=0 for a in mol.GetAtoms()),
        ring_sizes=[len(r) for r in rings],aromatic_atoms=sum(a.GetIsAromatic() for a in mol.GetAtoms()),small_sp_rings=small_sp_rings,
        smiles=Chem.MolToSmiles(copy,isomericSmiles=False),checked_planar_groups=int(f['num_systems_checked']),
        max_planar_deviation=float(f['max_distance']) if np.isfinite(f['max_distance']) else None)
