"""kraken phosphine library loader (ICLR E1 OOD slice #2).

Source: Gensch, T. et al. A comprehensive discovery platform for
organophosphorus ligands for catalysis. J. Am. Chem. Soc. 144, 1205-1217
(2022). DOI: 10.1021/jacs.1c09718.

**Important**: No public bulk xyz archive exists (confirmed by inspecting
Figshare collection 5789891, GitHub repo SigmanGroup/kraken, and MolSSI
interactive site). Authors deposit only SMILES + DFT descriptors. Bulk
geometries would require re-running the workflow, impractical for our
timeline.

Workaround: we download the SMILES CSV (1544 phosphines) and generate
3D conformers with RDKit ETKDGv3 + MMFF. Not DFT-quality but sufficient
for our OOD validity test, which checks whether baselines produce
chemically reasonable phosphine TOPOLOGIES (valence, connectivity,
unusual P coordination) -- not exact bond lengths.

Source URL:
    https://raw.githubusercontent.com/SigmanGroup/kraken/master/data/kraken_library.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


SLICE_NAME = "kraken_unusual_P"
ATOM_MAP = ["C", "H", "N", "O", "F", "P", "S", "Cl", "Br", "I", "B", "Si"]
_MAX_ATOMS = 100
_SMILES_COLUMN_CANDIDATES = ("SMILES", "smiles", "canonical_smiles")


def download(dest_dir: Path) -> None:
    """Download the kraken SMILES library CSV (~3 MB)."""
    import subprocess
    raw = dest_dir / "kraken_raw"
    raw.mkdir(parents=True, exist_ok=True)
    csv_path = raw / "kraken_library.csv"
    if csv_path.exists():
        print(f"kraken CSV already at {csv_path}; skipping download.")
        return
    url = "https://raw.githubusercontent.com/SigmanGroup/kraken/master/data/kraken_library.csv"
    print(f"downloading kraken library from {url} ...")
    subprocess.check_call(["wget", "-O", str(csv_path), url])


def _embed_3d(smi: str, mmff_iters: int = 500):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) != 0:
        if AllChem.EmbedMolecule(mol, useRandomCoords=True) != 0:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=mmff_iters)
    except Exception:
        pass
    return mol


def _rdkit_mol_to_arrays(mol, atom_map: list[str]):
    from rdkit import Chem
    atom_to_idx = {s: i for i, s in enumerate(atom_map)}
    conf = mol.GetConformer()
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    if any(s not in atom_to_idx for s in symbols):
        return None
    coords = np.array([list(conf.GetAtomPosition(i))
                       for i in range(mol.GetNumAtoms())], dtype=np.float32)
    a = np.array([atom_to_idx[s] for s in symbols], dtype=np.int64)
    bt_map = {
        Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2,
        Chem.BondType.TRIPLE: 3, Chem.BondType.AROMATIC: 4,
    }
    bond_idxs, bond_types = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_idxs.append([i, j])
        bond_types.append(bt_map.get(bond.GetBondType(), 1))
    return coords, a, np.array(bond_idxs, dtype=np.int64), np.array(bond_types, dtype=np.int32)


def _has_unusual_P(mol) -> bool:
    """P attached to >=3 heavy atoms with at least one heteroatom (not C/H)."""
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "P":
            continue
        neighbours = [nb.GetSymbol() for nb in atom.GetNeighbors()]
        heavy = [s for s in neighbours if s != "H"]
        if len(heavy) < 3:
            continue
        if any(s not in ("C", "H") for s in heavy):
            return True
    return False


def _iter_smiles(csv_path: Path):
    """Yield SMILES strings from the kraken CSV (any of several column names)."""
    import csv
    with open(csv_path, "rt", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        smiles_col = next(
            (c for c in _SMILES_COLUMN_CANDIDATES if c in fieldnames),
            None,
        )
        if smiles_col is None:
            raise RuntimeError(
                f"No SMILES column found. CSV columns: {fieldnames}"
            )
        for row in reader:
            smi = row.get(smiles_col, "").strip()
            if smi:
                yield smi


def process(
    raw_dir: Path,
    processed_dir: Path,
    atom_map: list[str] = ATOM_MAP,
    unusual_P_only: bool = True,
    max_mols: int | None = None,
) -> None:
    import torch
    from torch.nn.functional import one_hot
    from rdkit import Chem

    atom_to_idx = {s: i for i, s in enumerate(atom_map)}
    n_atom_types = len(atom_map)

    csv_path = raw_dir / "kraken_raw" / "kraken_library.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run kraken.download(raw_dir) first."
        )

    positions_list, atom_types_list, charges_list = [], [], []
    bond_types_list, bond_idxs_list = [], []
    node_idx_array, edge_idx_array = [], []
    n_nodes, n_edges, kept, seen = 0, 0, 0, 0

    for smi in _iter_smiles(csv_path):
        seen += 1
        mol = _embed_3d(smi)
        if mol is None:
            continue
        if unusual_P_only and not _has_unusual_P(mol):
            continue
        if mol.GetNumAtoms() > _MAX_ATOMS:
            continue
        arrs = _rdkit_mol_to_arrays(mol, atom_map)
        if arrs is None:
            continue
        coords, a, bi, bt = arrs
        if bi.shape[0] == 0:
            continue

        positions_list.append(torch.tensor(coords, dtype=torch.float32))
        atom_types_list.append(one_hot(torch.tensor(a), num_classes=n_atom_types).float())
        charges_list.append(
            one_hot(torch.zeros(len(a), dtype=torch.long) + 2, num_classes=6).float()
        )
        bond_types_list.append(torch.tensor(bt, dtype=torch.int32))
        bond_idxs_list.append(torch.tensor(bi, dtype=torch.int64))
        node_idx_array.append([n_nodes, n_nodes + len(a)])
        edge_idx_array.append([n_edges, n_edges + bi.shape[0]])
        n_nodes += len(a)
        n_edges += bi.shape[0]
        kept += 1
        if max_mols is not None and kept >= max_mols:
            break

    print(f"scanned {seen} SMILES, kept {kept} phosphines "
          f"(unusual_P_only={unusual_P_only}), "
          f"{n_nodes} atoms, {n_edges} bonds")
    if kept == 0:
        raise RuntimeError("No kraken molecules survived filtering.")

    out = {
        "positions": torch.cat(positions_list, dim=0),
        "atom_types": torch.cat(atom_types_list, dim=0),
        "atom_charges": torch.cat(charges_list, dim=0),
        "bond_types": torch.cat(bond_types_list, dim=0),
        "bond_idxs": torch.cat(bond_idxs_list, dim=0),
        "node_idx_array": torch.tensor(node_idx_array, dtype=torch.long),
        "edge_idx_array": torch.tensor(edge_idx_array, dtype=torch.long),
    }
    processed_dir.mkdir(parents=True, exist_ok=True)
    torch.save(out, processed_dir / "val_data_processed.pt")
    torch.save(out, processed_dir / "test_data_processed.pt")
    p_a = torch.full((n_atom_types,), 1.0 / n_atom_types)
    p_c = torch.full((6,), 1.0 / 6)
    p_e = torch.full((5,), 1.0 / 5)
    p_c_given_a = p_c.unsqueeze(0).expand(n_atom_types, -1).clone()
    torch.save((p_a, p_c, p_e, p_c_given_a),
               processed_dir / "train_data_marginal_dists.pt")
    print(f"wrote processed kraken data to {processed_dir}")


def main_cli() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=Path, required=True)
    ap.add_argument("--processed_dir", type=Path, required=True)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--unusual_P_only", action="store_true", default=True)
    ap.add_argument("--max_mols", type=int, default=None)
    args = ap.parse_args()
    if args.download:
        download(args.raw_dir)
    process(args.raw_dir, args.processed_dir,
            unusual_P_only=args.unusual_P_only, max_mols=args.max_mols)


if __name__ == "__main__":
    main_cli()
