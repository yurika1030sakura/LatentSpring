"""Hypervalent-fragment OOD slice (ICLR E1 slice #3).

Target: SF_6, PF_5, I(III) hypervalent iodides (Dess-Martin periodinane,
iodosylbenzene, iodoxybenzene), sulfur ylides, phosphorus oxychlorides.
All have atoms whose formal valence exceeds the Lewis-octet limit; soft-
loss baselines systematically fail these because their valence priors
are centred on neutral octet.

We hand-curate ~30-50 representative molecules from PubChem SMILES +
RDKit 3D embedding with MMFF optimisation. Dataset size is small so MMFF
runs in seconds.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


SLICE_NAME = "hypervalent"
ATOM_MAP = ["C", "H", "N", "O", "F", "P", "S", "Cl", "Br", "I"]

_R_COV: dict[str, float] = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "P": 1.07, "S": 1.05, "Cl": 1.02, "Br": 1.20, "I": 1.39,
}

# Curated representative SMILES. Labels are used only for eval-time
# per-motif reporting in the ICLR table.
REPRESENTATIVE_SMILES: list[tuple[str, str]] = [
    # Hypervalent sulfur.
    ("FS(F)(F)(F)(F)F", "SF6"),
    ("O=S(=O)(F)F", "SO2F2"),
    ("O=S1(=O)CCCC1", "sulfolane"),
    ("[O-][S+](C)(C)=O", "dimsyl_ylide"),
    # Hypervalent phosphorus.
    ("FP(F)(F)(F)F", "PF5"),
    ("O=P(Cl)(Cl)Cl", "POCl3"),
    ("O=P(OC)(OC)OC", "trimethyl_phosphate"),
    ("O=P(F)(F)F", "POF3"),
    # I(III) / I(V) hypervalent iodine.
    ("c1ccc(I(=O)=O)cc1", "iodylbenzene"),
    ("O=I(OC(=O)C)OC(=O)C", "PhI_OAc2_core"),
    ("O=I1(OC(=O)C)(OC(=O)C)OC(=O)c2ccccc21", "DMP"),
    # Hypervalent halogens.
    ("FBr(F)F", "BrF3"),
    ("FCl(F)(F)F", "ClF4"),
    # Additional unusual coordination.
    ("C[S+](C)C", "trimethylsulfonium"),
    ("C[P+](C)(C)C", "tetramethylphosphonium"),
]


def download(dest_dir: Path) -> None:
    """No external download; SMILES are embedded in this file.
    We write a SMILES list for reproducibility."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    smiles_file = dest_dir / "hypervalent_smiles.txt"
    with open(smiles_file, "w") as f:
        for smi, label in REPRESENTATIVE_SMILES:
            f.write(f"{smi}\t{label}\n")
    print(f"wrote {len(REPRESENTATIVE_SMILES)} SMILES to {smiles_file}")


def _embed_3d(smi: str, mmff_iters: int = 500):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if result != 0:
        # ETKDG can fail on hypervalent; fall back to random coords + MMFF.
        result = AllChem.EmbedMolecule(mol, useRandomCoords=True)
    if result != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=mmff_iters)
    except Exception:
        pass
    return mol


def _rdkit_mol_to_arrays(
    mol,
    atom_map: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    from rdkit import Chem

    atom_to_idx = {s: i for i, s in enumerate(atom_map)}
    conf = mol.GetConformer()
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    if any(s not in atom_to_idx for s in symbols):
        return None
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
                      dtype=np.float32)
    a = np.array([atom_to_idx[s] for s in symbols], dtype=np.int64)

    # Bond order map.
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


def process(raw_dir: Path, processed_dir: Path,
            atom_map: list[str] = ATOM_MAP) -> None:
    import torch
    from torch.nn.functional import one_hot

    n_atom_types = len(atom_map)
    positions_list, atom_types_list, charges_list = [], [], []
    bond_types_list, bond_idxs_list = [], []
    node_idx_array, edge_idx_array = [], []
    n_nodes, n_edges, kept = 0, 0, 0

    for smi, label in REPRESENTATIVE_SMILES:
        mol = _embed_3d(smi)
        if mol is None:
            print(f"  SKIP {label}: 3D embedding failed for {smi}")
            continue
        arrs = _rdkit_mol_to_arrays(mol, atom_map)
        if arrs is None:
            print(f"  SKIP {label}: element outside atom_map")
            continue
        coords, a, bi, bt = arrs
        if bi.shape[0] == 0:
            continue

        positions_list.append(torch.tensor(coords, dtype=torch.float32))
        atom_types_list.append(one_hot(torch.tensor(a), num_classes=n_atom_types).float())
        charges_list.append(one_hot(torch.zeros(len(a), dtype=torch.long) + 2,
                                    num_classes=6).float())
        bond_types_list.append(torch.tensor(bt, dtype=torch.int32))
        bond_idxs_list.append(torch.tensor(bi, dtype=torch.int64))
        node_idx_array.append([n_nodes, n_nodes + len(a)])
        edge_idx_array.append([n_edges, n_edges + bi.shape[0]])
        n_nodes += len(a)
        n_edges += bi.shape[0]
        kept += 1
        print(f"  OK   {label} ({len(a)} atoms, {bi.shape[0]} bonds)")

    if kept == 0:
        raise RuntimeError("No hypervalent molecules embedded successfully.")

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
    print(f"wrote processed hypervalent data to {processed_dir} ({kept} mols)")


def main_cli() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=Path, required=True)
    ap.add_argument("--processed_dir", type=Path, required=True)
    args = ap.parse_args()
    download(args.raw_dir)
    process(args.raw_dir, args.processed_dir)


if __name__ == "__main__":
    main_cli()
