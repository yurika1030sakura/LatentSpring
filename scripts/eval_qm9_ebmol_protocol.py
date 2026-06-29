"""QM9 generation evaluation under the EBMol protocol (Experiment Q1).

Consumes the JSON produced by ``scripts/sample_bgfm.py`` and reports
the EBMol-aligned QM9 metrics on 10k generated molecules:

  - atom stability       fraction of atoms with valence in [val_lo, val_hi]
  - molecule stability   fraction of molecules with all atoms stable
  - validity             fraction parseable as RDKit Mol
  - uniqueness           fraction of distinct canonical SMILES
  - valid_and_unique     joint
  - novelty              fraction of valid SMILES NOT in the training set

The metrics use RDKit and the EDM/EBMol valence convention. We
intentionally do not depend on OpenBabel because OpenBabel is not
installed in the production envs/flowmol environment; RDKit-only bond
inference (``Chem.rdmolops.SanitizeMol``) is sufficient to compute
all six numbers above.

Input JSON shape (one of):

  1. ``scripts/sample_bgfm.py`` output:
     {"samples": [{"atomic_numbers": [...], "atom_types": [...],
                   "positions": [[x,y,z], ...], "charge": int, "spin": int}],
      "accounting": {...}, "config": {...}}

  2. A bare list of sample dicts in the same per-sample shape.

Optional novelty input: ``--train_smiles_file`` is a newline-delimited
file of canonical SMILES present in the training set; novelty is then
1 minus the fraction of generated SMILES that appear in that set.

Usage
-----

    python scripts/eval_qm9_ebmol_protocol.py \
        --samples runs/eval/qm9/samples.json \
        --out_csv runs/eval/qm9/qm9_metrics.csv \
        --train_smiles_file data/qm9_train_smiles.txt
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional


def _load_samples(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "samples" in payload:
        return payload["samples"]
    raise ValueError(f"unsupported samples JSON shape in {path}")


def _xyz_block(atomic_numbers: list[int], positions: list[list[float]],
               total_charge: int = 0) -> str:
    """Build an .xyz block usable by RDKit's XYZ -> Mol path."""
    try:
        from rdkit.Chem import GetPeriodicTable
        pt = GetPeriodicTable()
        symbols = [pt.GetElementSymbol(int(z)) for z in atomic_numbers]
    except Exception:
        # Fallback table for the elements QM9 covers.
        Z_TO_SYM = {1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F",
                    15: "P", 16: "S", 17: "Cl", 35: "Br", 53: "I"}
        symbols = [Z_TO_SYM.get(int(z), "X") for z in atomic_numbers]
    lines = [str(len(atomic_numbers)), f"charge={total_charge}"]
    for sym, (x, y, z) in zip(symbols, positions):
        lines.append(f"{sym} {x:.6f} {y:.6f} {z:.6f}")
    return "\n".join(lines) + "\n"


# EDM/EBMol valence convention used to define atom stability.
ALLOWED_VALENCES = {
    1: {1},                 # H
    5: {3},                 # B
    6: {4},                 # C
    7: {3, 5},              # N (sp3 and ammonium-like)
    8: {2},                 # O
    9: {1},                 # F
    14: {4},                # Si
    15: {3, 5},             # P
    16: {2, 4, 6},          # S
    17: {1},                # Cl
    35: {1},                # Br
    53: {1},                # I
}


def _atom_and_mol_stability(mol) -> tuple[float, bool]:
    """Return (atom-stable fraction, molecule-stable bool) for one Mol."""
    n = mol.GetNumAtoms()
    if n == 0:
        return 0.0, False
    n_stable = 0
    all_stable = True
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        # Valence = degree + implicit H. Use the explicit-valence convention.
        v = atom.GetExplicitValence() + atom.GetNumImplicitHs()
        if v in ALLOWED_VALENCES.get(z, set()):
            n_stable += 1
        else:
            all_stable = False
    return n_stable / n, all_stable


def _try_mol_from_xyz(xyz_block: str, total_charge: int = 0):
    """Best-effort RDKit construction from xyz coordinates only."""
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds
    except Exception:
        return None
    try:
        mol = Chem.MolFromXYZBlock(xyz_block)
        if mol is None:
            return None
        rdDetermineBonds.DetermineBonds(mol, charge=int(total_charge))
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def _canonical_smiles(mol) -> Optional[str]:
    if mol is None:
        return None
    try:
        from rdkit import Chem
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    except Exception:
        return None


def evaluate_samples(samples: list[dict],
                     train_smiles: Optional[set[str]] = None) -> dict:
    n_total = len(samples)
    if n_total == 0:
        return {"n_samples": 0}
    atom_stab_sum = 0.0
    n_mol_stable = 0
    n_valid = 0
    smiles_list: list[str] = []
    for s in samples:
        atomic_numbers = s.get("atomic_numbers") or []
        positions = s.get("positions") or []
        if not atomic_numbers or not positions or len(atomic_numbers) != len(positions):
            continue
        charge = int(s.get("charge", 0))
        xyz = _xyz_block(atomic_numbers, positions, total_charge=charge)
        mol = _try_mol_from_xyz(xyz, total_charge=charge)
        if mol is None:
            continue
        atom_stab, mol_stab = _atom_and_mol_stability(mol)
        atom_stab_sum += atom_stab
        if mol_stab:
            n_mol_stable += 1
        smi = _canonical_smiles(mol)
        if smi is not None:
            n_valid += 1
            smiles_list.append(smi)

    unique_smiles = set(smiles_list)
    n_unique = len(unique_smiles)
    valid_and_unique = n_unique / n_total
    if train_smiles is not None and n_valid > 0:
        n_novel = sum(1 for s in unique_smiles if s not in train_smiles)
        novelty = n_novel / n_unique if n_unique else 0.0
    else:
        novelty = float("nan")

    return {
        "n_samples": n_total,
        "atom_stability": atom_stab_sum / n_total,
        "molecule_stability": n_mol_stable / n_total,
        "validity": n_valid / n_total,
        "uniqueness": (n_unique / n_valid) if n_valid else 0.0,
        "valid_and_unique": valid_and_unique,
        "novelty": novelty,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path, required=True,
                    help="JSON from scripts/sample_bgfm.py or a bare sample list.")
    ap.add_argument("--out_csv", type=Path, required=True)
    ap.add_argument("--train_smiles_file", type=Path, default=None,
                    help="Newline-delimited canonical SMILES of the training set.")
    args = ap.parse_args()

    samples = _load_samples(args.samples)
    train_smiles: Optional[set[str]] = None
    if args.train_smiles_file is not None and args.train_smiles_file.exists():
        train_smiles = {ln.strip() for ln in args.train_smiles_file.read_text().splitlines() if ln.strip()}

    metrics = evaluate_samples(samples, train_smiles)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(metrics.keys()))
        writer.writerow([metrics[k] for k in metrics.keys()])
    print(f"[qm9] wrote {args.out_csv}: {metrics}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
