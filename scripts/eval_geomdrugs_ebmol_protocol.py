"""GEOM-Drugs generation evaluation under the EBMol revised protocol (Q1).

Consumes the JSON produced by ``scripts/sample_bgfm.py`` and reports
the EBMol GEOM-Drugs metrics on 10k generated molecules:

  - atom_stability, molecule_stability, validity, uniqueness, novelty,
    valid_and_unique  (same definitions as the QM9 protocol; the
    valence convention differs --- see ALLOWED_VALENCES_GEOM below)
  - vendi_diversity   Vendi-score diversity on the valid-connected
                      subset, using Morgan fingerprints + Tanimoto.

The revised protocol additionally:

  - Treats samples that fail RDKit sanitization as invalid (rather
    than silently dropped).
  - Reports the all-samples version of validity (no aggressive
    filtering) alongside the valid-connected-only version.

Vendi diversity follows Friedman & Dieng (TMLR 2023): given a kernel
matrix K_ij over n samples, Vendi(K) = exp(H(K / n)) where H is the
Shannon entropy of the eigenvalues of K / n.

Usage
-----

    python scripts/eval_geomdrugs_ebmol_protocol.py \
        --samples runs/eval/geomdrugs/samples.json \
        --out_csv runs/eval/geomdrugs/geom_metrics.csv \
        --train_smiles_file data/geomdrugs_train_smiles.txt \
        --vendi_subset 1000
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Optional

# Reuse QM9 helpers.
from scripts.eval_qm9_ebmol_protocol import (
    _load_samples, _xyz_block, _try_mol_from_xyz, _canonical_smiles,
    _atom_and_mol_stability,
)


# Revised GEOM-Drugs valence convention. Drug-like molecules cover more
# elements than QM9; we accept any of the textbook valences below.
ALLOWED_VALENCES_GEOM = {
    1: {1}, 5: {3}, 6: {4}, 7: {3, 5}, 8: {2}, 9: {1},
    14: {4}, 15: {3, 5}, 16: {2, 4, 6},
    17: {1}, 35: {1}, 53: {1},
}


def _vendi_diversity(smiles: list[str], radius: int = 2, n_bits: int = 2048,
                     subset: Optional[int] = None) -> float:
    """Vendi-score diversity over Morgan-fingerprint Tanimoto kernel.

    A pure-numpy implementation that avoids extra dependencies.
    Returns NaN if RDKit/numpy are missing or the input is empty.
    """
    try:
        import numpy as np
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs
    except Exception:
        return float("nan")
    if not smiles:
        return float("nan")
    if subset is not None and subset > 0 and len(smiles) > subset:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(smiles), size=subset, replace=False)
        smiles = [smiles[i] for i in idx]
    fps = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=n_bits))
    n = len(fps)
    if n == 0:
        return float("nan")
    K = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            K[i, j] = K[j, i] = sim
    # Vendi: exp(Shannon entropy of eigenvalues of K/n).
    eigs = np.linalg.eigvalsh(K / n)
    eigs = np.clip(eigs, 1e-12, None)
    eigs = eigs / eigs.sum()
    entropy = float(-(eigs * np.log(eigs)).sum())
    return math.exp(entropy)


def _is_connected(mol) -> bool:
    """RDKit-only connectivity check (one fragment after sanitization)."""
    try:
        from rdkit.Chem import GetMolFrags
        return len(GetMolFrags(mol, asMols=False)) == 1
    except Exception:
        return False


def _atom_stability_geom(mol) -> tuple[float, bool]:
    """GEOM-Drugs valence convention variant of atom stability."""
    n = mol.GetNumAtoms()
    if n == 0:
        return 0.0, False
    n_stable = 0
    all_stable = True
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        v = atom.GetExplicitValence() + atom.GetNumImplicitHs()
        if v in ALLOWED_VALENCES_GEOM.get(z, set()):
            n_stable += 1
        else:
            all_stable = False
    return n_stable / n, all_stable


def evaluate_samples(samples: list[dict],
                     train_smiles: Optional[set[str]] = None,
                     vendi_subset: Optional[int] = 1000) -> dict:
    n_total = len(samples)
    if n_total == 0:
        return {"n_samples": 0}
    atom_stab_sum = 0.0
    n_mol_stable = 0
    n_valid = 0
    n_valid_connected = 0
    smiles_list: list[str] = []
    smiles_connected: list[str] = []
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
        atom_stab, mol_stab = _atom_stability_geom(mol)
        atom_stab_sum += atom_stab
        if mol_stab:
            n_mol_stable += 1
        smi = _canonical_smiles(mol)
        if smi is None:
            continue
        n_valid += 1
        smiles_list.append(smi)
        if _is_connected(mol):
            n_valid_connected += 1
            smiles_connected.append(smi)

    unique_smiles = set(smiles_list)
    n_unique = len(unique_smiles)
    if train_smiles is not None and n_unique > 0:
        n_novel = sum(1 for s in unique_smiles if s not in train_smiles)
        novelty = n_novel / n_unique
    else:
        novelty = float("nan")

    vendi = _vendi_diversity(smiles_connected, subset=vendi_subset)

    return {
        "n_samples": n_total,
        "atom_stability": atom_stab_sum / n_total,
        "molecule_stability": n_mol_stable / n_total,
        "validity": n_valid / n_total,
        "valid_connected": n_valid_connected / n_total,
        "uniqueness": (n_unique / n_valid) if n_valid else 0.0,
        "valid_and_unique": n_unique / n_total,
        "novelty": novelty,
        "vendi_diversity": vendi,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--out_csv", type=Path, required=True)
    ap.add_argument("--train_smiles_file", type=Path, default=None)
    ap.add_argument("--vendi_subset", type=int, default=1000,
                    help="Cap Vendi computation at this many samples; O(N^2) Tanimoto kernel.")
    args = ap.parse_args()

    samples = _load_samples(args.samples)
    train_smiles = None
    if args.train_smiles_file is not None and args.train_smiles_file.exists():
        train_smiles = {ln.strip() for ln in args.train_smiles_file.read_text().splitlines() if ln.strip()}
    metrics = evaluate_samples(samples, train_smiles=train_smiles,
                               vendi_subset=args.vendi_subset)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(metrics.keys()))
        writer.writerow([metrics[k] for k in metrics.keys()])
    print(f"[geomdrugs] wrote {args.out_csv}: {metrics}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
