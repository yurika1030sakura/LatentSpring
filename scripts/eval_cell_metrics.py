"""Stage 1 of uniform 5-cell ablation evaluation (in `envs/flowmol`).

Samples N molecules from a checkpoint, computes connectivity / fragment
metrics (bond-free safe), and exports geometry to JSON for Stage 2
(OMol25 force/energy/relax in `envs/omol25`).

Output:
  <out_dir>/<cell_id>_samples.json     # for Stage 2 (omol25 env)
  <out_dir>/<cell_id>_stage1.csv       # connectivity/fragmentation/validity

Stage 2 produces:
  <out_dir>/<cell_id>_stage2_omol25.csv  # max_force, energy, relax_steps

Final merge (uniform schema across all cells):
  <out_dir>/<cell_id>_eval.csv

Usage:
  conda activate envs/flowmol
  python scripts/eval_cell_metrics.py \\
      --cell_id 4d \\
      --checkpoint runs/omol25_4m_bgfm/lightning_logs/version_X/checkpoints/last.ckpt \\
      --config configs/omol25_4m_bgfm.yaml \\
      --eval_data /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/data/qm9_processed \\
      --n_samples 500 \\
      --out_dir runs/eval/cell_4d
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


_PERIODIC_SYMBOLS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt",
    "Au", "Hg", "Tl", "Pb", "Bi",
]
SYMBOL_TO_Z = {sym: idx + 1 for idx, sym in enumerate(_PERIODIC_SYMBOLS)}


def _connectivity_from_geometry(positions: np.ndarray, atomic_numbers: np.ndarray,
                                 cov_mult: float = 1.2) -> tuple[int, list[int]]:
    """Compute connected components on covalent-radius graph.

    Returns (n_components, component_sizes_sorted_desc).
    Bond-free safe — uses only positions + Z.
    """
    from ase import Atoms
    from ase.neighborlist import natural_cutoffs, neighbor_list
    atoms = Atoms(numbers=atomic_numbers, positions=positions)
    cutoffs = natural_cutoffs(atoms, mult=cov_mult)
    i, j = neighbor_list("ij", atoms, cutoffs)
    n = len(atomic_numbers)
    parent = list(range(n))
    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def _union(x, y):
        rx, ry = _find(x), _find(y)
        if rx != ry:
            parent[rx] = ry
    for a, b in zip(i, j):
        _union(int(a), int(b))
    comps = {}
    for k in range(n):
        r = _find(k)
        comps[r] = comps.get(r, 0) + 1
    sizes = sorted(comps.values(), reverse=True)
    return len(sizes), sizes


def _try_xyz2mol(positions: np.ndarray, atomic_numbers: np.ndarray,
                 charge: int = 0) -> tuple[bool, str | None]:
    """Try post-hoc bond perception via RDKit DetermineBonds. Returns
    (valid, smiles) where valid means RDKit accepted the structure."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.RWMol()
        for z in atomic_numbers:
            mol.AddAtom(Chem.Atom(int(z)))
        conf = Chem.Conformer(len(atomic_numbers))
        for i, p in enumerate(positions):
            conf.SetAtomPosition(i, (float(p[0]), float(p[1]), float(p[2])))
        mol.AddConformer(conf)
        # Attempt bond determination; fail-fast on errors.
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
        try:
            Chem.rdDetermineBonds.DetermineBonds(mol, charge=charge)
        except Exception:
            return False, None
        Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(mol)
        return True, smiles
    except Exception:
        return False, None


def _patch_model_from_config(cfg: dict, model, atom_map: list[str], no_patch: bool):
    """Apply cfm_mol patches if cfg requests them."""
    if no_patch:
        return
    from cfm_mol.domain import default_d_min_table
    from cfm_mol.flow_model import patch_flowmol
    n_real = len(atom_map)
    has_fake = cfg["mol_fm"].get("fake_atom_p", 0.0) > 0
    has_mask = cfg["mol_fm"].get("parameterization", "") == "ctmc"
    n_total = n_real + int(has_fake) + int(has_mask)
    d_min = torch.zeros(n_total, n_total)
    d_min[:n_real, :n_real] = default_d_min_table(n_atom_types=n_real, atom_map=atom_map)
    e_weight = cfg["mol_fm"].get("total_loss_weights", {}).get("e", 2.0)
    bond_free = float(e_weight) == 0.0
    patch_flowmol(model, d_min,
                  tangent=True, retract=True, gluing=True,
                  discrete_projection=not bond_free,
                  train_time_discrete=not bond_free,
                  atom_map=atom_map)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell_id", required=True,
                    help="Identifier for this ablation cell (e.g. '4d', 'B', '4a').")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--eval_data", type=Path, required=True,
                    help="Reference data dir to sample size distribution from.")
    ap.add_argument("--n_samples", type=int, default=500)
    ap.add_argument("--n_timesteps", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--no_patch", action="store_true",
                    help="Skip cfm_mol patch (raw FlowMol3 / Cell 0).")
    ap.add_argument("--charge", type=int, default=0,
                    help="Total charge for xyz2mol bond perception.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[stage1] cell_id={args.cell_id}  device={device}")
    print(f"[stage1] checkpoint={args.checkpoint}")
    print(f"[stage1] config={args.config}")

    # Build model + load checkpoint
    from flowmol.model_utils.load import (
        data_module_from_config, model_from_config, read_config_file,
    )
    cfg = read_config_file(args.config)
    cfg.get("mol_fm", {}).pop("bgfm", None)  # avoid FlowMol __init__ TypeError
    cfg["dataset"]["processed_data_dir"] = str(args.eval_data)

    atom_map = cfg["dataset"]["atom_map"]
    model = model_from_config(cfg)
    _patch_model_from_config(cfg, model, atom_map, no_patch=args.no_patch)
    state = torch.load(str(args.checkpoint), map_location="cpu")
    state_dict = state.get("state_dict", state)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[stage1] loaded ckpt: missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(device).eval()

    # Sample
    print(f"[stage1] sampling {args.n_samples} molecules ...")
    samples = []
    n_done = 0
    while n_done < args.n_samples:
        n_take = min(args.batch_size, args.n_samples - n_done)
        with torch.no_grad():
            mols = model.sample_random_sizes(
                n_molecules=n_take, n_timesteps=args.n_timesteps, device=device,
            )
        for m in mols:
            # FlowMol3's SampledMolecule.atom_types is a list of element-symbol
            # strings (already argmax-decoded in extract_moldata_from_graph).
            # Older variants stored a one-hot tensor; handle both robustly.
            at = m.atom_types
            if hasattr(at, "argmax"):
                sym = [atom_map[i] for i in at.argmax(dim=-1).cpu().numpy()]
            else:
                sym = list(at)
            z = np.array([SYMBOL_TO_Z[s] for s in sym], dtype=np.int64)
            pos = m.positions.detach().cpu().numpy().astype(np.float64)
            samples.append({"atomic_numbers": z.tolist(),
                            "positions": pos.tolist(),
                            "charge": int(args.charge),
                            "spin": 1})
        n_done += n_take
        print(f"[stage1] {n_done}/{args.n_samples}")

    # Connectivity + xyz2mol metrics
    rows = []
    for idx, s in enumerate(samples):
        z = np.array(s["atomic_numbers"])
        pos = np.array(s["positions"])
        n_comp, comp_sizes = _connectivity_from_geometry(pos, z)
        connected = (n_comp == 1)
        valid, smiles = _try_xyz2mol(pos, z, charge=int(args.charge))
        rows.append({
            "cell_id": args.cell_id,
            "mol_idx": idx,
            "n_atoms": int(len(z)),
            "n_components": int(n_comp),
            "largest_frag_size": int(comp_sizes[0]) if comp_sizes else 0,
            "largest_frag_frac": float(comp_sizes[0] / max(1, len(z))) if comp_sizes else 0.0,
            "connected": bool(connected),
            "rdkit_valid": bool(valid),
            "smiles": smiles or "",
        })

    # Save outputs
    json_path = args.out_dir / f"{args.cell_id}_samples.json"
    with open(json_path, "w") as f:
        json.dump(samples, f)
    csv_path = args.out_dir / f"{args.cell_id}_stage1.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    n_valid = sum(1 for r in rows if r["rdkit_valid"])
    n_conn = sum(1 for r in rows if r["connected"])
    print(f"\n[stage1 summary]")
    print(f"  n_samples            : {len(rows)}")
    print(f"  rdkit valid          : {n_valid}/{len(rows)} = {n_valid/max(1,len(rows)):.3f}")
    print(f"  connected (1 comp)   : {n_conn}/{len(rows)} = {n_conn/max(1,len(rows)):.3f}")
    print(f"  avg n_components     : {np.mean([r['n_components'] for r in rows]):.2f}")
    print(f"  avg largest frag frac: {np.mean([r['largest_frag_frac'] for r in rows]):.3f}")
    print(f"\n[stage1] wrote {json_path}")
    print(f"[stage1] wrote {csv_path}")
    print(f"[stage1] next: scripts/eval_cell_omol25_stage2.py --samples_json {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
