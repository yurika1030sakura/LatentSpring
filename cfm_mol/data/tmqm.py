"""tmQM transition-metal complex loader (ICLR E1 OOD slice #1).

Source: Balcells, D.; Skjelstad, B. B. tmQM: The Quantum Mechanical
Properties of Transition Metal Complexes. J. Chem. Inf. Model. 60,
6135-6146 (2020). 2024 re-release: 108k complexes covering all 3d/4d/5d
transition metals.

Download: `git clone https://github.com/uiocompcat/tmQM`. The repo layout
is:
    tmQM/tmQM_X{1,2,3}.xyz.gz   -- Cartesian coords for 108k complexes
                                    (multi-molecule concatenated format)
    tmQM/tmQM_X{1,2,3}.BO.gz    -- Wiberg bond orders + atomic valences
    tmQM/tmQM_y.csv             -- DFT properties + SMILES
    tmQM/tmQM_X.q               -- natural atomic charges

We parse xyz + BO together so bonds come from actual quantum-chemistry
Wiberg indices (floats), not a covalent-radius heuristic. Bond-order
discretisation:
    BO < 0.3      -> no bond
    0.3 <= BO < 1.3 -> single (order 1)
    1.3 <= BO < 2.3 -> double (order 2)
    2.3 <= BO       -> triple (order 3)

The 0.3 threshold captures M-L coordination bonds which often sit at
BO ~ 0.3-0.7 in transition-metal complexes.

Two slice modes:
  - broad  (default): {Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Mo, Ru, Rh,
                       Pd, Ir, Pt} -- common 3d/4d/5d metals, INCLUDES
                       Cu/Fe/Zn which are the most abundant in tmQM.
                       This is the primary E1 OOD slice.
  - narrow: {Pd, Ni, Rh, Ir} -- catalysis-focused sub-slice.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np


NARROW_METALS = {"Pd", "Ni", "Rh", "Ir"}
BROAD_METALS = {
    # First row (3d).
    "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    # Second row (4d).
    "Mo", "Ru", "Rh", "Pd",
    # Third row (5d).
    "Ir", "Pt",
}

SLICE_NAMES = {
    "narrow": "tmqm_catalysis4",
    "broad":  "tmqm_broad15",
}

NARROW_ATOM_MAP = [
    "C", "H", "N", "O", "F", "P", "S", "Cl", "Br", "I",
    "Pd", "Ni", "Rh", "Ir",
]
BROAD_ATOM_MAP = [
    "C", "H", "N", "O", "F", "P", "S", "Cl", "Br", "I",
    "Si", "As", "Se", "B",   # tmQM allows B/Si/As/Se ligands per README
    "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Mo", "Ru", "Rh", "Pd", "Ir", "Pt",
]

ATOM_MAP = BROAD_ATOM_MAP
SLICE_NAME = SLICE_NAMES["broad"]
_MAX_ATOMS = 80
_BO_THRESHOLD = 0.3


def download(dest_dir: Path) -> None:
    """Git-clone the tmQM repo into `dest_dir / 'tmqm_raw/'`."""
    import subprocess
    raw = dest_dir / "tmqm_raw"
    if (raw / "tmQM" / "tmQM_X1.xyz.gz").exists():
        print(f"tmqm_raw already cloned at {raw}; skipping.")
        return
    raw.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([
        "git", "clone", "--depth", "1",
        "https://github.com/uiocompcat/tmQM.git", str(raw),
    ])


# ---------------------------------------------------------------------------
# Multi-molecule xyz parser
# ---------------------------------------------------------------------------

def _iter_xyz_molecules(path: Path):
    """Yield (header, symbols, coords) for each molecule in a multi-molecule
    xyz.gz archive."""
    with gzip.open(path, "rt") as f:
        while True:
            line = f.readline()
            if not line:
                return
            line = line.strip()
            if not line:
                continue
            try:
                n = int(line)
            except ValueError:
                continue
            header = f.readline().rstrip("\n")
            symbols = []
            coords = np.zeros((n, 3), dtype=np.float32)
            for i in range(n):
                parts = f.readline().split()
                if len(parts) < 4:
                    break
                symbols.append(parts[0])
                coords[i] = [float(x) for x in parts[1:4]]
            yield header, symbols, coords


def _parse_csd_code(header: str) -> str | None:
    """Extract `WELROW` from `CSD_code = WELROW | ...`."""
    for part in header.split("|"):
        part = part.strip()
        if part.startswith("CSD_code"):
            return part.split("=", 1)[1].strip()
    return None


# ---------------------------------------------------------------------------
# Multi-molecule BO parser
# ---------------------------------------------------------------------------

def _iter_bo_molecules(path: Path):
    """Yield (csd_code, bond_dict) for each molecule in a .BO.gz file.

    bond_dict maps (i, j) with i<j (0-indexed) to Wiberg bond order (float).
    """
    with gzip.open(path, "rt") as f:
        csd_code = None
        bond_dict: dict[tuple[int, int], float] = {}
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if csd_code is not None:
                    yield csd_code, bond_dict
                csd_code = None
                bond_dict = {}
                continue
            if line.startswith("CSD_code") or line.lstrip().startswith("CSD_code"):
                csd_code = _parse_csd_code(line)
                bond_dict = {}
                continue
            parts = line.split()
            if not parts:
                continue
            try:
                atom_idx = int(parts[0]) - 1   # 1-indexed -> 0-indexed
            except ValueError:
                continue
            # parts[1] = symbol, parts[2] = total valence (float)
            # remaining triples: (symbol, neighbour_1indexed, bond_order_float)
            rest = parts[3:]
            for k in range(0, len(rest), 3):
                if k + 2 >= len(rest):
                    break
                try:
                    neighbour_idx = int(rest[k + 1]) - 1
                    bo = float(rest[k + 2])
                except ValueError:
                    continue
                i, j = sorted((atom_idx, neighbour_idx))
                if i == j:
                    continue
                # Keep the LARGER BO if we see the bond from both sides.
                key = (i, j)
                bond_dict[key] = max(bond_dict.get(key, 0.0), bo)
        if csd_code is not None:
            yield csd_code, bond_dict


def _bond_order_to_integer(bo: float) -> int:
    if bo < _BO_THRESHOLD:
        return 0
    if bo < 1.3:
        return 1
    if bo < 2.3:
        return 2
    return 3


def _bonds_from_wiberg(bond_dict: dict[tuple[int, int], float]) -> tuple[np.ndarray, np.ndarray]:
    edges = []
    orders = []
    for (i, j), bo in bond_dict.items():
        order = _bond_order_to_integer(bo)
        if order == 0:
            continue
        edges.append([i, j])
        orders.append(order)
    if not edges:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.int32)
    return np.asarray(edges, dtype=np.int64), np.asarray(orders, dtype=np.int32)


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def _has_target_metal(symbols: list[str], metal_set: set[str]) -> tuple[bool, str | None]:
    for s in symbols:
        if s in metal_set:
            return True, s
    return False, None


def process(
    raw_dir: Path,
    processed_dir: Path,
    atom_map: list[str] | None = None,
    max_atoms: int = _MAX_ATOMS,
    mode: str = "broad",
    max_mols: int | None = 20_000,
    strip_metal: bool = False,
) -> None:
    """Convert tmQM raw archives into FlowMol3 processed format.

    `mode`:
      - 'broad': BROAD_METALS + BROAD_ATOM_MAP. Includes Cu/Fe/Zn. Primary.
      - 'narrow': NARROW_METALS + NARROW_ATOM_MAP. Pd/Ni/Rh/Ir only.

    `max_mols` caps the OOD eval set size. None = keep all.

    `strip_metal=True` removes the metal atom (and its bonds) from each
    complex before writing. The result is a LIGAND-FRAMEWORK dataset in
    the 10-organic atom_map {C, H, N, O, F, P, S, Cl, Br, I}, suitable
    for evaluating a GEOM-trained model on "unusual organic bonding
    patterns induced by metal coordination, observed in the absence of
    the metal". This gives atom-map parity with GEOM-trained models
    while still probing the OOD regime. Per-complex fragments that have
    all-non-organic atoms are dropped; complexes where stripping the
    metal produces a disconnected graph are KEPT (our
    project_connectivity handles them at sample time).
    """
    ORGANIC_ATOMS = ("C", "H", "N", "O", "F", "P", "S", "Cl", "Br", "I")
    import torch
    from torch.nn.functional import one_hot

    if mode == "broad":
        metal_set = BROAD_METALS
        if atom_map is None:
            atom_map = BROAD_ATOM_MAP if not strip_metal else list(ORGANIC_ATOMS)
    elif mode == "narrow":
        metal_set = NARROW_METALS
        if atom_map is None:
            atom_map = NARROW_ATOM_MAP if not strip_metal else list(ORGANIC_ATOMS)
    else:
        raise ValueError(f"mode must be 'broad' or 'narrow'; got {mode!r}")

    if strip_metal:
        print(f"[strip_metal] atom_map restricted to 10-organic set: {atom_map}")

    atom_to_idx = {s: i for i, s in enumerate(atom_map)}
    n_atom_types = len(atom_map)

    tmqm_dir = raw_dir / "tmqm_raw" / "tmQM"
    if not tmqm_dir.exists():
        raise FileNotFoundError(
            f"{tmqm_dir} not found. Run tmqm.download(raw_dir) first."
        )

    xyz_files = sorted(tmqm_dir.glob("tmQM_X*.xyz.gz"))
    bo_files = sorted(tmqm_dir.glob("tmQM_X*.BO.gz"))
    if len(xyz_files) != len(bo_files):
        raise RuntimeError(
            f"mismatched xyz/BO files: {len(xyz_files)} vs {len(bo_files)}"
        )
    print(f"parsing {len(xyz_files)} xyz archives (mode={mode}) ...")

    positions_list, atom_types_list, charges_list = [], [], []
    bond_types_list, bond_idxs_list = [], []
    node_idx_array, edge_idx_array = [], []
    n_nodes, n_edges, kept = 0, 0, 0
    per_metal_counts: dict[str, int] = {}

    for xyz_path, bo_path in zip(xyz_files, bo_files):
        # Build csd_code -> bonds map for this archive.
        bo_map = {code: bonds for code, bonds in _iter_bo_molecules(bo_path)}

        for header, symbols, coords in _iter_xyz_molecules(xyz_path):
            if max_mols is not None and kept >= max_mols:
                break
            if len(symbols) > max_atoms:
                continue
            has_m, metal = _has_target_metal(symbols, metal_set)
            if not has_m:
                continue
            # If NOT stripping, enforce atom_map up-front (tmQM has some rare
            # metals we don't track even in broad mode). If stripping, defer
            # the atom_map check until AFTER the metal has been removed.
            if not strip_metal:
                if any(s not in atom_to_idx for s in symbols):
                    continue
            else:
                # Non-metal atoms must be in the organic atom_map; metal
                # atoms are tolerated here and removed below.
                if any((s not in atom_to_idx) and (s not in metal_set)
                       for s in symbols):
                    continue

            csd_code = _parse_csd_code(header)
            if csd_code is None or csd_code not in bo_map:
                continue

            bond_idxs, bond_types = _bonds_from_wiberg(bo_map[csd_code])
            if bond_idxs.shape[0] == 0:
                continue

            # Strip metal: drop metal atoms + all incident bonds, keep only
            # the ligand framework. Reindex the remaining atoms + bonds.
            if strip_metal:
                keep = np.array([s in ORGANIC_ATOMS for s in symbols])
                if not keep.any():
                    continue
                old_to_new = -np.ones(len(symbols), dtype=np.int64)
                old_to_new[keep] = np.arange(keep.sum())
                symbols = [symbols[i] for i in range(len(symbols)) if keep[i]]
                coords = coords[keep]
                # Drop bonds touching removed atoms; reindex the rest.
                ok_bond = np.array([
                    (keep[i] and keep[j]) for (i, j) in bond_idxs
                ])
                bond_idxs = bond_idxs[ok_bond]
                if bond_idxs.shape[0] == 0:
                    continue
                bond_idxs = np.stack(
                    [old_to_new[bond_idxs[:, 0]], old_to_new[bond_idxs[:, 1]]],
                    axis=1,
                )
                bond_types = bond_types[ok_bond]
                # After stripping, ALL symbols should be in atom_map already
                # (organic set), but double-check.
                if any(s not in atom_to_idx for s in symbols):
                    continue

            a = np.array([atom_to_idx[s] for s in symbols], dtype=np.int64)
            positions_list.append(torch.tensor(coords, dtype=torch.float32))
            atom_types_list.append(
                one_hot(torch.tensor(a), num_classes=n_atom_types).float()
            )
            charges_list.append(
                one_hot(torch.zeros(len(symbols), dtype=torch.long) + 2,
                        num_classes=6).float()
            )
            bond_types_list.append(torch.tensor(bond_types, dtype=torch.int32))
            bond_idxs_list.append(torch.tensor(bond_idxs, dtype=torch.int64))
            node_idx_array.append([n_nodes, n_nodes + len(symbols)])
            edge_idx_array.append([n_edges, n_edges + bond_idxs.shape[0]])
            n_nodes += len(symbols)
            n_edges += bond_idxs.shape[0]
            kept += 1
            per_metal_counts[metal] = per_metal_counts.get(metal, 0) + 1
        if max_mols is not None and kept >= max_mols:
            break

    if kept == 0:
        raise RuntimeError("No tmQM molecules survived filtering.")

    print(f"kept {kept} complexes, {n_nodes} atoms, {n_edges} bonds "
          f"(real Wiberg bonds, not heuristic)")
    print("per-metal breakdown:")
    for m, c in sorted(per_metal_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {m:3s}: {c}")

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
    print(f"wrote processed tmQM data to {processed_dir}")


def main_cli() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=Path, required=True)
    ap.add_argument("--processed_dir", type=Path, required=True)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--mode", choices=["broad", "narrow"], default="broad")
    ap.add_argument("--max_mols", type=int, default=20_000)
    ap.add_argument("--strip_metal", action="store_true",
                    help="Remove metal atoms + incident bonds; emit a "
                         "ligand-framework-only dataset in the 10-organic "
                         "atom map. Use for atom-map-compatible OOD eval "
                         "against a GEOM-trained model.")
    args = ap.parse_args()
    if args.download:
        download(args.raw_dir)
    process(args.raw_dir, args.processed_dir, mode=args.mode,
            max_mols=args.max_mols, strip_metal=args.strip_metal)


if __name__ == "__main__":
    main_cli()
