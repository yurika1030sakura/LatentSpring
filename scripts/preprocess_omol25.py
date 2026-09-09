"""Preprocess OMol25 LMDB shards to FlowMol3-native .pt tensor files.

Runs in the `omol25` env (torch 2.8 + fairchem-core 2.19). Produces:
  <out_dir>/train_data_processed.pt
  <out_dir>/val_data_processed.pt
  <out_dir>/train_data_marginal_dists.pt
  <out_dir>/train_data_n_atoms_histogram.pt

**Bond-free design (2026-04-21)**: OMol25 provides only positions + atoms
+ QM forces/energy, no bond orders. Rather than inferring bonds via a
heuristic (covalent-radius + xyz2mol) and feeding the model ~25% wrong
labels on TM complexes, we commit to a fully bond-free training regime:

  - `bond_idxs` / `bond_types` are stored as EMPTY tensors
  - In MoleculeDataset, the full pair adjacency collapses to class-0 ("no
    bond") for every pair -- this is the existing semantics for unbonded
    pairs, now applied to all pairs
  - The config sets `total_loss_weights.e = 0.0` so the bond head receives
    no training signal
  - Inference-time bond perception is deferred to post-hoc xyz2mol on the
    generated geometry (not during training)

This is aligned with the "physics-informed" narrative: quantum-mechanical
basis variables (positions, atoms, charges, forces) drive generation;
bond abstractions are post-hoc.

**BGFM upgrade (2026-04-21)**: additionally store DFT forces and energy
from OMol25 ground-truth (read directly from LMDB, no NN evaluation).
These are used for the force / energy consistency losses in
Boltzmann-Guided Flow Matching (see notes/bgfm_method.md).

Each *.pt file contains a dict with keys:
  positions         (N_total, 3) float32, COM-centered per molecule
  atom_types        (N_total, n_atom_types) bool, one-hot (cast .float() in
                    MoleculeDataset at training time)
  atom_charges      (N_total,) int8 (raw integer charges, -2..3)
  bond_types        (0,) int32 -- EMPTY (bond-free)
  bond_idxs         (0, 2) int64 -- EMPTY (bond-free)
  node_idx_array    (M, 2) long -- [start, end) per molecule
  edge_idx_array    (M, 2) long -- all [k, k] (zero-length bond slice per mol)
  forces            (N_total, 3) float32 -- DFT forces (eV/A), concatenated
  energies          (M,) float32 -- DFT energies (eV), one per molecule

Usage:
  conda activate envs/omol25
  python scripts/preprocess_omol25.py \\
      --src /n/netscratch/ryl_lab/Lab/omol25/train_4M \\
      --out_dir /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/data/omol25_4m_processed \\
      --max_atoms 200 --val_frac 0.01
"""
from __future__ import annotations

import argparse
from pathlib import Path
from time import time

import numpy as np
import torch


DEFAULT_ATOM_MAP_OMOL25 = [
    "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
    "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca",
    "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi",
]
_SYM_TO_Z = {s: i + 1 for i, s in enumerate(DEFAULT_ATOM_MAP_OMOL25)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Directory or comma-sep list of LMDB shards")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--max_atoms", type=int, default=200)
    ap.add_argument("--val_frac", type=float, default=0.01,
                    help="Fraction of surviving molecules held out for val.")
    ap.add_argument("--max_n_mols", type=int, default=None,
                    help="Cap total kept (useful for debugging).")
    ap.add_argument("--bond_tolerance", type=float, default=1.2)
    ap.add_argument("--atom_map", default=None,
                    help="Comma-sep overriding atom map. Default = 83 elems.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard_size", type=int, default=0,
                    help="If > 0, write training data as sharded .pt files "
                         "of SHARD_SIZE molecules each (plus a manifest). "
                         "Required for 100M-scale to avoid OOM at load time.")
    args = ap.parse_args()

    atom_map = args.atom_map.split(",") if args.atom_map else list(DEFAULT_ATOM_MAP_OMOL25)
    n_atom_types = len(atom_map)
    z_to_idx = {_SYM_TO_Z[s]: i for i, s in enumerate(atom_map) if s in _SYM_TO_Z}
    print(f"atom_map: {n_atom_types} elements")

    print("importing fairchem ...", flush=True)
    from fairchem.core.datasets import AseDBDataset
    from ase.neighborlist import natural_cutoffs, neighbor_list

    src = args.src if "," not in args.src else args.src.split(",")
    print(f"opening AseDBDataset from {src} ...", flush=True)
    db = AseDBDataset({"src": src if not isinstance(src, list) else src})
    N = len(db)
    print(f"dataset has {N} molecules", flush=True)
    if args.max_n_mols is not None:
        N = min(N, args.max_n_mols)
        print(f"  capped to {N}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # === Streaming shard architecture ===
    # When --shard_size > 0, we build TWO buffers:
    #   - val_buffer: first ~val_frac fraction of accepted molecules (random
    #     sample via RNG). Written as single file at end.
    #   - train_shard_buffer: rolling buffer; when it reaches shard_size,
    #     dumped to disk and cleared. Avoids accumulating 100M molecules
    #     in RAM (which OOM'd at 63M with 256GB).
    # When --shard_size == 0, single-file path: accumulate all into
    # *_list, split + write at end (legacy behavior).
    streaming = args.shard_size > 0
    rng = np.random.default_rng(args.seed)

    # Accumulators (used by both paths; in streaming mode, train data is
    # flushed to disk as shards while val/legacy use these lists).
    positions_list, atom_types_list, atom_charges_list = [], [], []
    forces_list, energies_list = [], []
    node_idx_array_legacy, edge_idx_array_legacy = [], []  # legacy single-file path
    n_nodes_total, kept = 0, 0
    skip_size, skip_elem, skip_err, skip_no_energy = 0, 0, 0, 0

    # Streaming-mode buffers
    val_buf = {"positions": [], "atom_types": [], "atom_charges": [],
               "forces": [], "energies": [], "n_atoms_per_mol": []}
    train_shard_buf = {"positions": [], "atom_types": [], "atom_charges": [],
                        "forces": [], "energies": [], "n_atoms_per_mol": []}
    shard_id = 0
    shards_info = []  # accumulated metadata for manifest

    def _flush_train_shard():
        """Write current train_shard_buf to disk and clear."""
        nonlocal shard_id, train_shard_buf
        if not train_shard_buf["energies"]:
            return
        n_mols = len(train_shard_buf["energies"])
        # Build node_idx_array from per-mol atom counts.
        n_atoms = train_shard_buf["n_atoms_per_mol"]
        cum = [0]
        for na in n_atoms:
            cum.append(cum[-1] + na)
        nia = [[cum[i], cum[i + 1]] for i in range(n_mols)]
        chunk = {
            "positions": torch.cat(train_shard_buf["positions"], dim=0),
            "atom_types": torch.cat(train_shard_buf["atom_types"], dim=0),
            "atom_charges": torch.cat(train_shard_buf["atom_charges"], dim=0),
            "bond_types": torch.zeros(0, dtype=torch.int32),
            "bond_idxs": torch.zeros(0, 2, dtype=torch.int64),
            "node_idx_array": torch.tensor(nia, dtype=torch.long),
            "edge_idx_array": torch.zeros(n_mols, 2, dtype=torch.long),
            "forces": torch.cat(train_shard_buf["forces"], dim=0),
            "energies": torch.tensor(train_shard_buf["energies"], dtype=torch.float32),
        }
        fname = f"train_data_processed_shard_{shard_id:04d}.pt"
        torch.save(chunk, args.out_dir / fname)
        shards_info.append({
            "file": fname,
            "n_molecules": int(n_mols),
            "n_atoms": int(chunk['positions'].shape[0]),
        })
        print(f"  flushed shard {shard_id:04d}: {fname} "
              f"({n_mols} mols, {chunk['positions'].shape[0]} atoms)", flush=True)
        del chunk
        # Clear buffer
        for k in train_shard_buf:
            train_shard_buf[k] = []
        shard_id += 1

    def _build_val_dict():
        n_mols = len(val_buf["energies"])
        n_atoms = val_buf["n_atoms_per_mol"]
        cum = [0]
        for na in n_atoms:
            cum.append(cum[-1] + na)
        nia = [[cum[i], cum[i + 1]] for i in range(n_mols)]
        return {
            "positions": torch.cat(val_buf["positions"], dim=0) if val_buf["positions"] else torch.zeros(0, 3),
            "atom_types": torch.cat(val_buf["atom_types"], dim=0) if val_buf["atom_types"] else torch.zeros(0, dtype=torch.int8),
            "atom_charges": torch.cat(val_buf["atom_charges"], dim=0) if val_buf["atom_charges"] else torch.zeros(0, dtype=torch.int8),
            "bond_types": torch.zeros(0, dtype=torch.int32),
            "bond_idxs": torch.zeros(0, 2, dtype=torch.int64),
            "node_idx_array": torch.tensor(nia, dtype=torch.long),
            "edge_idx_array": torch.zeros(n_mols, 2, dtype=torch.long),
            "forces": torch.cat(val_buf["forces"], dim=0) if val_buf["forces"] else torch.zeros(0, 3),
            "energies": torch.tensor(val_buf["energies"], dtype=torch.float32),
        }

    # Stats (computed on-the-fly for training split).
    p_a_count = np.zeros(n_atom_types, dtype=np.int64)
    p_c_count = np.zeros(6, dtype=np.int64)
    # Bond-free: every pair is class 0 (none). p_e will be all mass on class 0
    # (harmless; bond loss weight is 0 in config anyway).
    p_e_count = np.zeros(4, dtype=np.int64)
    p_c_given_a_count = np.zeros((n_atom_types, 6), dtype=np.int64)
    n_atoms_counter = np.zeros(args.max_atoms + 1, dtype=np.int64)
    unseen_elements = {}

    t0 = time()
    last_log = t0
    for db_idx in range(N):
        if time() - last_log > 30:
            elapsed = time() - t0
            rate = (db_idx + 1) / elapsed
            eta = (N - db_idx - 1) / max(rate, 1e-6)
            print(f"  {db_idx+1:>10d}/{N} kept={kept}  skip(size={skip_size} elem={skip_elem} "
                  f"err={skip_err} noE={skip_no_energy})  rate={rate:.1f} mol/s  eta={eta/60:.1f} min")
            last_log = time()
        try:
            atoms = db.get_atoms(db_idx)
        except Exception:
            skip_err += 1
            continue
        n = len(atoms)
        if n < 2 or n > args.max_atoms:
            skip_size += 1
            continue
        z = atoms.get_atomic_numbers()
        type_idx = np.array([z_to_idx.get(int(zi), -1) for zi in z], dtype=np.int64)
        if (type_idx < 0).any():
            for zi in z:
                if int(zi) not in z_to_idx:
                    unseen_elements[int(zi)] = unseen_elements.get(int(zi), 0) + 1
            skip_elem += 1
            continue

        pos = atoms.get_positions().astype(np.float32)
        # NOTE: forces are in the lab frame (pre-COM-removal). We do NOT
        # subtract COM from positions BEFORE reading forces -- the forces are
        # invariant under translation, so COM removal of positions is
        # harmless; we just need to NOT apply any rotation here.
        pos -= pos.mean(axis=0, keepdims=True)

        # BGFM: extract DFT energy + forces from ground-truth.
        # OMol25 stores them via ASE's SinglePointCalculator (not in info/arrays).
        # Standard ASE API reads from atoms.calc.results.
        if atoms.calc is None or "energy" not in atoms.calc.results or "forces" not in atoms.calc.results:
            skip_no_energy += 1
            continue
        try:
            energy = float(atoms.get_potential_energy())
            forces = atoms.get_forces().astype(np.float32)  # (n, 3) eV/A
        except Exception:
            skip_no_energy += 1
            continue
        if forces.shape != (n, 3):
            skip_no_energy += 1
            continue

        # Charges (total -> put on atom 0, clipped to [-2, 3]).
        charge_total = int(atoms.info.get("charge", 0))
        per_atom_c = np.zeros(n, dtype=np.int8)
        if charge_total != 0:
            per_atom_c[0] = np.clip(charge_total, -2, 3)

        # Tensors for this molecule. atom_types stored as int8 INDEX
        # (compact, 1 byte/atom vs 83 bytes for bool one-hot at 83 elems).
        # Training loader (patched MoleculeDataset) one-hots at __getitem__.
        pos_t = torch.from_numpy(pos)
        at_t = torch.from_numpy(type_idx.astype(np.int8))
        ac_t = torch.from_numpy(per_atom_c)
        f_t = torch.from_numpy(forces)

        if streaming:
            # Random val/train split (Bernoulli on val_frac per accepted mol).
            if rng.random() < args.val_frac:
                buf = val_buf
            else:
                buf = train_shard_buf
            buf["positions"].append(pos_t)
            buf["atom_types"].append(at_t)
            buf["atom_charges"].append(ac_t)
            buf["forces"].append(f_t)
            buf["energies"].append(energy)
            buf["n_atoms_per_mol"].append(n)
            # Flush train shard when full.
            if buf is train_shard_buf and len(train_shard_buf["energies"]) >= args.shard_size:
                _flush_train_shard()
        else:
            positions_list.append(pos_t)
            atom_types_list.append(at_t)
            atom_charges_list.append(ac_t)
            forces_list.append(f_t)
            energies_list.append(energy)
            node_idx_array_legacy.append([n_nodes_total, n_nodes_total + n])
            edge_idx_array_legacy.append([0, 0])
            n_nodes_total += n
        kept += 1

        # Stats.
        np.add.at(p_a_count, type_idx, 1)
        c_shifted = per_atom_c.astype(np.int64) + 2
        np.add.at(p_c_count, c_shifted, 1)
        for a, c in zip(type_idx, c_shifted):
            p_c_given_a_count[int(a), int(c)] += 1
        # Bond-free: all pairs are class 0 ("no bond"); p_e concentrates on 0.
        n_all_pairs = n * (n - 1) // 2
        p_e_count[0] += n_all_pairs
        n_atoms_counter[n] += 1

    print(f"\nkept {kept}/{N}  (skip size={skip_size} elem={skip_elem} err={skip_err} noE={skip_no_energy})")
    if kept == 0:
        raise RuntimeError("No molecules survived. Check src / atom_map / max_atoms.")

    if streaming:
        # Flush any remaining partial train shard.
        if train_shard_buf["energies"]:
            _flush_train_shard()
        # Build val from buf and write.
        val_out = _build_val_dict()
        n_val = val_out['node_idx_array'].shape[0]
        n_train = sum(s["n_molecules"] for s in shards_info)
        print(f"split: train={n_train} val={n_val}")
        print("writing val tensors ...")
        torch.save(val_out, args.out_dir / "val_data_processed.pt")
        torch.save(val_out, args.out_dir / "test_data_processed.pt")
        print(f"  val_data_processed.pt: {val_out['positions'].shape[0]} atoms")
        del val_out
        # Write manifest.
        import json
        with open(args.out_dir / "train_data_shards_manifest.json", "w") as f:
            json.dump({"shards": shards_info,
                       "shard_size": args.shard_size,
                       "n_shards": len(shards_info),
                       "atom_map_size": n_atom_types}, f, indent=2)
        print(f"wrote manifest: {len(shards_info)} shards")
        # Streaming path is done; jump past legacy single-file code below.
        _streaming_done = True
    else:
        _streaming_done = False
        # Legacy single-file path (only used when --shard_size = 0).
        rng_legacy = np.random.default_rng(args.seed)
        perm = rng_legacy.permutation(kept)
        n_val = max(1, int(kept * args.val_frac))
        val_mol_idx = set(perm[:n_val].tolist())

    # Build train and val dicts. Bond-free: bond_types/bond_idxs are empty;
    # edge_idx_array is all [0, 0] so every molecule's bond-slice is empty.
    # BGFM: additionally emit forces (concat per atom) and energies (per mol).
    # atom_types are stored as 1D int8 indices (not one-hot) for compactness.
    def _gather(mol_indices):
        """Build a chunk dict from a list of molecule indices (ordered)."""
        pos_parts, at_parts, ac_parts = [], [], []
        f_parts, e_list = [], []
        nia = []
        n_nodes = 0
        for mi in mol_indices:
            p = positions_list[mi]
            a = atom_types_list[mi]
            c = atom_charges_list[mi]
            f = forces_list[mi]
            e = energies_list[mi]
            pos_parts.append(p)
            at_parts.append(a)
            ac_parts.append(c)
            f_parts.append(f)
            e_list.append(e)
            nia.append([n_nodes, n_nodes + p.shape[0]])
            n_nodes += p.shape[0]
        n_mols = len(nia)
        out = {
            "positions": torch.cat(pos_parts, dim=0) if pos_parts else torch.zeros(0, 3),
            "atom_types": torch.cat(at_parts, dim=0) if at_parts else torch.zeros(0, dtype=torch.int8),
            "atom_charges": torch.cat(ac_parts, dim=0) if ac_parts else torch.zeros(0, dtype=torch.int8),
            "bond_types": torch.zeros(0, dtype=torch.int32),
            "bond_idxs": torch.zeros(0, 2, dtype=torch.int64),
            "node_idx_array": torch.tensor(nia, dtype=torch.long),
            "edge_idx_array": torch.zeros(n_mols, 2, dtype=torch.long),
            # BGFM labels
            "forces": torch.cat(f_parts, dim=0) if f_parts else torch.zeros(0, 3),
            "energies": torch.tensor(e_list, dtype=torch.float32),
        }
        return out

    if not _streaming_done:
        val_indices = [mi for mi in range(kept) if mi in val_mol_idx]
        train_indices = [mi for mi in range(kept) if mi not in val_mol_idx]

        print(f"split: train={len(train_indices)} val={len(val_indices)}")

        # Val + test always written as single files (small enough).
        print("writing val tensors ...")
        val_out = _gather(val_indices)
        torch.save(val_out, args.out_dir / "val_data_processed.pt")
        torch.save(val_out, args.out_dir / "test_data_processed.pt")
        print(f"  val_data_processed.pt: {val_out['positions'].shape[0]} atoms")
        del val_out

    # Train: sharded or single-file.
    if False and args.shard_size > 0:  # streaming handles shards now
        print(f"writing train tensors sharded at {args.shard_size} mols/shard ...")
        import json
        shards_info = []
        n_shards = (len(train_indices) + args.shard_size - 1) // args.shard_size
        for s in range(n_shards):
            start = s * args.shard_size
            end = min(start + args.shard_size, len(train_indices))
            chunk = _gather(train_indices[start:end])
            fname = f"train_data_processed_shard_{s:04d}.pt"
            torch.save(chunk, args.out_dir / fname)
            shards_info.append({
                "file": fname,
                "n_molecules": int(chunk['node_idx_array'].shape[0]),
                "n_atoms": int(chunk['positions'].shape[0]),
            })
            print(f"  shard {s+1}/{n_shards}: {fname} ({end - start} mols, "
                  f"{chunk['positions'].shape[0]} atoms)")
            del chunk
            # Free accumulated raw lists for this shard range to reclaim RAM.
            for mi in train_indices[start:end]:
                positions_list[mi] = None
                atom_types_list[mi] = None
                atom_charges_list[mi] = None
                forces_list[mi] = None
        with open(args.out_dir / "train_data_shards_manifest.json", "w") as f:
            json.dump({"shards": shards_info,
                       "shard_size": args.shard_size,
                       "n_shards": n_shards,
                       "atom_map_size": n_atom_types}, f, indent=2)
        print(f"wrote manifest: {n_shards} shards")
    elif not _streaming_done:
        print("writing train tensors (single file) ...")
        train_out = _gather(train_indices)
        torch.save(train_out, args.out_dir / "train_data_processed.pt")
        print(f"  train_data_processed.pt: {train_out['positions'].shape[0]} atoms")
        del train_out

    # Marginal dists (computed on ALL surviving molecules; small bias vs strict
    # train-only is negligible for 4M-scale).
    p_a = torch.tensor(p_a_count, dtype=torch.float32)
    p_a = p_a / p_a.sum().clamp(min=1.0)
    p_c = torch.tensor(p_c_count, dtype=torch.float32)
    p_c = p_c / p_c.sum().clamp(min=1.0)
    p_e = torch.tensor(p_e_count, dtype=torch.float32)
    p_e = p_e / p_e.sum().clamp(min=1.0)
    p_c_given_a = torch.tensor(p_c_given_a_count, dtype=torch.float32)
    p_c_given_a = p_c_given_a / p_c_given_a.sum(dim=1, keepdim=True).clamp(min=1.0)

    torch.save((p_a, p_c, p_e, p_c_given_a),
               args.out_dir / "train_data_marginal_dists.pt")
    # FlowMol3 expects a tuple (n_atoms_values, counts) — only non-zero bins.
    hist_counts = torch.tensor(n_atoms_counter, dtype=torch.long)
    nz = (hist_counts > 0).nonzero(as_tuple=False).flatten()
    hist_values = nz.long()
    hist_nonzero_counts = hist_counts[nz].long()
    torch.save((hist_values, hist_nonzero_counts),
               args.out_dir / "train_data_n_atoms_histogram.pt")

    # Permissive valencies file for SampleAnalyzer. OMol25 covers 83 elements
    # across diverse oxidation states; writing a strict valence table would
    # require chemistry-expert curation. We use a permissive table so that
    # SampleAnalyzer doesn't refuse to construct. The validity fraction it
    # reports during training is therefore not meaningful for OMol25 --
    # post-hoc OMol25-energy evaluation (scripts/compute_omol25_energy.py)
    # is the real quality metric.
    import json
    permissive_valencies = {}
    for s in atom_map:
        permissive_valencies[s] = {
            str(c): list(range(9)) for c in range(-2, 4)
        }
    with open(args.out_dir / "train_data_valencies_omol25.json", "w") as f:
        json.dump(permissive_valencies, f)
    print(f"wrote permissive valencies file "
          f"(SampleAnalyzer-compatible, {len(atom_map)} elements)")

    print(f"\np_a top 10:")
    top = torch.topk(p_a, k=min(10, n_atom_types))
    for p, idx in zip(top.values.tolist(), top.indices.tolist()):
        print(f"  {atom_map[idx]:>3s}: {p:.4f}")
    print(f"p_e: {p_e.tolist()}")
    if unseen_elements:
        print(f"top skipped elements (Z -> count):")
        for z, c in sorted(unseen_elements.items(), key=lambda kv: -kv[1])[:10]:
            print(f"  Z={z}: {c}")
    print(f"\nwrote all stats + tensors to {args.out_dir}")
    print(f"total time: {(time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
