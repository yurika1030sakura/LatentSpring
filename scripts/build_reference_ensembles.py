"""Build GFN2-xTB reference conformer ensembles for the P1 *global ensemble* eval.

Motivation
----------
Every Boltzmann-fidelity number we have so far (per-group Pearson r between
log p_theta and -E/kT) is measured on a *single* tight cloud of Gaussian
perturbations around one data geometry.  That only probes the density
*within one basin*.  A model can rank geometries perfectly inside every basin
and still assign arbitrary relative mass *between* basins -- which is exactly
what a Boltzmann distribution is not allowed to do:

    p(x) = exp(-E(x)/kT) / Z    =>    log p(x) + E(x)/kT = -log Z   GLOBALLY,
    so the constant must be the SAME in every basin.

This script builds the reference side of that test: for a set of small
OMol25 validation molecules with genuine torsional flexibility, it enumerates
distinct conformational basins with RDKit ETKDG + GFN2-xTB geometry
optimisation, then clusters the optimised structures into basins by
heavy-atom best-RMSD.

Pipeline (option (c) of the three proposed; chosen because it does not depend
on xtb MD/metadynamics stability):
    1. select val molecules: n_atoms in range, organic elements only,
       single covalent fragment, no radicals, >= min_rot rotatable bonds
    2. RDKit bond perception from the OMol25 xyz (rdDetermineBonds)
    3. ETKDGv3 multi-conformer embedding (+ the OMol25 data geometry itself)
    4. GFN2-xTB `--opt` on every conformer  ->  (E, x*) local minima
    5. greedy clustering of the minima by heavy-atom GetBestRMS  ->  basins
    6. keep systems with >= min_basins basins and a non-trivial energy spread

Output (one JSON, plus per-system .npz geometry payloads):
    <out_dir>/ensembles.json     metadata + basin energies (small, human readable)
    <out_dir>/geoms.npz          all basin geometries (float32 arrays)

Run in envs/flowmol (needs rdkit + xtb on PATH).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HARTREE_TO_EV = 27.211386245988

_E_LINE = re.compile(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh")

ORGANIC = {"H", "C", "N", "O", "F", "S", "Cl", "Br", "P", "I"}


# --------------------------------------------------------------------------
# xtb helpers (kept self-contained so this script can run standalone)
# --------------------------------------------------------------------------
def _xtb_binary() -> str | None:
    return shutil.which("xtb")


def _write_xyz(path: Path, symbols, positions, comment="") -> None:
    with open(path, "w") as f:
        f.write(f"{len(symbols)}\n{comment}\n")
        for s, (x, y, z) in zip(symbols, positions):
            f.write(f"{s} {x:.8f} {y:.8f} {z:.8f}\n")


def _read_xyz(path: Path):
    lines = path.read_text().splitlines()
    n = int(lines[0].split()[0])
    pos = []
    for ln in lines[2:2 + n]:
        p = ln.split()
        pos.append([float(p[1]), float(p[2]), float(p[3])])
    return np.asarray(pos, dtype=np.float64)


def xtb_optimize(symbols, positions, charge=0, uhf=0, level="normal",
                 cycles=250, timeout=900):
    """GFN2-xTB geometry optimisation. Returns (E_hartree, positions) or (None, None)."""
    xtb = _xtb_binary()
    if xtb is None:
        return None, None
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        xyz = td / "in.xyz"
        _write_xyz(xyz, symbols, positions)
        cmd = [xtb, str(xyz), "--opt", level, "--cycles", str(int(cycles)),
               "--gfn", "2", "--chrg", str(int(charge)), "--uhf", str(int(uhf)),
               "--silent"]
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OMP_STACKSIZE="1G")
        try:
            out = subprocess.run(cmd, cwd=td, capture_output=True, text=True,
                                 timeout=timeout, env=env)
        except (subprocess.TimeoutExpired, OSError):
            return None, None
        opt = td / "xtbopt.xyz"
        if not opt.exists():
            return None, None
        energies = [float(m.group(1)) for m in _E_LINE.finditer(out.stdout)]
        if not energies:
            return None, None
        try:
            newpos = _read_xyz(opt)
        except Exception:
            return None, None
        return energies[-1], newpos


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------
def load_val(processed_dir: Path, atom_map):
    import torch
    d = torch.load(str(processed_dir / "val_data_processed.pt"), map_location="cpu")
    nia = d["node_idx_array"].numpy()
    at = d["atom_types"].numpy()
    ac = d["atom_charges"].numpy()
    pos = d["positions"].numpy()
    return nia, at, ac, pos


def mol_from_xyz(symbols, positions, charge):
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
    n = len(symbols)
    block = f"{n}\n\n" + "\n".join(
        f"{s} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}" for s, p in zip(symbols, positions))
    m = Chem.MolFromXYZBlock(block)
    if m is None:
        raise ValueError("MolFromXYZBlock failed")
    rdDetermineBonds.DetermineBonds(m, charge=int(charge))
    Chem.SanitizeMol(m)
    return m


def screen_molecule(symbols, positions, charge, min_rot):
    """Return (mol, smiles, n_rot) if the molecule is a suitable test system."""
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    m = mol_from_xyz(symbols, positions, charge)
    if any(a.GetNumRadicalElectrons() for a in m.GetAtoms()):
        raise ValueError("radical")
    frags = Chem.GetMolFrags(m)
    if len(frags) != 1:
        raise ValueError(f"{len(frags)} fragments (non-covalent complex)")
    n_rot = rdMolDescriptors.CalcNumRotatableBonds(m)
    if n_rot < min_rot:
        raise ValueError(f"n_rot={n_rot} < {min_rot}")
    smi = Chem.MolToSmiles(Chem.RemoveHs(m))
    return m, smi, int(n_rot)


# --------------------------------------------------------------------------
# per-system ensemble construction
# --------------------------------------------------------------------------
def build_one(task):
    """Worker: build the basin ensemble for one molecule. Returns a dict."""
    (val_index, symbols, data_pos, charge, n_confs, seed, rms_thresh,
     ediff_thresh_eV, opt_level, max_basins) = task
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolTransforms  # noqa: F401
    from rdkit.Chem import rdMolAlign

    res = {"val_index": int(val_index), "status": "ok"}
    try:
        m, smi, n_rot = screen_molecule(symbols, data_pos, charge, min_rot=0)
    except Exception as exc:
        return {"val_index": int(val_index), "status": f"screen_fail:{exc}"}
    res["smiles"] = smi
    res["n_rotatable"] = n_rot

    # --- ETKDG conformers -------------------------------------------------
    mh = Chem.Mol(m)
    params = AllChem.ETKDGv3()
    # NB: randomSeed=0 is a documented RDKit foot-gun -- it makes every embedding
    # attempt produce the SAME coordinates, so pruneRmsThresh collapses the run to
    # a single conformer. Map any user seed onto a nonzero stream.
    params.randomSeed = int(seed) * 7919 + 12345
    params.pruneRmsThresh = 0.12
    params.useSmallRingTorsions = True
    params.numThreads = 1
    try:
        cids = list(AllChem.EmbedMultipleConfs(mh, numConfs=int(n_confs), params=params))
    except Exception as exc:
        return {"val_index": int(val_index), "status": f"embed_fail:{exc}"}
    if len(cids) == 0:
        return {"val_index": int(val_index), "status": "embed_fail:0confs"}

    start_geoms = [np.asarray(mh.GetConformer(c).GetPositions(), dtype=np.float64)
                   for c in cids]
    # Always include the OMol25 data geometry as a starting point so at least one
    # basin is anchored to the training distribution.
    start_geoms.insert(0, np.asarray(data_pos, dtype=np.float64))

    # --- xTB optimisation of every start geometry -------------------------
    minima = []          # (E_hartree, positions, is_from_data)
    for i, g in enumerate(start_geoms):
        E, p = xtb_optimize(symbols, g, charge=charge, level=opt_level)
        if E is None:
            continue
        minima.append((float(E), p, i == 0))
    if len(minima) < 2:
        return {"val_index": int(val_index), "status": f"xtb_fail:{len(minima)}_minima"}

    # Discard any structure whose bonding changed during optimisation (xTB can
    # isomerise / dissociate); a basin must be the SAME molecule.
    ref_smi = smi
    kept = []
    for E, p, from_data in minima:
        try:
            mm = mol_from_xyz(symbols, p, charge)
            s2 = Chem.MolToSmiles(Chem.RemoveHs(mm))
        except Exception:
            continue
        if s2 != ref_smi:
            continue
        kept.append((E, p, from_data))
    if len(kept) < 2:
        return {"val_index": int(val_index),
                "status": f"topology_change:{len(kept)}_kept"}

    kept.sort(key=lambda t: t[0])

    # --- greedy RMSD clustering into basins -------------------------------
    heavy = Chem.RemoveHs(Chem.Mol(m))
    heavy_idx = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() > 1]

    def _mol_with(pos):
        mm = Chem.Mol(heavy)
        mm.RemoveAllConformers()
        conf = Chem.Conformer(mm.GetNumAtoms())
        for j, ai in enumerate(heavy_idx):
            conf.SetAtomPosition(j, tuple(float(v) for v in pos[ai]))
        mm.AddConformer(conf, assignId=True)
        return mm

    basins = []   # list of dicts
    for E, p, from_data in kept:
        placed = False
        mp = _mol_with(p)
        for b in basins:
            if abs((E - b["E"]) * HARTREE_TO_EV) > 3.0:
                continue  # far in energy -> cannot be the same minimum
            try:
                rms = rdMolAlign.GetBestRMS(Chem.Mol(mp), Chem.Mol(b["mol"]))
            except Exception:
                rms = 1e9
            if rms < rms_thresh:
                b["members"].append(float(E))
                b["from_data"] = b["from_data"] or from_data
                placed = True
                break
        if not placed:
            basins.append({"E": float(E), "pos": p, "mol": mp,
                           "members": [float(E)], "from_data": bool(from_data)})

    # merge basins that ended up energetically indistinguishable AND close in RMSD
    basins.sort(key=lambda b: b["E"])
    if len(basins) > max_basins:
        basins = basins[:max_basins]

    E0 = basins[0]["E"]
    rel_eV = [(b["E"] - E0) * HARTREE_TO_EV for b in basins]
    if len(basins) < 2:
        return {"val_index": int(val_index), "status": "single_basin"}
    if max(rel_eV) < ediff_thresh_eV:
        return {"val_index": int(val_index),
                "status": f"degenerate_basins:spread={max(rel_eV):.4f}eV"}

    res["n_start"] = len(start_geoms)
    res["n_minima_kept"] = len(kept)
    res["charge"] = int(charge)
    res["symbols"] = list(symbols)
    res["basins"] = [
        {"basin_id": i,
         "energy_hartree": b["E"],
         "energy_eV": b["E"] * HARTREE_TO_EV,
         "rel_energy_eV": rel_eV[i],
         "n_members": len(b["members"]),
         "from_data_geometry": b["from_data"]}
        for i, b in enumerate(basins)]
    res["_positions"] = np.stack([b["pos"] for b in basins]).astype(np.float32)
    # pairwise heavy-atom RMSD between basins (evidence they are distinct)
    nb = len(basins)
    D = np.zeros((nb, nb), dtype=np.float32)
    for i in range(nb):
        for j in range(i + 1, nb):
            try:
                D[i, j] = D[j, i] = rdMolAlign.GetBestRMS(
                    Chem.Mol(basins[i]["mol"]), Chem.Mol(basins[j]["mol"]))
            except Exception:
                D[i, j] = D[j, i] = float("nan")
    res["basin_rmsd_matrix"] = D.tolist()
    res["min_interbasin_rmsd"] = float(np.nanmin(D[np.triu_indices(nb, 1)])) if nb > 1 else 0.0
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=Path,
                    default=Path("/n/holylabs/woo_lab/Lab/yulili/bgfm/processed_data/omol25_4m_processed"))
    ap.add_argument("--config", type=Path,
                    default=Path("/n/home04/yulili/bgfm/configs/sweep/p0_D_energy_s1.yaml"),
                    help="only used for dataset.atom_map")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--min_atoms", type=int, default=10)
    ap.add_argument("--max_atoms", type=int, default=30)
    ap.add_argument("--min_rot", type=int, default=2)
    ap.add_argument("--n_candidates", type=int, default=200,
                    help="how many screened molecules to attempt (before basin filters)")
    ap.add_argument("--n_systems", type=int, default=40,
                    help="stop after this many systems with >= min_basins basins")
    ap.add_argument("--n_confs", type=int, default=40)
    ap.add_argument("--min_basins", type=int, default=3)
    ap.add_argument("--max_basins", type=int, default=8)
    ap.add_argument("--rms_thresh", type=float, default=0.35,
                    help="heavy-atom best-RMSD (A) below which two minima are one basin")
    ap.add_argument("--ediff_thresh_eV", type=float, default=0.02,
                    help="required energy spread across basins")
    ap.add_argument("--opt_level", default="normal")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_workers", type=int, default=8)
    ap.add_argument("--neutral_only", action="store_true", default=True)
    args = ap.parse_args()

    if _xtb_binary() is None:
        print("[ensembles] ERROR: xtb not on PATH", file=sys.stderr)
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import yaml
    cfg = yaml.safe_load(open(args.config))
    atom_map = cfg["dataset"]["atom_map"]

    nia, at, ac, pos = load_val(args.processed_dir, atom_map)
    n_atoms = nia[:, 1] - nia[:, 0]
    eligible = np.where((n_atoms >= args.min_atoms) & (n_atoms <= args.max_atoms))[0]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(eligible)
    print(f"[ensembles] {len(eligible)} val molecules in "
          f"[{args.min_atoms},{args.max_atoms}] atoms", flush=True)

    tasks, screened = [], 0
    for mi in eligible:
        if len(tasks) >= args.n_candidates:
            break
        s, e = int(nia[mi, 0]), int(nia[mi, 1])
        symbols = [atom_map[int(i)] for i in at[s:e]]
        if not set(symbols) <= ORGANIC:
            continue
        charge = int(ac[s:e].sum())
        if args.neutral_only and charge != 0:
            continue
        p = pos[s:e].astype(np.float64)
        screened += 1
        try:
            screen_molecule(symbols, p, charge, args.min_rot)
        except Exception:
            continue
        tasks.append((int(mi), symbols, p, charge, args.n_confs, args.seed,
                      args.rms_thresh, args.ediff_thresh_eV, args.opt_level,
                      args.max_basins))
    print(f"[ensembles] screened {screened}, {len(tasks)} candidates pass "
          f"(organic, 1 fragment, no radical, n_rot>={args.min_rot})", flush=True)

    results, kept = [], []
    with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
        for r in ex.map(build_one, tasks):
            results.append(r)
            if r.get("status") == "ok" and len(r.get("basins", [])) >= args.min_basins:
                kept.append(r)
                print(f"[ensembles] KEEP val_index={r['val_index']} "
                      f"{r['smiles']} basins={len(r['basins'])} "
                      f"spread={r['basins'][-1]['rel_energy_eV']:.3f} eV "
                      f"minRMSD={r['min_interbasin_rmsd']:.2f} A", flush=True)
            else:
                print(f"[ensembles] drop  val_index={r['val_index']} "
                      f"{r.get('status')} nb={len(r.get('basins', []))}", flush=True)
            if len(kept) >= args.n_systems:
                break

    # ---- serialise -------------------------------------------------------
    geoms = {}
    meta = []
    for r in kept:
        vi = r["val_index"]
        geoms[f"pos_{vi}"] = r.pop("_positions")
        meta.append(r)
    np.savez_compressed(args.out_dir / "geoms.npz", **geoms)
    with open(args.out_dir / "ensembles.json", "w") as f:
        json.dump({"n_systems": len(meta),
                   "args": {k: (str(v) if isinstance(v, Path) else v)
                            for k, v in vars(args).items()},
                   "systems": meta}, f, indent=2)
    status_counts = {}
    for r in results:
        s = r.get("status", "?").split(":")[0]
        status_counts[s] = status_counts.get(s, 0) + 1
    with open(args.out_dir / "build_log.json", "w") as f:
        json.dump({"status_counts": status_counts,
                   "all": [{k: v for k, v in r.items() if k != "_positions"}
                           for r in results]}, f, indent=2)
    print(f"[ensembles] kept {len(meta)} systems -> {args.out_dir}", flush=True)
    print(f"[ensembles] status counts: {status_counts}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
