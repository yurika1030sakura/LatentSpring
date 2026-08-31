"""Stage-1 variant with CONTROLLED perturbations (P3: kill the geometry-distance confound).

Why this script exists
----------------------
``eval_boltzmann_stage1.py`` perturbs a held-out molecule with isotropic Gaussian
noise ``x* + sigma*eps``.  Every perturbation lands at a *different* distance from
the reference: with sigma = 0.15 A the Kabsch-aligned RMSD of a 14-atom molecule
has mean 0.24 A and std 0.029 A -- a 12% spread.  Both ``log p_theta`` (which falls
off away from the data manifold) and ``E`` (which rises away from the minimum)
are monotone in that distance, so a model that has learned nothing but "further
from the manifold => lower density" already scores a positive per-group
correlation.  The published +0.43 could in principle be that artefact.

This script regenerates the same JSON contract as Stage 1 but lets you choose the
perturbation family, so the confound can be removed by construction:

  --perturb_mode gaussian      the incumbent, kept as the in-script control
  --perturb_mode fixed_rmsd    random direction RESCALED so every perturbation in
                               a group has EXACTLY the same RMSD to the reference.
                               Within-group RMSD variance is 0 => the "distance"
                               explanation is mathematically unavailable, and any
                               surviving corr(log p, -E) is direction-selective.
  --perturb_mode torsion       rotate rotatable dihedrals only (bond lengths and
                               bond angles exactly preserved)
  --perturb_mode bond_angle    perturb bond angles only (bond lengths preserved)
  --perturb_mode normal_mode   displace along low-frequency normal modes (ANM
                               proxy by default, real ``xtb --hess`` optionally)

``--rmsd`` may be given for torsion / bond_angle / normal_mode too: the
internal-coordinate perturbation is then rescaled onto the same fixed RMSD shell,
which is the strongest control available (same distance, same deformation family,
different direction).

Multiple scales are supported (``--sigma 0.1,0.15,0.3`` / ``--rmsd 0.1,0.2,0.4``).
Each (molecule, scale) pair becomes its OWN ``group_id`` so that within-group RMSD
stays constant; ``mol_id`` and ``scale`` are also stored per record so downstream
analysis can pool across scales.

Output
------
``<out_dir>/boltzmann_samples.json`` -- byte-compatible with
``eval_boltzmann_stage2.py`` / ``eval_boltzmann_independent.py`` (they read only
``group_id``/``atomic_numbers``/``positions``/``charge``/``log_p_theta``); extra
keys ``mol_id``, ``perturb_mode``, ``scale``, ``rmsd_target``, ``rmsd_actual`` are
ignored by them and consumed by ``analyze_partial_correlation.py``.
``<out_dir>/perturbation_meta.csv`` -- flat (group_id, pert_id, rmsd_actual, ...)
table for the partial-correlation analysis, so no JSON parsing is needed there.

Usage
-----
    $FLOWMOL_PY scripts/eval_boltzmann_controlled.py \
        --checkpoint <ckpt> --config <cfg> --eval_data <processed_dir> \
        --perturb_mode fixed_rmsd --rmsd 0.15 \
        --n_molecules 20 --n_perturb 8 --drop_reference \
        --out_dir <out>/boltz1
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cfm_mol.geom_perturb import (  # noqa: E402
    PERTURB_MODES, anm_modes, build_adjacency, kabsch_rmsd, perturb_bond_angle,
    perturb_fixed_rmsd, perturb_gaussian, perturb_normal_mode, perturb_torsion,
    xtb_modes,
)

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


def _parse_floats(s: str | None) -> list[float]:
    if s is None or str(s).strip() == "":
        return []
    return [float(tok) for tok in str(s).replace(";", ",").split(",") if tok.strip()]


def resolve_scales(args) -> tuple[str, list[float]]:
    """Return (scale_kind, scale_values) for the requested perturbation mode."""
    sig = _parse_floats(args.sigma)
    rms = _parse_floats(args.rmsd)
    m = args.perturb_mode
    if m == "gaussian":
        if not sig:
            raise SystemExit("--perturb_mode gaussian requires --sigma")
        return "sigma", sig
    if m == "fixed_rmsd":
        if not rms:
            raise SystemExit("--perturb_mode fixed_rmsd requires --rmsd")
        return "rmsd", rms
    if m in ("torsion", "bond_angle"):
        if rms:
            return "rmsd", rms          # internal-coord perturbation on an RMSD shell
        if not sig:
            raise SystemExit(f"--perturb_mode {m} requires --rmsd or --sigma (degrees)")
        return "sigma_deg", sig
    if m == "normal_mode":
        if rms:
            return "rmsd", rms
        if not sig:
            raise SystemExit("--perturb_mode normal_mode requires --rmsd or --sigma")
        return "sigma", sig
    raise SystemExit(f"unknown perturb_mode {m}")


def make_perturbation(rng, mode: str, scale_kind: str, scale: float,
                      base: np.ndarray, z: np.ndarray, charge: int, args,
                      cache: dict):
    """Dispatch to cfm_mol.geom_perturb.  Returns (positions|None, info)."""
    if mode == "gaussian":
        return perturb_gaussian(rng, base, sigma=scale)
    if mode == "fixed_rmsd":
        return perturb_fixed_rmsd(rng, base, rmsd=scale, masses=cache.get("masses"))
    if mode == "torsion":
        adj = cache.setdefault("adj", build_adjacency(z, base, tol=args.bond_tol))
        if scale_kind == "rmsd":
            return perturb_torsion(rng, base, z, sigma_deg=args.torsion_sigma_deg,
                                   n_torsions=args.n_torsions, adj=adj,
                                   match_rmsd=scale)
        return perturb_torsion(rng, base, z, sigma_deg=scale,
                               n_torsions=args.n_torsions, adj=adj)
    if mode == "bond_angle":
        adj = cache.setdefault("adj", build_adjacency(z, base, tol=args.bond_tol))
        if scale_kind == "rmsd":
            return perturb_bond_angle(rng, base, z, sigma_deg=args.angle_sigma_deg,
                                      n_angles=args.n_angles, adj=adj,
                                      match_rmsd=scale)
        return perturb_bond_angle(rng, base, z, sigma_deg=scale,
                                  n_angles=args.n_angles, adj=adj)
    if mode == "normal_mode":
        modes = cache.get("modes")
        if modes is None:
            modes = None
            if args.nm_backend == "xtb":
                modes = xtb_modes(z, base, charge=charge, n_modes=args.n_normal_modes)
                if modes is None:
                    print("[boltz1c] xtb --hess failed, falling back to ANM modes",
                          flush=True)
            if modes is None:
                modes = anm_modes(z, base, cutoff=args.anm_cutoff,
                                  n_modes=args.n_normal_modes)
            cache["modes"] = modes
        if scale_kind == "rmsd":
            return perturb_normal_mode(rng, base, z, rmsd=scale, modes=modes)
        return perturb_normal_mode(rng, base, z, sigma=scale, modes=modes)
    raise SystemExit(f"unknown perturb_mode {mode}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--eval_data", type=Path, required=True)
    ap.add_argument("--n_molecules", type=int, default=60)
    ap.add_argument("--n_perturb", type=int, default=8,
                    help="Perturbations per (molecule, scale) group.")
    ap.add_argument("--perturb_mode", default="fixed_rmsd", choices=list(PERTURB_MODES))
    ap.add_argument("--sigma", default=None,
                    help="Comma-separated scales. Angstrom for gaussian/normal_mode, "
                         "DEGREES for torsion/bond_angle.")
    ap.add_argument("--rmsd", default=None,
                    help="Comma-separated target Kabsch RMSDs in Angstrom. Required "
                         "for fixed_rmsd; optional (= match_rmsd) for the "
                         "internal-coordinate modes.")
    ap.add_argument("--drop_reference", action="store_true",
                    help="Do NOT emit the unperturbed reference (pert_id 0). The "
                         "reference is a high-leverage point that inflates the "
                         "per-group correlation.")
    ap.add_argument("--n_torsions", type=int, default=2)
    ap.add_argument("--torsion_sigma_deg", type=float, default=30.0,
                    help="Dihedral std used to pick the DIRECTION when --rmsd sets "
                         "the magnitude.")
    ap.add_argument("--n_angles", type=int, default=2)
    ap.add_argument("--angle_sigma_deg", type=float, default=5.0)
    ap.add_argument("--nm_backend", default="anm", choices=["anm", "xtb"])
    ap.add_argument("--anm_cutoff", type=float, default=8.0)
    ap.add_argument("--n_normal_modes", type=int, default=12)
    ap.add_argument("--bond_tol", type=float, default=1.25)
    ap.add_argument("--n_ode_steps", type=int, default=12)
    ap.add_argument("--n_hutchinson", type=int, default=2)
    ap.add_argument("--max_graphs_per_batch", type=int, default=16,
                    help="Chunk the FFJORD batch to bound activation memory.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--no_patch", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_atoms_filter", type=int, default=60)
    ap.add_argument("--dry_run", action="store_true",
                    help="Build perturbations and report RMSD statistics, but do "
                         "not load the model or compute log p (CPU-only check).")
    args = ap.parse_args()

    scale_kind, scales = resolve_scales(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[boltz1c] mode={args.perturb_mode} scale_kind={scale_kind} scales={scales} "
          f"n_perturb={args.n_perturb} drop_reference={args.drop_reference}", flush=True)

    import torch
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    from flowmol.model_utils.load import model_from_config, read_config_file
    from flowmol.data_processing.dataset import MoleculeDataset

    cfg = read_config_file(args.config)
    cfg.get("mol_fm", {}).pop("bgfm", None)
    cfg["dataset"]["processed_data_dir"] = str(args.eval_data)
    atom_map = cfg["dataset"]["atom_map"]

    model = None
    if not args.dry_run:
        import dgl  # noqa: F401
        from flowmol.data_processing.utils import get_batch_idxs, get_upper_edge_mask
        from cfm_mol.bgfm_density import log_density_via_flow

        model = model_from_config(cfg)
        if not args.no_patch:
            from cfm_mol.domain import default_d_min_table
            from cfm_mol.flow_model import patch_flowmol
            n_real = len(atom_map)
            has_fake = cfg["mol_fm"].get("fake_atom_p", 0.0) > 0
            has_mask = cfg["mol_fm"].get("parameterization", "") == "ctmc"
            n_total = n_real + int(has_fake) + int(has_mask)
            d_min = torch.zeros(n_total, n_total)
            d_min[:n_real, :n_real] = default_d_min_table(n_atom_types=n_real,
                                                          atom_map=atom_map)
            e_weight = cfg["mol_fm"].get("total_loss_weights", {}).get("e", 2.0)
            bond_free = float(e_weight) == 0.0
            patch_flowmol(model, d_min, tangent=True, retract=True, gluing=True,
                          discrete_projection=not bond_free,
                          train_time_discrete=not bond_free, atom_map=atom_map)
        state = torch.load(str(args.checkpoint), map_location="cpu")
        sd = state.get("state_dict", state)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        n_bad = sum(int((~torch.isfinite(v)).sum()) for v in sd.values()
                    if torch.is_tensor(v) and v.is_floating_point())
        print(f"[boltz1c] loaded ckpt: missing={len(missing)} unexpected={len(unexpected)} "
              f"non_finite_params={n_bad}", flush=True)
        if n_bad:
            raise SystemExit("ABORT: checkpoint has non-finite parameters")
        model = model.to(device).eval()

    ds_cfg = dict(cfg["dataset"])
    ds_cfg["fake_atom_p"] = 0.0
    ds_cfg["fake_atom_std"] = 1.0
    ds_cfg["explicit_aromaticity"] = cfg["mol_fm"].get("explicit_aromaticity", False)
    val = MoleculeDataset("val", ds_cfg, prior_config=cfg["mol_fm"]["prior_config"])

    max_atoms = args.max_atoms_filter
    if max_atoms is not None and max_atoms > 0:
        n_atoms_arr = val.node_idx_array[:, 1] - val.node_idx_array[:, 0]
        eligible = (n_atoms_arr <= max_atoms).nonzero(as_tuple=True)[0].tolist()
        print(f"[boltz1c] size filter <= {max_atoms} atoms: {len(eligible)}/{len(val)} "
              f"eligible", flush=True)
    else:
        eligible = list(range(len(val)))
    if not eligible:
        raise SystemExit("no eligible val molecules")
    import random
    random.seed(args.seed)
    random.shuffle(eligible)
    idxs = eligible[: args.n_molecules]

    rng = np.random.default_rng(args.seed)
    records = []
    meta_rows = []
    group_id = 0
    n_skipped = 0
    per_scale_rmsd: dict[float, list[float]] = {s: [] for s in scales}
    within_group_std: list[float] = []

    from cfm_mol.geom_perturb import atomic_masses

    for mi, di in enumerate(idxs):
        g0 = val[di]
        n_atoms = g0.num_nodes()
        base_pos = g0.ndata["x_1_true"].detach().cpu().numpy().astype(float)
        base_pos = base_pos - base_pos.mean(axis=0, keepdims=True)
        atom_idx = g0.ndata["a_1_true"].argmax(dim=-1).cpu().numpy()
        z = np.array([SYMBOL_TO_Z[atom_map[i]] for i in atom_idx], dtype=np.int64)
        total_charge = int((g0.ndata["c_1_true"].argmax(dim=-1) - 2).sum().item())

        for scale in scales:
            cache = {"masses": atomic_masses(z)}
            pert_positions = []
            pert_ids = []
            infos = []
            if not args.drop_reference:
                pert_positions.append(base_pos.copy())
                pert_ids.append(0)
                infos.append({"scale_used": 0.0})
            # pert_id 0 is ALWAYS the unperturbed reference; when --drop_reference
            # is set it simply never appears, so downstream "drop pert_id==0"
            # filters stay valid either way.
            n_fail = 0
            attempts = 0
            last_fail_info = None
            while len(pert_positions) < args.n_perturb + (0 if args.drop_reference else 1) \
                    and attempts < args.n_perturb * 10:
                attempts += 1
                pos, info = make_perturbation(rng, args.perturb_mode, scale_kind,
                                              scale, base_pos, z, total_charge,
                                              args, cache)
                if pos is None:
                    n_fail += 1
                    last_fail_info = info
                    if info.get("failed") in ("no_rotatable_bonds", "no_bond_angles",
                                              "no_modes"):
                        break          # structural, retrying cannot help
                    continue
                pos = pos - pos.mean(axis=0, keepdims=True)
                pert_positions.append(pos)
                pert_ids.append(len(pert_positions) if args.drop_reference
                                else len(pert_positions) - 1)
                infos.append(info)
            if len(pert_positions) < 3:
                print(f"[boltz1c] mol {mi} (val idx {di}): only "
                      f"{len(pert_positions)} usable geometries "
                      f"({last_fail_info}); skipping", flush=True)
                n_skipped += 1
                continue

            # ---- RMSD bookkeeping (this is the whole point of the script) ----
            rmsds = [kabsch_rmsd(p, base_pos) for p in pert_positions]
            nz = [r for r, pid in zip(rmsds, pert_ids) if r > 1e-9]
            if nz:
                per_scale_rmsd[scale].extend(nz)
                within_group_std.append(float(np.std(nz)))

            logp = None
            if not args.dry_run:
                logp = _log_density_chunked(
                    model, g0, pert_positions, device, args)

            for p_i, (pid, pos, info) in enumerate(zip(pert_ids, pert_positions, infos)):
                rec = {
                    "group_id": group_id,
                    "pert_id": int(pid),
                    "mol_id": int(mi),
                    "val_index": int(di),
                    "atomic_numbers": z.tolist(),
                    "positions": pos.astype(float).tolist(),
                    "charge": total_charge,
                    "spin": 1,
                    "perturb_mode": args.perturb_mode,
                    "scale_kind": scale_kind,
                    "scale": float(scale),
                    "rmsd_target": float(scale) if scale_kind == "rmsd" else None,
                    "rmsd_actual": float(rmsds[p_i]),
                }
                if logp is not None:
                    rec["log_p_theta"] = float(logp[p_i])
                records.append(rec)
                meta_rows.append({
                    "group_id": group_id, "pert_id": int(pid), "mol_id": int(mi),
                    "val_index": int(di), "n_atoms": int(n_atoms),
                    "perturb_mode": args.perturb_mode, "scale_kind": scale_kind,
                    "scale": float(scale), "rmsd_actual": float(rmsds[p_i]),
                    "log_p_theta": ("" if logp is None else float(logp[p_i])),
                })

            # For torsion / bond_angle the requested RMSD shell may be
            # UNREACHABLE (e.g. a rigid molecule with one short rotatable
            # branch): scale_to_rmsd then returns the closest achievable
            # geometry, which silently breaks the fixed-RMSD guarantee. Flag it
            # loudly rather than letting it contaminate the control.
            if scale_kind == "rmsd" and nz:
                dev = max(abs(r - scale) for r in nz) / max(scale, 1e-9)
                if dev > 0.01:
                    print(f"[boltz1c] WARNING group {group_id} (mol {mi}): "
                          f"target rmsd {scale} unreachable in mode "
                          f"{args.perturb_mode}; achieved "
                          f"{np.min(nz):.4f}..{np.max(nz):.4f} "
                          f"(max rel. deviation {dev:.1%})", flush=True)
            if nz:
                print(f"[boltz1c] group {group_id} (mol {mi}, N={n_atoms}, "
                      f"scale={scale}): rmsd mean={np.mean(nz):.6f} "
                      f"std={np.std(nz):.3e} min={np.min(nz):.4f} "
                      f"max={np.max(nz):.4f} n={len(nz)}", flush=True)
            group_id += 1

    out_json = args.out_dir / "boltzmann_samples.json"
    with open(out_json, "w") as f:
        json.dump(records, f)
    meta_csv = args.out_dir / "perturbation_meta.csv"
    with open(meta_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()) if meta_rows else
                           ["group_id", "pert_id", "mol_id", "val_index", "n_atoms",
                            "perturb_mode", "scale_kind", "scale", "rmsd_actual",
                            "log_p_theta"])
        w.writeheader()
        for r in meta_rows:
            w.writerow(r)

    print("\n[boltz1c] ===== RMSD CONTROL SUMMARY =====", flush=True)
    for s in scales:
        v = np.asarray(per_scale_rmsd[s], dtype=float)
        if v.size == 0:
            continue
        print(f"[boltz1c] scale={s}: n={v.size} rmsd mean={v.mean():.6f} "
              f"std={v.std():.4e} (pooled across groups)", flush=True)
    if within_group_std:
        ws = np.asarray(within_group_std)
        print(f"[boltz1c] WITHIN-GROUP rmsd std: max={ws.max():.4e} "
              f"mean={ws.mean():.4e}  <-- must be ~0 for fixed_rmsd / match_rmsd",
              flush=True)
    print(f"[boltz1c] groups={group_id} records={len(records)} skipped_mols={n_skipped}",
          flush=True)
    print(f"[boltz1c] wrote {out_json}", flush=True)
    print(f"[boltz1c] wrote {meta_csv}", flush=True)
    if not args.dry_run:
        print(f"[boltz1c] next: scripts/eval_boltzmann_independent.py "
              f"--samples_json {out_json}", flush=True)
    return 0


def _log_density_chunked(model, g0, pert_positions, device, args):
    """FFJORD log p for a list of geometries of the same molecule, in chunks."""
    import dgl
    import torch
    from flowmol.data_processing.utils import get_batch_idxs, get_upper_edge_mask
    from cfm_mol.bgfm_density import log_density_via_flow

    out = []
    chunk = max(1, int(args.max_graphs_per_batch))
    for start in range(0, len(pert_positions), chunk):
        sub = pert_positions[start:start + chunk]
        graphs = []
        for pos in sub:
            gp = g0.clone()
            gp.ndata["x_1_true"] = torch.as_tensor(pos, dtype=torch.float32)
            graphs.append(gp)
        gbatch = dgl.batch(graphs).to(device)
        gbatch.ndata["x_t"] = gbatch.ndata["x_1_true"]
        gbatch.ndata["a_t"] = gbatch.ndata["a_1_true"]
        gbatch.ndata["c_t"] = gbatch.ndata["c_1_true"]
        gbatch.edata["e_t"] = gbatch.edata["e_1_true"]
        nbi, _ = get_batch_idxs(gbatch)
        uem = get_upper_edge_mask(gbatch)
        with torch.enable_grad():
            lp = log_density_via_flow(model, gbatch, nbi, uem,
                                      n_ode_steps=args.n_ode_steps,
                                      n_hutchinson=args.n_hutchinson, prior_std=1.0)
        out.extend(lp.detach().cpu().numpy().tolist())
    return np.asarray(out, dtype=float)


if __name__ == "__main__":
    sys.exit(main())
