"""P6 -- validate the FFJORD log-likelihood estimator used by the BGFM eval.

The headline BGFM number (per-group Pearson r between log p_theta and -E_xTB/kT)
is computed with an FFJORD reverse-time ODE whose divergence is a *stochastic*
Hutchinson estimate at n_ode_steps=12 / n_hutchinson=2. If that estimator has
geometry-correlated noise or bias, the correlation could be an artifact of the
estimator rather than a property of the learned density. This script quantifies
that, on FROZEN geometries so every configuration sees exactly the same inputs.

What it measures
----------------
  A. grid sweep   log p over n_ode_steps x n_hutchinson, repeated R times with
                  independent Hutchinson noise  -> mean, within-record std
  B. noise        per-record estimator std, and whether it CORRELATES with
                  geometry (n_atoms, RMSD from the reference conformer, and --
                  the dangerous one -- the xTB energy itself)
  C. bias         cell mean minus the highest-precision cell mean, and whether
                  that residual bias correlates with energy (a bias that tracks
                  E would manufacture the headline correlation)
  D. exact        Hutchinson vs exact divergence (divergence_exact_atomwise) on
                  small molecules -> bias of the trace estimator itself
  E. metric       per-group Pearson r / Spearman / slope / NRV as a function of
                  (n_ode_steps, n_hutchinson), plus the noise-attenuation
                  correction  r_true ~= r_obs * sqrt(1 + sigma_noise^2/sigma_signal^2)

Everything is written incrementally to <out_dir>/cells/*.json so a preempted
(gpu_requeue) job resumes where it stopped: re-running skips finished cells.

Usage (envs/flowmol):
  python scripts/validate_likelihood_estimator.py \
      --checkpoint <ckpt> --config <cfg> --eval_data <processed_dir> \
      --n_molecules 16 --n_perturb 8 --out_dir <dir>
  python scripts/validate_likelihood_estimator.py --analyze_only --out_dir <dir>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

HARTREE_TO_EV = 27.211386245988

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
SYMBOL_TO_Z = {sym: i + 1 for i, sym in enumerate(_PERIODIC_SYMBOLS)}


# ---------------------------------------------------------------------------
# small stats helpers (kept dependency-free; scipy only for spearman if present)
# ---------------------------------------------------------------------------
try:
    from scipy.stats import spearmanr as _spearmanr
except Exception:  # pragma: no cover
    _spearmanr = None


def _atomic_json_dump(obj, path: Path):
    """Write JSON atomically.

    The job runs under a wall clock and may be preempted (gpu_requeue). A
    half-written cell file would be silently 'skipped as done' on resume and
    then break the analysis, so write to a temp file and rename.
    """
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _pearson(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _pearson_p(r, n):
    """Two-sided p-value for a Pearson r with n samples (t approximation)."""
    if not np.isfinite(r) or n < 4 or abs(r) >= 1.0:
        return float("nan")
    t = r * math.sqrt((n - 2) / max(1e-12, 1 - r * r))
    # survival function of |t| with n-2 dof via incomplete beta (use erf approx
    # through scipy if available, else normal approx which is fine for n>=20)
    try:
        from scipy.stats import t as _tdist
        return float(2 * _tdist.sf(abs(t), n - 2))
    except Exception:
        return float(2 * 0.5 * math.erfc(abs(t) / math.sqrt(2)))


def _partial_corr(x, y, z):
    """Pearson corr(x, y | z), z a single control variable."""
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[m], y[m], z[m]
    if x.size < 4:
        return float("nan")
    def resid(a):
        A = np.vstack([z, np.ones_like(z)]).T
        coef, *_ = np.linalg.lstsq(A, a, rcond=None)
        return a - A @ coef
    return _pearson(resid(x), resid(y))


# ---------------------------------------------------------------------------
# geometry construction (frozen, shared by every estimator configuration)
# ---------------------------------------------------------------------------

def build_geometries(args):
    """Select molecules and construct the frozen perturbation set.

    Returns (records, val_dataset_handle, per_group_dgl_template_index).
    Records are plain python dicts (JSON-serialisable) so they can be cached.
    """
    import torch
    from flowmol.data_processing.dataset import MoleculeDataset
    from flowmol.model_utils.load import read_config_file

    cfg = read_config_file(args.config)
    cfg.get("mol_fm", {}).pop("bgfm", None)
    cfg["dataset"]["processed_data_dir"] = str(args.eval_data)
    atom_map = cfg["dataset"]["atom_map"]

    ds_cfg = dict(cfg["dataset"])
    ds_cfg["fake_atom_p"] = 0.0
    ds_cfg["fake_atom_std"] = 1.0
    ds_cfg["explicit_aromaticity"] = cfg["mol_fm"].get("explicit_aromaticity", False)
    val = MoleculeDataset("val", ds_cfg, prior_config=cfg["mol_fm"]["prior_config"])

    n_atoms_arr = (val.node_idx_array[:, 1] - val.node_idx_array[:, 0]).numpy()
    eligible = np.nonzero((n_atoms_arr >= args.min_atoms) &
                          (n_atoms_arr <= args.max_atoms))[0]
    if eligible.size == 0:
        raise RuntimeError(
            f"no val molecules with {args.min_atoms} <= n_atoms <= {args.max_atoms}")
    rng = np.random.default_rng(args.seed)
    rng.shuffle(eligible)
    idxs = eligible[: args.n_molecules].tolist()
    print(f"[p6] {len(eligible)} eligible val mols in "
          f"[{args.min_atoms},{args.max_atoms}] atoms; using {len(idxs)}", flush=True)

    groups = []
    for gi, di in enumerate(idxs):
        g0 = val[di]
        base = g0.ndata['x_1_true'].detach().clone()
        base = base - base.mean(dim=0, keepdim=True)
        at_idx = g0.ndata['a_1_true'].argmax(dim=-1).numpy()
        z = [SYMBOL_TO_Z[atom_map[int(i)]] for i in at_idx]
        charge = int((g0.ndata['c_1_true'].argmax(dim=-1) - 2).sum().item())
        # Dedicated generator per group: geometries are IDENTICAL across every
        # estimator configuration and across job restarts, which is the whole
        # point of this study.
        gen = torch.Generator().manual_seed(args.seed * 100003 + gi)
        positions, rmsds = [], []
        for p in range(args.n_perturb + 1):
            if p == 0:
                pos = base.clone()
            else:
                noise = torch.randn(base.shape, generator=gen) * args.sigma
                pos = base + noise
                pos = pos - pos.mean(dim=0, keepdim=True)
            positions.append(pos.numpy().astype(float))
            d = (pos - base).numpy()
            rmsds.append(float(np.sqrt((d ** 2).sum(axis=1).mean())))
        groups.append({
            "group_id": gi, "dataset_idx": int(di), "n_atoms": int(base.shape[0]),
            "atomic_numbers": z, "charge": charge,
            "positions": [p.tolist() for p in positions],
            "rmsd_from_ref": rmsds,
        })
    return cfg, val, idxs, groups


# ---------------------------------------------------------------------------
# xTB energies (independent potential; identical to the headline eval)
# ---------------------------------------------------------------------------

def compute_xtb(groups, out_path: Path):
    from eval_xtb_relaxation import (_xtb_single_point_energy, _write_xyz,
                                     _xtb_binary)
    if _xtb_binary() is None:
        print("[p6] WARNING: xtb not on PATH -- skipping energies", flush=True)
        return None
    energies = {}
    t0 = time.time()
    for g in groups:
        es = []
        for p, pos in enumerate(g["positions"]):
            with tempfile.TemporaryDirectory() as td:
                xyz = Path(td) / "m.xyz"
                _write_xyz(xyz, g["atomic_numbers"], pos, int(g["charge"]))
                E, _ = _xtb_single_point_energy(Path(td), xyz, int(g["charge"]))
            es.append(float(E) * HARTREE_TO_EV if E is not None else float("nan"))
        energies[str(g["group_id"])] = es
    _atomic_json_dump(energies, out_path)
    print(f"[p6] xtb energies for {len(groups)} groups in {time.time()-t0:.0f}s "
          f"-> {out_path}", flush=True)
    return energies


# ---------------------------------------------------------------------------
# model + log p
# ---------------------------------------------------------------------------

def build_model(cfg, args):
    import torch
    from flowmol.model_utils.load import model_from_config
    atom_map = cfg["dataset"]["atom_map"]
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
    bad = sum(int((~torch.isfinite(v)).sum()) for v in sd.values()
              if v.is_floating_point())
    if bad:
        raise RuntimeError(f"checkpoint has {bad} non-finite params -- refusing")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[p6] ckpt loaded: missing={len(missing)} unexpected={len(unexpected)}",
          flush=True)
    return model


def logp_for_group(model, val, dataset_idx, positions, device,
                   n_ode_steps, n_hutchinson, torch_seed):
    """log p for every geometry of one group, in a single batched FFJORD pass.

    n_hutchinson semantics:
        > 0  ordinary Hutchinson with independent probes for every graph
        = 0  EXACT divergence (3N backward passes)
        < 0  |n_hutchinson| probes with COMMON RANDOM NUMBERS: the same probe
             pattern is tiled across all geometries of the group. Since the
             headline metric only uses log p DIFFERENCES within a group, CRN
             cancels most of the trace-estimator noise and gives a
             variance-reduced read of the same quantity.
    """
    import torch
    import dgl
    from flowmol.data_processing.utils import get_batch_idxs, get_upper_edge_mask
    from cfm_mol.bgfm_density import log_density_via_flow

    g0 = val[dataset_idx]
    graphs = []
    for pos in positions:
        gp = g0.clone()
        gp.ndata['x_1_true'] = torch.tensor(np.asarray(pos), dtype=torch.float32)
        graphs.append(gp)
    gb = dgl.batch(graphs).to(device)
    gb.ndata['x_t'] = gb.ndata['x_1_true']
    gb.ndata['a_t'] = gb.ndata['a_1_true']
    gb.ndata['c_t'] = gb.ndata['c_1_true']
    gb.edata['e_t'] = gb.edata['e_1_true']
    nbi, _ = get_batch_idxs(gb)
    uem = get_upper_edge_mask(gb)

    torch.manual_seed(torch_seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(torch_seed)

    kwargs = {}
    nh = n_hutchinson
    if n_hutchinson < 0:
        nh = -n_hutchinson
        n_geom = len(positions)
        n_atoms = gb.num_nodes() // n_geom

        def xi_fn(step, k, x, _na=n_atoms, _ng=n_geom, _s=torch_seed):
            # One probe pattern per (ODE step, Hutchinson sample), TILED across
            # the geometries of this group -> common random numbers.
            g = torch.Generator(device="cpu").manual_seed(
                _s + 1_000_003 * step + 7919 * k)
            base = (torch.randint(0, 2, (_na, 3), generator=g,
                                  dtype=torch.float32) * 2 - 1)
            return base.repeat(_ng, 1).to(x.device, x.dtype)

        try:
            from inspect import signature
            if "xi_fn" in signature(log_density_via_flow).parameters:
                kwargs["xi_fn"] = xi_fn
            else:
                raise RuntimeError(
                    "log_density_via_flow has no xi_fn hook -- CRN cells "
                    "require the xi_provider support in cfm_mol/bgfm_loss.py")
        except Exception:
            raise

    with torch.enable_grad():
        logp = log_density_via_flow(model, gb, nbi, uem,
                                    n_ode_steps=n_ode_steps,
                                    n_hutchinson=nh, prior_std=1.0, **kwargs)
    return logp.detach().float().cpu().numpy().tolist()


def run_cell(model, val, groups, idxs, device, n_ode_steps, n_hutchinson, rep,
             seed, cell_path: Path, subset=None):
    t0 = time.time()
    out = {"n_ode_steps": n_ode_steps, "n_hutchinson": n_hutchinson, "rep": rep,
           "logp": {}}
    for g in groups:
        gid = g["group_id"]
        if subset is not None and gid not in subset:
            continue
        ts = (seed * 7919 + gid * 104729 + rep * 1299709
              + n_ode_steps * 31 + max(n_hutchinson, 0) * 17)
        out["logp"][str(gid)] = logp_for_group(
            model, val, idxs[gid], g["positions"], device,
            n_ode_steps, n_hutchinson, ts)
    out["seconds"] = time.time() - t0
    _atomic_json_dump(out, cell_path)
    print(f"[p6] cell steps={n_ode_steps} probes={n_hutchinson} rep={rep} "
          f"({len(out['logp'])} groups) in {out['seconds']:.0f}s", flush=True)
    return out


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

def _group_metrics(logp, negE):
    """Pearson / Spearman / slope / NRV for one group."""
    logp = np.asarray(logp, float)
    negE = np.asarray(negE, float)
    m = np.isfinite(logp) & np.isfinite(negE)
    logp, negE = logp[m], negE[m]
    if logp.size < 3 or np.std(logp) < 1e-12 or np.std(negE) < 1e-12:
        return None
    r = float(np.corrcoef(logp, negE)[0, 1])
    slope = float(np.polyfit(negE, logp, 1)[0])
    sp = float(_spearmanr(logp, negE).correlation) if _spearmanr else float("nan")
    # NRV_m = Var(log p - negE) / Var(negE)  [ = Var(log p + E/kT)/Var(E/kT) ]
    nrv = float(np.var(logp - negE) / max(1e-12, np.var(negE)))
    return {"r": r, "spearman": sp, "slope": slope, "nrv": nrv, "n": int(logp.size)}


def analyze(out_dir: Path, kT: float, drop_ref: bool):
    groups = json.load(open(out_dir / "geometries.json"))
    energies = {}
    ep = out_dir / "xtb_energies.json"
    if ep.exists():
        energies = json.load(open(ep))

    cells = {}
    for f in sorted((out_dir / "cells").glob("*.json")):
        try:
            d = json.load(open(f))
        except Exception as e:      # truncated by a wall-clock kill
            print(f"[p6] WARNING: unreadable cell {f.name} ({e}) -- skipping")
            continue
        key = (int(d["n_ode_steps"]), int(d["n_hutchinson"]))
        cells.setdefault(key, []).append(d)

    gmeta = {g["group_id"]: g for g in groups}
    sel = slice(1, None) if drop_ref else slice(None)

    def negE(gid):
        es = energies.get(str(gid))
        if es is None:
            return None
        return (-np.asarray(es, float) / kT)[sel]

    report = {"kT_eV": kT, "drop_reference_conformer": drop_ref, "cells": {}}

    # ---- per-cell aggregate ------------------------------------------------
    ref_key = None
    for key in sorted(cells):
        reps = cells[key]
        gids = sorted(int(k) for k in reps[0]["logp"])
        # stack (n_rep, n_geom) per group
        per_group_r, per_group_nrv, per_group_slope, per_group_sp = [], [], [], []
        noise_sd, marg_noise_sd, signal_sd, rec_rows = [], [], [], []
        r_repavg = []
        for gid in gids:
            arr = np.array([np.asarray(rp["logp"][str(gid)], float)[sel]
                            for rp in reps if str(gid) in rp["logp"]])
            if arr.size == 0:
                continue
            mean_lp = arr.mean(axis=0)
            sd_lp = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean_lp)
            # Pearson r is invariant to a per-group shift of log p, so the noise
            # that actually degrades the metric is the noise on WITHIN-GROUP
            # CENTRED log p, not the marginal noise. (For common-random-number
            # cells the two differ a lot: CRN correlates the trace noise across
            # the geometries of a group, so it cancels in the centred values.)
            arr_c = arr - arr.mean(axis=1, keepdims=True)
            sd_c = (arr_c.std(axis=0, ddof=1) if arr.shape[0] > 1
                    else np.zeros_like(mean_lp))
            noise_sd.append(float(np.mean(sd_c)))
            marg_noise_sd.append(float(np.mean(sd_lp)))
            signal_sd.append(float(np.std(mean_lp)))
            e = negE(gid)
            if e is not None:
                for rp_i in range(arr.shape[0]):
                    m = _group_metrics(arr[rp_i], e)
                    if m:
                        per_group_r.append(m["r"]); per_group_nrv.append(m["nrv"])
                        per_group_slope.append(m["slope"]); per_group_sp.append(m["spearman"])
                m2 = _group_metrics(mean_lp, e)
                if m2:
                    r_repavg.append(m2["r"])
            rm = np.asarray(gmeta[gid]["rmsd_from_ref"], float)[sel]
            for j in range(mean_lp.size):
                rec_rows.append({
                    "gid": gid, "n_atoms": gmeta[gid]["n_atoms"],
                    "rmsd": float(rm[j]), "mean_logp": float(mean_lp[j]),
                    "sd_logp": float(sd_lp[j]),
                    "sd_logp_centred": float(sd_c[j]),
                    "negE": float(e[j]) if e is not None else float("nan"),
                })
        n_atoms_v = [r["n_atoms"] for r in rec_rows]
        rmsd_v = [r["rmsd"] for r in rec_rows]
        sd_v = [r["sd_logp"] for r in rec_rows]
        sdc_v = [r["sd_logp_centred"] for r in rec_rows]
        negE_v = [r["negE"] for r in rec_rows]
        mlp_v = [r["mean_logp"] for r in rec_rows]
        # within-group correlation between estimator noise and energy
        wg = []
        for gid in set(r["gid"] for r in rec_rows):
            rows = [r for r in rec_rows if r["gid"] == gid]
            wg.append(_pearson([r["sd_logp_centred"] for r in rows],
                               [r["negE"] for r in rows]))
        entry = {
            "n_rep": len(reps),
            "n_groups": len(set(r["gid"] for r in rec_rows)),
            "n_records": len(rec_rows),
            "logp_mean": float(np.mean(mlp_v)) if mlp_v else float("nan"),
            "estimator_sd_mean": float(np.mean(sd_v)) if sd_v else float("nan"),
            "estimator_sd_median": float(np.median(sd_v)) if sd_v else float("nan"),
            "estimator_sd_centred_mean": (float(np.mean(sdc_v)) if sdc_v
                                          else float("nan")),
            "within_group_signal_sd_mean": float(np.mean(signal_sd)) if signal_sd else float("nan"),
            "snr_signal_over_noise": (float(np.mean(signal_sd) / max(1e-12, np.mean(noise_sd)))
                                      if signal_sd and noise_sd else float("nan")),
            "mean_per_group_r": float(np.mean(per_group_r)) if per_group_r else float("nan"),
            "sem_per_group_r": (float(np.std(per_group_r, ddof=1) / math.sqrt(len(per_group_r)))
                                if len(per_group_r) > 1 else float("nan")),
            "mean_per_group_r_repavg": float(np.mean(r_repavg)) if r_repavg else float("nan"),
            "mean_per_group_spearman": float(np.mean(per_group_sp)) if per_group_sp else float("nan"),
            "median_per_group_nrv": float(np.median(per_group_nrv)) if per_group_nrv else float("nan"),
            "median_per_group_slope": float(np.median(per_group_slope)) if per_group_slope else float("nan"),
            "corr_noise_vs_n_atoms": _pearson(sdc_v, n_atoms_v),
            "corr_noise_vs_rmsd": _pearson(sdc_v, rmsd_v),
            "corr_noise_vs_negE_pooled": _pearson(sdc_v, negE_v),
            "corr_marginal_noise_vs_n_atoms": _pearson(sd_v, n_atoms_v),
            "mean_within_group_corr_noise_vs_negE": (
                float(np.nanmean(wg)) if wg else float("nan")),
            "seconds_per_rep": float(np.mean([rp.get("seconds", float("nan"))
                                              for rp in reps])),
        }
        # NULL control: pair each group's log p with a DIFFERENT group's energy
        # vector (all groups have the same number of geometries). If the
        # pipeline can manufacture a correlation out of estimator noise +
        # analysis choices, it shows up here. Expectation is ~0.
        null_r = []
        gid_list = sorted(set(r["gid"] for r in rec_rows))
        for gid in gid_list:
            lp = np.array([np.asarray(rp["logp"][str(gid)], float)[sel]
                           for rp in reps if str(gid) in rp["logp"]]).mean(axis=0)
            for other in gid_list:          # every mismatched pairing, not just one
                if other == gid:
                    continue
                e_other = negE(other)
                if e_other is None or e_other.shape != lp.shape:
                    continue
                m = _group_metrics(lp, e_other)
                if m:
                    null_r.append(m["r"])
        entry["null_r_shuffled_group_energies"] = (
            float(np.mean(null_r)) if null_r else float("nan"))
        entry["null_r_abs_mean"] = (
            float(np.mean(np.abs(null_r))) if null_r else float("nan"))
        # noise-attenuation correction: r_obs is attenuated by measurement noise
        # in log p.  r_true ~= r_obs * sqrt(1 + sigma_noise^2 / sigma_signal^2)
        s, n = (entry["within_group_signal_sd_mean"],
                entry["estimator_sd_centred_mean"])
        if entry["n_rep"] < 2:
            # With a single repeat there is no repeat-to-repeat SD to measure;
            # reporting 0 would make SNR explode and the attenuation correction
            # meaningless. Mark as unknown instead.
            for k in ("estimator_sd_mean", "estimator_sd_median",
                      "estimator_sd_centred_mean", "snr_signal_over_noise",
                      "corr_noise_vs_n_atoms", "corr_noise_vs_rmsd",
                      "corr_noise_vs_negE_pooled",
                      "corr_marginal_noise_vs_n_atoms",
                      "mean_within_group_corr_noise_vs_negE"):
                entry[k] = float("nan")
            entry["attenuation_corrected_r"] = float("nan")
        elif np.isfinite(s) and np.isfinite(n) and s > 0:
            # signal_sd measured on the repeat-MEAN already has noise var/n_rep
            var_true = max(1e-12, s ** 2 - (n ** 2) / max(1, entry["n_rep"]))
            entry["attenuation_corrected_r"] = (
                entry["mean_per_group_r"] * math.sqrt((var_true + n ** 2) / var_true))
        report["cells"][f"steps{key[0]}_probes{key[1]}"] = entry
        if key[1] > 0:
            ref_key = key if ref_key is None or (key[0] * 100 + key[1]) > (
                ref_key[0] * 100 + ref_key[1]) else ref_key

    # ---- bias of each cell relative to the highest-precision cell ----------
    if ref_key is not None:
        ref_reps = cells[ref_key]
        ref_mean = {}
        for gid_s in ref_reps[0]["logp"]:
            ref_mean[int(gid_s)] = np.array(
                [np.asarray(rp["logp"][gid_s], float) for rp in ref_reps]).mean(axis=0)
        report["reference_cell"] = f"steps{ref_key[0]}_probes{ref_key[1]}"
        biases = {}
        for key in sorted(cells):
            if key == ref_key:
                continue
            d_all, e_all, na_all, rm_all, wgb = [], [], [], [], []
            for gid, rmean in ref_mean.items():
                reps = [rp for rp in cells[key] if str(gid) in rp["logp"]]
                if not reps:
                    continue
                cm = np.array([np.asarray(rp["logp"][str(gid)], float)
                               for rp in reps]).mean(axis=0)
                if cm.shape != rmean.shape:
                    continue
                d = (cm - rmean)[sel]
                e = negE(gid)
                d_all.extend(d.tolist())
                na_all.extend([gmeta[gid]["n_atoms"]] * d.size)
                rm_all.extend(np.asarray(gmeta[gid]["rmsd_from_ref"], float)[sel].tolist())
                if e is not None:
                    e_all.extend(e.tolist())
                    wgb.append(_pearson(d, e))
            if not d_all:
                continue
            biases[f"steps{key[0]}_probes{key[1]}"] = {
                "mean_bias": float(np.mean(d_all)),
                "sd_bias": float(np.std(d_all)),
                "mean_abs_bias": float(np.mean(np.abs(d_all))),
                "corr_bias_vs_n_atoms": _pearson(d_all, na_all),
                "corr_bias_vs_rmsd": _pearson(d_all, rm_all),
                "mean_within_group_corr_bias_vs_negE": (
                    float(np.nanmean(wgb)) if wgb else float("nan")),
            }
        report["bias_vs_reference"] = biases

    # ---- exact-vs-Hutchinson ----------------------------------------------
    exact_path = out_dir / "exact_vs_hutchinson.json"
    if exact_path.exists():
        report["exact_vs_hutchinson"] = json.load(open(exact_path))

    # ---- global geometry confound sanity check ----------------------------
    if energies:
        # pooled (within-group centred) r between log p and negE at the
        # headline cell, plus partial correlation controlling for RMSD
        hk = (12, 2)
        if hk in cells:
            lp_c, e_c, rm_c = [], [], []
            reps = cells[hk]
            for gid_s in reps[0]["logp"]:
                gid = int(gid_s)
                e = negE(gid)
                if e is None:
                    continue
                lp = np.array([np.asarray(rp["logp"][gid_s], float)[sel]
                               for rp in reps]).mean(axis=0)
                rm = np.asarray(gmeta[gid]["rmsd_from_ref"], float)[sel]
                lp_c.extend((lp - lp.mean()).tolist())
                e_c.extend((e - e.mean()).tolist())
                rm_c.extend((rm - rm.mean()).tolist())
            report["headline_cell_pooled"] = {
                "pooled_r": _pearson(lp_c, e_c),
                "pooled_r_partial_given_rmsd": _partial_corr(lp_c, e_c, rm_c),
                "corr_logp_vs_rmsd": _pearson(lp_c, rm_c),
                "corr_negE_vs_rmsd": _pearson(e_c, rm_c),
                "n": len(lp_c),
            }

    json.dump(report, open(out_dir / "report.json", "w"), indent=2)
    print(json.dumps(report, indent=2))
    return report


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--eval_data", type=Path)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--n_molecules", type=int, default=16)
    ap.add_argument("--n_perturb", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--min_atoms", type=int, default=8)
    ap.add_argument("--max_atoms", type=int, default=40)
    ap.add_argument("--ode_steps", default="4,8,12,24,48")
    ap.add_argument("--probes", default="1,2,4,8")
    ap.add_argument("--n_repeat", type=int, default=3)
    ap.add_argument("--crn_cells", default="",
                    help="extra common-random-number cells, 'steps:probes' "
                         "comma separated, e.g. '12:2,48:8'. CRN tiles one probe "
                         "pattern across all geometries of a group, cancelling "
                         "most of the trace noise in within-group log p "
                         "DIFFERENCES (which is all the metric uses). Reported "
                         "with probes shown NEGATIVE.")
    ap.add_argument("--exact_max_atoms", type=int, default=15)
    ap.add_argument("--exact_n_mols", type=int, default=4)
    ap.add_argument("--exact_n_geom", type=int, default=3)
    ap.add_argument("--exact_ode_steps", default="12,48")
    ap.add_argument("--exact_hutch_reps", type=int, default=8)
    ap.add_argument("--kT_eV", type=float, default=1.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_patch", action="store_true")
    ap.add_argument("--skip_xtb", action="store_true")
    ap.add_argument("--analyze_only", action="store_true")
    ap.add_argument("--drop_ref", action="store_true",
                    help="analysis: drop pert_id=0 (the unperturbed conformer)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "cells").mkdir(exist_ok=True)

    if args.analyze_only:
        analyze(args.out_dir, args.kT_eV, args.drop_ref)
        return 0

    for req in ("checkpoint", "config", "eval_data"):
        if getattr(args, req) is None:
            ap.error(f"--{req} required unless --analyze_only")

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[p6] device={device}", flush=True)

    geo_path = args.out_dir / "geometries.json"
    cfg, val, idxs, groups = build_geometries(args)
    if geo_path.exists():
        # keep the frozen geometries from the first run (resume safety)
        cached = json.load(open(geo_path))
        if len(cached) == len(groups) and all(
                c["dataset_idx"] == g["dataset_idx"] for c, g in zip(cached, groups)):
            groups = cached
            print("[p6] reusing cached geometries.json", flush=True)
        else:
            raise RuntimeError(
                "geometries.json exists but does not match current args -- "
                "use a fresh --out_dir")
    else:
        json.dump(groups, open(geo_path, "w"))
    idxs = [g["dataset_idx"] for g in groups]

    ep = args.out_dir / "xtb_energies.json"
    if not args.skip_xtb and not ep.exists():
        compute_xtb(groups, ep)

    model = build_model(cfg, args).to(device).eval()

    steps_list = [int(s) for s in args.ode_steps.split(",") if s]
    probes_list = [int(s) for s in args.probes.split(",") if s]
    # cheapest first so a preempted job still yields a usable table
    crn = []
    for tok in args.crn_cells.split(","):
        if tok.strip():
            s, p = tok.split(":")
            crn.append((int(s), -abs(int(p))))
    plan = sorted([(s, p, r) for s in steps_list for p in probes_list
                   for r in range(args.n_repeat)]
                  + [(s, p, r) for (s, p) in crn for r in range(args.n_repeat)],
                  key=lambda t: (t[0] * (abs(t[1]) + 1), t[2]))
    for (s, p, r) in plan:
        cp = args.out_dir / "cells" / f"s{s}_p{p}_r{r}.json"
        if cp.exists():
            print(f"[p6] skip existing {cp.name}", flush=True)
            continue
        try:
            run_cell(model, val, groups, idxs, device, s, p, r, args.seed, cp)
        except RuntimeError as e:
            print(f"[p6] cell s={s} p={p} r={r} FAILED: {e}", flush=True)
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    # ---- exact divergence comparison on the smallest molecules ------------
    exact_path = args.out_dir / "exact_vs_hutchinson.json"
    if args.exact_n_mols > 0 and not exact_path.exists():
        small = [g for g in groups if g["n_atoms"] <= args.exact_max_atoms]
        small = sorted(small, key=lambda g: g["n_atoms"])[: args.exact_n_mols]
        if not small:
            print(f"[p6] no molecules <= {args.exact_max_atoms} atoms in the "
                  f"selection -- selecting a dedicated small set", flush=True)
            sub_args = argparse.Namespace(**vars(args))
            sub_args.min_atoms = 3
            sub_args.max_atoms = args.exact_max_atoms
            sub_args.n_molecules = args.exact_n_mols
            sub_args.seed = args.seed + 1
            _, _, sidx, small = build_geometries(sub_args)
            json.dump(small, open(args.out_dir / "geometries_small.json", "w"))
        res = {"molecules": []}
        for g in small:
            pos = g["positions"][: args.exact_n_geom]
            di = g["dataset_idx"]
            for ns in [int(s) for s in args.exact_ode_steps.split(",") if s]:
                try:
                    lp_exact = logp_for_group(model, val, di, pos, device, ns, 0,
                                              args.seed)
                except RuntimeError as e:
                    print(f"[p6] exact div failed (n_atoms={g['n_atoms']}, "
                          f"steps={ns}): {e}", flush=True)
                    if device.startswith("cuda"):
                        torch.cuda.empty_cache()
                    continue
                entry = {"group_id": g["group_id"], "n_atoms": g["n_atoms"],
                         "n_ode_steps": ns, "logp_exact": lp_exact,
                         "hutchinson": {}}
                for nh in [1, 2, 4, 8]:
                    reps = [logp_for_group(model, val, di, pos, device, ns, nh,
                                           args.seed + 777 * k)
                            for k in range(args.exact_hutch_reps)]
                    a = np.array(reps)                     # (R, n_geom)
                    entry["hutchinson"][str(nh)] = {
                        "mean": a.mean(axis=0).tolist(),
                        "sd": a.std(axis=0, ddof=1).tolist(),
                        "bias_vs_exact": (a.mean(axis=0)
                                          - np.asarray(lp_exact)).tolist(),
                    }
                res["molecules"].append(entry)
                print(f"[p6] exact-div gid={g['group_id']} n_atoms={g['n_atoms']} "
                      f"steps={ns} done", flush=True)
        _atomic_json_dump(res, exact_path)

    analyze(args.out_dir, args.kT_eV, args.drop_ref)
    return 0


if __name__ == "__main__":
    sys.exit(main())
