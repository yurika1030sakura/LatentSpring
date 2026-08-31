"""P1 GLOBAL-ENSEMBLE eval: does the model put the right RELATIVE mass on
different conformational basins?

Background
----------
All existing BGFM Boltzmann numbers are *within-basin*: perturb one data
geometry by sigma=0.15 A and correlate log p_theta with -E/kT.  That measures
local ranking only.  Boltzmann is a much stronger statement:

    p(x) = exp(-E(x)/kT)/Z   <=>   u(x) := log p_theta(x) + E(x)/kT = -log Z

i.e. u must be the SAME constant in every basin, not just constant-ish inside
each one.  A model can score r ~ 0.9 within every basin and still be arbitrarily
wrong about p(basin A)/p(basin B).

What this script does (option "FFJORD on basin representatives", the route the
advisor flagged as most directly answering the question -- see NOTE below on why
we cannot instead sample composition-conditioned geometries from FlowMol3):

  stage `logp`  (GPU, envs/flowmol)
      For each system and each xTB basin minimum x*_b produced by
      scripts/build_reference_ensembles.py, evaluate log p_theta at
        - the minimum itself, and
        - m Gaussian perturbations x*_b + delta,  delta ~ N(0, sigma^2) in the
          COM-free subspace   (these are IS samples from a KNOWN proposal q_b)
      via the FFJORD reverse-time ODE (cfm_mol.bgfm_density.log_density_via_flow).

  stage `analyze`  (CPU, envs/flowmol; needs xtb)
      GFN2-xTB single points on every geometry, then:

      (1) POINTWISE ACROSS BASINS
          regress log p_theta(x*_b) on -beta E(x*_b) over basins.
          slope (ideal 1), Pearson r, and the calibration metric
          NRV = Var_b[log p + beta E] / Var_b[beta E]   (ideal 0).

      (2) BASIN MASS (this is the real "relative probability of a basin")
          Self-normalised importance sampling with the SAME Gaussian proposal
          q_b for every basin, so the (basin-independent) normalising constant
          of q cancels:
              log Zhat_theta(b) = LSE_i[ log p_theta(x_i) - log q(x_i) ] - log m
              log Zhat_ref(b)   = LSE_i[ -beta E(x_i)     - log q(x_i) ] - log m
          Occupancies P_theta = softmax_b log Zhat_theta, P_ref likewise.
          Reported: KL(P_ref || P_theta), total variation, W1 on the basin-energy
          axis (eV), importance-weight ESS fraction over basins, top-1 agreement,
          and the regression of all pairwise Delta log Zhat_theta on
          Delta log Zhat_ref (ideal slope 1).

      (3) WITHIN- vs BETWEEN-BASIN DECOMPOSITION
          u = log p_theta + beta E.  Var(u) splits into within-basin and
          between-basin parts.  The headline diagnostic of this experiment is
          whether a model that looks good within basins is bad between them.

NOTE on "sample conditional geometry from the model with fixed atom types"
--------------------------------------------------------------------------
FlowMol3's sampler jointly generates (x, a, c, e) from the prior; there is no
composition-conditioning API, and the CTMC discrete channels are resampled every
step, so a plain `model.sample()` cannot be pinned to a given molecular formula
without new conditional-sampling machinery (an inpainting/guidance scheme on the
discrete channels).  We therefore evaluate the density directly on reference
geometries, which needs no new sampler and answers the scientific question
(relative basin mass) more directly than sampling would at this sample budget.
If composition-conditioned sampling is added later, the natural follow-up is to
compare model-sampled basin occupancies against the same xTB reference; the
metrics below (KL / TV / W1 / ESS over basin occupancies) are written so that a
sampled occupancy vector can be dropped in for P_theta unchanged.

Usage
-----
  # stage 1 (GPU)
  $FLOWMOL_PY scripts/eval_global_ensemble.py logp \
      --checkpoint <ckpt> --config <cfg> \
      --ensembles_dir <.../ensembles> --out_dir <.../ge_<arm>> \
      --n_perturb 8 --sigma 0.15 --n_ode_steps 12 --n_hutchinson 2 --n_repeats 2

  # stage 2 (CPU, xtb)
  $FLOWMOL_PY scripts/eval_global_ensemble.py analyze \
      --logp_json <.../ge_<arm>/basin_logp.json> --out_dir <.../ge_<arm>>
"""
from __future__ import annotations

import argparse
import json
import math
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

_PERIODIC_SYMBOLS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
]
SYMBOL_TO_Z = {s: i + 1 for i, s in enumerate(_PERIODIC_SYMBOLS)}


# ==========================================================================
# stage 1 : log p_theta on basin representatives
# ==========================================================================
def stage_logp(args) -> int:
    import torch
    import dgl
    from flowmol.model_utils.load import model_from_config, read_config_file
    from flowmol.data_processing.dataset import MoleculeDataset
    from flowmol.data_processing.utils import get_batch_idxs, get_upper_edge_mask
    from cfm_mol.bgfm_density import log_density_via_flow

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    ens = json.load(open(args.ensembles_dir / "ensembles.json"))
    geoms = np.load(args.ensembles_dir / "geoms.npz")
    systems = ens["systems"][: args.n_systems] if args.n_systems > 0 else ens["systems"]
    print(f"[ge-logp] {len(systems)} systems from {args.ensembles_dir}", flush=True)

    cfg = read_config_file(args.config)
    cfg.get("mol_fm", {}).pop("bgfm", None)
    cfg["dataset"]["processed_data_dir"] = str(args.eval_data)
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
        d_min[:n_real, :n_real] = default_d_min_table(n_atom_types=n_real, atom_map=atom_map)
        e_w = cfg["mol_fm"].get("total_loss_weights", {}).get("e", 2.0)
        bond_free = float(e_w) == 0.0
        patch_flowmol(model, d_min, tangent=True, retract=True, gluing=True,
                      discrete_projection=not bond_free,
                      train_time_discrete=not bond_free, atom_map=atom_map)

    state = torch.load(str(args.checkpoint), map_location="cpu")
    sd = state.get("state_dict", state)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    n_bad = sum(int((~torch.isfinite(v)).sum()) for v in sd.values()
                if torch.is_tensor(v) and v.is_floating_point())
    print(f"[ge-logp] ckpt missing={len(missing)} unexpected={len(unexpected)} "
          f"non_finite_params={n_bad}", flush=True)
    if n_bad:
        raise RuntimeError("checkpoint has non-finite parameters")
    model = model.to(device).eval()

    ds_cfg = dict(cfg["dataset"])
    ds_cfg["fake_atom_p"] = 0.0
    ds_cfg["fake_atom_std"] = 1.0
    ds_cfg["explicit_aromaticity"] = cfg["mol_fm"].get("explicit_aromaticity", False)
    val = MoleculeDataset("val", ds_cfg, prior_config=cfg["mol_fm"]["prior_config"])

    # Resume support: gpu_requeue preempts, and one system costs minutes, so the
    # per-system records are flushed to a partial file and reloaded on restart.
    partial_path = args.out_dir / "basin_logp_partial.json"
    records, done_systems = [], set()
    if partial_path.exists() and not args.restart:
        try:
            prev = json.load(open(partial_path))
            records = prev.get("records", [])
            done_systems = {int(r["system_id"]) for r in records}
            print(f"[ge-logp] resuming: {len(done_systems)} systems already done",
                  flush=True)
        except Exception as exc:
            print(f"[ge-logp] could not read partial file ({exc}); starting fresh",
                  flush=True)

    for si, sysrec in enumerate(systems):
        vi = sysrec["val_index"]
        if int(vi) in done_systems:
            continue
        symbols = sysrec["symbols"]
        charge = int(sysrec.get("charge", 0))
        basin_pos = geoms[f"pos_{vi}"].astype(np.float64)   # (n_basin, N, 3)
        nb, N, _ = basin_pos.shape

        g0 = val[vi].to(device)
        if g0.num_nodes() != N:
            print(f"[ge-logp] SKIP system {vi}: dataset N={g0.num_nodes()} != {N}", flush=True)
            continue
        # Consistency: the ensemble was built from this val molecule, so the
        # atom ordering must be identical. Assert, do not hope.
        ds_syms = [atom_map[int(i)] for i in g0.ndata["a_1_true"].argmax(-1).cpu().numpy()]
        if ds_syms != list(symbols):
            print(f"[ge-logp] SKIP system {vi}: atom order/type mismatch", flush=True)
            continue
        z = [SYMBOL_TO_Z[s] for s in symbols]
        total_charge = int((g0.ndata["c_1_true"].argmax(-1) - 2).sum().item())

        # ---- build all geometries: per basin, minimum + m Gaussian perturbations
        geo_list, tags = [], []
        gen = torch.Generator(device="cpu").manual_seed(args.seed * 100003 + vi)
        for b in range(nb):
            x0 = torch.tensor(basin_pos[b], dtype=torch.float32)
            x0 = x0 - x0.mean(0, keepdim=True)
            geo_list.append(x0)
            tags.append((b, 0, 0.0))
            for p in range(args.n_perturb):
                noise = torch.randn(x0.shape, generator=gen) * args.sigma
                noise = noise - noise.mean(0, keepdim=True)   # stay COM-free
                xp = x0 + noise
                geo_list.append(xp)
                # ||delta||^2 in the COM-free subspace -> log q up to a constant
                tags.append((b, p + 1, float((noise ** 2).sum())))

        # ---- FFJORD log p (batched, repeated for estimator-noise control)
        #
        # COMMON RANDOM NUMBERS. The Hutchinson divergence error on a single
        # log p is several nats, while the inter-basin Boltzmann signal at
        # kT = 1 eV is O(0.1-1) nat -- absolute log p is useless here. But
        # every geometry in a system is the SAME molecule, so we can tile one
        # probe over all graphs in the batch and reuse the same probe sequence
        # for every chunk. The estimator error then becomes almost entirely
        # common-mode and cancels in the log p DIFFERENCES that all our
        # metrics are built from. `--no_crn` disables this for a control.
        #
        # EXACT BLOCKED DIVERGENCE (`--exact_div`, the default). Even with CRN the
        # Hutchinson error only halves, because the error term sum_{i!=j} xi_i xi_j
        # J_ij changes as fast as J does between geometries. So instead make the
        # divergence exact. Trick: the batched graph is block-diagonal (graphs do
        # not interact), so a probe that is sqrt(3N) * e_{(i,c)} REPLICATED IN EVERY
        # GRAPH yields, per graph, exactly 3N * J^{(g)}_{(i,c),(i,c)}. Averaging the
        # 3N such probes (which is what divergence_hutchinson does) gives the exact
        # trace for EVERY graph at a cost of 3N backward passes TOTAL -- independent
        # of how many geometries are in the batch. Zero stochastic error.
        n_hutch = 3 * N if args.exact_div else args.n_hutchinson
        n_reps = 1 if args.exact_div else args.n_repeats
        logp_reps = []
        for rep in range(n_reps):
            torch.manual_seed(args.seed * 7919 + rep * 13 + vi)
            probe_cache = {}
            pgen = torch.Generator(device="cpu").manual_seed(
                args.seed * 104729 + rep * 7717 + vi)
            scale = math.sqrt(3.0 * N)

            def xi_fn(step, k, xx, _cache=probe_cache, _g=pgen, _N=N, _sc=scale):
                if args.exact_div:
                    base = torch.zeros(_N, 3, dtype=torch.float32)
                    base[k // 3, k % 3] = _sc
                else:
                    key = (step, k)
                    if key not in _cache:
                        _cache[key] = (torch.randint(0, 2, (_N, 3), generator=_g,
                                                     dtype=torch.float32) * 2 - 1)
                    base = _cache[key]
                base = base.to(device=xx.device, dtype=xx.dtype)
                return base.repeat(xx.shape[0] // _N, 1)

            reps = []
            for start in range(0, len(geo_list), args.batch_geoms):
                chunk = geo_list[start:start + args.batch_geoms]
                graphs = []
                for x in chunk:
                    gp = g0.clone()
                    gp.ndata["x_1_true"] = x.to(device)
                    graphs.append(gp)
                gb = dgl.batch(graphs).to(device)
                gb.ndata["x_t"] = gb.ndata["x_1_true"]
                gb.ndata["a_t"] = gb.ndata["a_1_true"]
                gb.ndata["c_t"] = gb.ndata["c_1_true"]
                gb.edata["e_t"] = gb.edata["e_1_true"]
                nbi, _ = get_batch_idxs(gb)
                uem = get_upper_edge_mask(gb)
                with torch.enable_grad():
                    lp = log_density_via_flow(
                        model, gb, nbi, uem, n_ode_steps=args.n_ode_steps,
                        n_hutchinson=n_hutch, prior_std=1.0,
                        xi_fn=(None if (args.no_crn and not args.exact_div)
                               else xi_fn))
                reps.append(lp.detach().cpu().numpy())
            logp_reps.append(np.concatenate(reps))
        L = np.stack(logp_reps)                      # (n_reps, n_geoms)
        logp_mean = L.mean(0)
        logp_std = L.std(0, ddof=1) if n_reps > 1 else np.zeros_like(logp_mean)
        # The quantity that actually limits this experiment is the noise on log p
        # DIFFERENCES within a system, not on absolute log p. Measure it by
        # centring each repeat before taking the across-repeat std.
        if n_reps > 1:
            Lc = L - L.mean(axis=1, keepdims=True)
            diff_noise = float(np.mean(Lc.std(0, ddof=1)))
        else:
            diff_noise = float("nan")

        for k, (b, p, dsq) in enumerate(tags):
            records.append({
                "system_id": int(vi), "basin_id": int(b), "pert_id": int(p),
                "atomic_numbers": z,
                "positions": [[float(v) for v in row]
                              for row in geo_list[k].numpy()],
                "charge": total_charge, "declared_charge": charge,
                "disp_sq": dsq,
                "log_p_theta": float(logp_mean[k]),
                "log_p_std": float(logp_std[k]),
                "ref_energy_eV_basin_min": (
                    float(sysrec["basins"][b]["energy_eV"]) if p == 0 else None),
                "logp_diff_noise": diff_noise,
                # confound check: is the basin the OMol25 data geometry relaxed
                # into? A model that merely memorises training geometries would
                # top-rank this basin regardless of its energy.
                "from_data_basin": bool(
                    sysrec["basins"][b].get("from_data_geometry", False)),
            })
        print(f"[ge-logp] {si+1}/{len(systems)} val_index={vi} basins={nb} "
              f"geoms={len(geo_list)} abs_logp_repeat_std="
              f"{float(np.mean(logp_std)):.3f} DIFF_noise={diff_noise:.4f}",
              flush=True)
        tmp = partial_path.with_suffix(".tmp")
        # Same schema as the final file, so `analyze` can consume the partial
        # directly if the job is preempted before every system is done.
        json.dump({"sigma": args.sigma, "n_perturb": args.n_perturb,
                   "n_ode_steps": args.n_ode_steps,
                   "n_hutchinson": args.n_hutchinson,
                   "exact_div": bool(args.exact_div),
                   "checkpoint": str(args.checkpoint),
                   "ensembles_dir": str(args.ensembles_dir),
                   "records": records}, open(tmp, "w"))
        os.replace(tmp, partial_path)   # atomic: never leave a half-written file

    out = args.out_dir / "basin_logp.json"
    json.dump({"sigma": args.sigma, "n_perturb": args.n_perturb,
               "n_ode_steps": args.n_ode_steps,
               "divergence": ("exact_blocked" if args.exact_div
                              else f"hutchinson_{args.n_hutchinson}"
                                   f"{'_crn' if not args.no_crn else ''}"),
               "n_repeats": args.n_repeats,
               "checkpoint": str(args.checkpoint),
               "ensembles_dir": str(args.ensembles_dir),
               "records": records}, open(out, "w"))
    print(f"[ge-logp] wrote {out} ({len(records)} geometries)", flush=True)
    return 0


# ==========================================================================
# stage 2 : xTB energies + metrics
# ==========================================================================
def _xtb_sp(task):
    """Single-point GFN2-xTB energy in eV, or None."""
    z, pos, charge = task
    xtb = shutil.which("xtb")
    if xtb is None:
        return None
    syms = [_PERIODIC_SYMBOLS[i - 1] for i in z]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        with open(td / "m.xyz", "w") as f:
            f.write(f"{len(syms)}\n\n")
            for s, p in zip(syms, pos):
                f.write(f"{s} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        try:
            out = subprocess.run(
                [xtb, "m.xyz", "--gfn", "2", "--chrg", str(int(charge)), "--silent"],
                cwd=td, capture_output=True, text=True, timeout=300, env=env)
        except (subprocess.TimeoutExpired, OSError):
            return None
    es = [float(m.group(1)) for m in _E_LINE.finditer(out.stdout)]
    return es[-1] * HARTREE_TO_EV if es else None


def _lse(a):
    a = np.asarray(a, float)
    m = a.max()
    return float(m + np.log(np.exp(a - m).sum())) if np.isfinite(m) else float("-inf")


def _lin(x, y):
    """slope, intercept, pearson r of y on x (nan-safe)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan"), float("nan"), float("nan")
    s, i = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    return float(s), float(i), r


def _partial_r(y, x, z):
    """Partial correlation r(y, x | z) -- controls for a nuisance covariate.

    Used to rule out the "the model just likes compact/typical geometries"
    confound: if log p tracks the radius of gyration rather than the energy,
    r(log p, -beta E | Rg) collapses while the raw r stays high.
    """
    y, x, z = map(lambda v: np.asarray(v, float), (y, x, z))
    if y.size < 4 or min(np.std(y), np.std(x), np.std(z)) < 1e-12:
        return float("nan")
    ryx = np.corrcoef(y, x)[0, 1]
    ryz = np.corrcoef(y, z)[0, 1]
    rxz = np.corrcoef(x, z)[0, 1]
    den = math.sqrt(max(1e-12, (1 - ryz ** 2) * (1 - rxz ** 2)))
    return float((ryx - ryz * rxz) / den)


def _rg(positions):
    p = np.asarray(positions, float)
    p = p - p.mean(0, keepdims=True)
    return float(np.sqrt((p ** 2).sum(1).mean()))


def _nrv(logp, beta_E):
    """Normalised residual variance Var[log p + beta E]/Var[beta E] (ideal 0)."""
    u = np.asarray(logp, float) + np.asarray(beta_E, float)
    vd = np.var(beta_E)
    return float(np.var(u) / vd) if vd > 1e-12 else float("nan")


def _w1_on_energy(E, p, q):
    """1-D Wasserstein-1 between two discrete distributions supported on the
    basin energies E (eV). Ground metric = |E_i - E_j| in eV."""
    order = np.argsort(E)
    Es, ps, qs = np.asarray(E)[order], np.asarray(p)[order], np.asarray(q)[order]
    cp, cq = np.cumsum(ps), np.cumsum(qs)
    return float(np.sum(np.abs(cp[:-1] - cq[:-1]) * np.diff(Es)))


def stage_analyze(args) -> int:
    if shutil.which("xtb") is None:
        print("[ge-analyze] ERROR: xtb not on PATH", file=sys.stderr)
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)
    blob = json.load(open(args.logp_json))
    recs = blob["records"]
    sigma = float(blob["sigma"])

    # ---- xTB single points (cache next to the logp json) -----------------
    cache = args.out_dir / "xtb_energies.json"
    E = None
    if cache.exists() and not args.recompute_xtb:
        E = json.load(open(cache))
        if len(E) != len(recs):
            # The logp file grew (resumed run added systems). A stale cache would
            # silently zip-truncate and leave later records without an energy.
            print(f"[ge-analyze] cache has {len(E)} entries but {len(recs)} "
                  f"records -- recomputing", flush=True)
            E = None
        else:
            print(f"[ge-analyze] loaded {len(E)} cached xTB energies", flush=True)
    if E is None:
        tasks = [(r["atomic_numbers"], r["positions"], r["charge"]) for r in recs]
        with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
            E = list(ex.map(_xtb_sp, tasks, chunksize=4))
        json.dump(E, open(cache, "w"))
        print(f"[ge-analyze] computed {sum(e is not None for e in E)}/{len(E)} "
              f"xTB single points", flush=True)
    for r, e in zip(recs, E):
        r["E_eV"] = e

    beta = 1.0 / args.kT_eV
    log_kT_note = f"kT={args.kT_eV} eV (beta={beta:.4g} 1/eV)"

    # ---- group by system / basin ----------------------------------------
    systems = {}
    for r in recs:
        if r["E_eV"] is None or not np.isfinite(r["log_p_theta"]):
            continue
        systems.setdefault(r["system_id"], {}).setdefault(r["basin_id"], []).append(r)

    rows, pooled_pair_x, pooled_pair_y = [], [], []
    for sid, basins in sorted(systems.items()):
        bids = sorted(basins)
        if len(bids) < args.min_basins:
            continue
        # ---------- (1) pointwise at the basin minima
        mins = {}
        for b in bids:
            m = [r for r in basins[b] if r["pert_id"] == 0]
            if m:
                mins[b] = m[0]
        if len(mins) < args.min_basins:
            continue
        bm = sorted(mins)
        lp_min = np.array([mins[b]["log_p_theta"] for b in bm])
        E_min = np.array([mins[b]["E_eV"] for b in bm])
        x = -beta * E_min
        slope_pt, _, r_pt = _lin(x, lp_min)
        nrv_pt = _nrv(lp_min, beta * E_min)
        rg_min = np.array([_rg(mins[b]["positions"]) for b in bm])
        r_pt_partial_rg = _partial_r(lp_min, x, rg_min)

        # ---------- (2) basin mass by importance sampling
        logZ_t, logZ_r, Eb, ok_b = [], [], [], []
        within_r, within_nrv, is_ess_t, is_ess_r = [], [], [], []
        for b in bm:
            pert = [r for r in basins[b] if r["pert_id"] > 0]
            if len(pert) < 3:
                continue
            lp = np.array([r["log_p_theta"] for r in pert])
            Ee = np.array([r["E_eV"] for r in pert])
            dsq = np.array([r["disp_sq"] for r in pert])
            log_q = -dsq / (2.0 * sigma ** 2)          # + basin-independent const
            m = len(pert)
            logZ_t.append(_lse(lp - log_q) - math.log(m))
            logZ_r.append(_lse(-beta * Ee - log_q) - math.log(m))
            # How trustworthy is that IS estimate? ESS of the self-normalised
            # weights w_i = f(x_i)/q(x_i). ESS/m near 1 = well resolved; near 1/m
            # = one sample carries all the mass and log Zhat is essentially the
            # max, i.e. pure proposal noise.
            for lw, acc in ((lp - log_q, is_ess_t), (-beta * Ee - log_q, is_ess_r)):
                acc.append(math.exp(2 * _lse(lw) - _lse(2 * lw)) / m)
            Eb.append(mins[b]["E_eV"])
            ok_b.append(b)
            _, _, rw = _lin(-beta * Ee, lp)
            within_r.append(rw)
            within_nrv.append(_nrv(lp, beta * Ee))
        if len(ok_b) < args.min_basins:
            continue
        logZ_t, logZ_r, Eb = np.array(logZ_t), np.array(logZ_r), np.array(Eb)
        Pt = np.exp(logZ_t - _lse(logZ_t))
        Pr = np.exp(logZ_r - _lse(logZ_r))
        eps = 1e-300
        kl = float(np.sum(Pr * (np.log(Pr + eps) - np.log(Pt + eps))))
        kl_rev = float(np.sum(Pt * (np.log(Pt + eps) - np.log(Pr + eps))))
        tv = float(0.5 * np.abs(Pr - Pt).sum())
        w1 = _w1_on_energy(Eb, Pr, Pt)
        w = Pr / np.maximum(Pt, eps)
        ess = float((w.sum() ** 2) / (len(w) * np.sum(w ** 2)))
        top1 = bool(int(np.argmax(Pt)) == int(np.argmax(Pr)))
        data_basins = {b for b in ok_b
                       if any(r.get("from_data_basin") for r in basins[b])}
        model_top1_is_data = (int(ok_b[int(np.argmax(Pt))]) in data_basins
                              if data_basins else None)
        ref_top1_is_data = (int(ok_b[int(np.argmax(Pr))]) in data_basins
                            if data_basins else None)
        # pairwise Delta regression (all i<j)
        px, py = [], []
        for i in range(len(ok_b)):
            for j in range(i + 1, len(ok_b)):
                px.append(logZ_r[i] - logZ_r[j])
                py.append(logZ_t[i] - logZ_t[j])
        slope_mass, _, r_mass = _lin(px, py)
        pooled_pair_x += px
        pooled_pair_y += py
        nrv_mass = (float(np.var(logZ_t - logZ_r) / np.var(logZ_r))
                    if np.var(logZ_r) > 1e-12 else float("nan"))
        # PRINCIPAL free-energy error. For a Boltzmann model
        #   log Zhat_theta(b) - log Zhat_ref(b) = -log Z   for EVERY basin b,
        # so the spread of that difference across basins IS the error on relative
        # basin free energies (in nats; x kT -> eV). It includes the local
        # entropy of each basin, because both Z-hats integrate the same Gaussian
        # cloud. Its yardstick is the true spread sqrt(Var_b[log Zhat_ref]) * kT:
        # a model that is FLAT across basins scores ratio = 1, Boltzmann = 0.
        dF_err_mass_eV = float(np.std(logZ_t - logZ_r) * args.kT_eV)
        dF_true_mass_eV = float(np.std(logZ_r) * args.kT_eV)
        # Minima-only variant (0 K / no entropy): u_b at the basin minimum.
        u_min = lp_min + beta * E_min
        dF_err_min_eV = float(np.std(u_min) * args.kT_eV)
        dF_true_min_eV = float(np.std(beta * E_min) * args.kT_eV)

        # ---------- (3) within/between variance decomposition of u = logp + beta E
        allr = [r for b in ok_b for r in basins[b]]
        u_all = np.array([r["log_p_theta"] + beta * r["E_eV"] for r in allr])
        be_all = np.array([beta * r["E_eV"] for r in allr])
        gb = np.array([r["basin_id"] for r in allr])
        u_within, be_within, u_means, be_means = [], [], [], []
        for b in ok_b:
            sel = gb == b
            if sel.sum() < 2:
                continue
            u_within.append(np.var(u_all[sel]))
            be_within.append(np.var(be_all[sel]))
            u_means.append(u_all[sel].mean())
            be_means.append(be_all[sel].mean())
        var_u_within = float(np.mean(u_within))
        var_u_between = float(np.var(u_means))
        var_be_within = float(np.mean(be_within))
        var_be_between = float(np.var(be_means))
        nrv_within = var_u_within / var_be_within if var_be_within > 1e-12 else float("nan")
        nrv_between = var_u_between / var_be_between if var_be_between > 1e-12 else float("nan")
        # NOTE: var_be_between here is the spread of the basin-mean beta*E over ALL
        # geometries, which is dominated by the RANDOM energies of the sigma=0.15 A
        # perturbations (they sit 1-3 eV above the minimum), NOT by basin free
        # energy differences. So nrv_between is only a decomposition diagnostic --
        # the interpretable free-energy errors are dF_err_mass_eV (principal) and
        # dF_err_min_eV (0 K), computed above.
        dF_err_eV = float(math.sqrt(var_u_between) * args.kT_eV)
        dF_true_eV = float(math.sqrt(var_be_between) * args.kT_eV)

        rows.append(dict(
            system_id=sid, n_basins=len(ok_b),
            basin_E_spread_eV=float(Eb.max() - Eb.min()),
            # (1)
            r_pointwise=r_pt, slope_pointwise=slope_pt, nrv_pointwise=nrv_pt,
            r_pointwise_partial_Rg=r_pt_partial_rg,
            r_logp_vs_Rg=float(np.corrcoef(lp_min, rg_min)[0, 1])
            if np.std(rg_min) > 1e-12 else float("nan"),
            # (2)
            r_basin_mass=r_mass, slope_basin_mass=slope_mass, nrv_basin_mass=nrv_mass,
            kl_ref_model=kl, kl_model_ref=kl_rev, tv=tv, w1_energy_eV=w1,
            ess_frac_basins=ess, top1_match=top1,
            model_top1_is_data_basin=model_top1_is_data,
            ref_top1_is_data_basin=ref_top1_is_data,
            # (3)
            mean_within_basin_r=float(np.nanmean(within_r)),
            mean_within_basin_nrv=float(np.nanmean(within_nrv)),
            var_u_within=var_u_within, var_u_between=var_u_between,
            nrv_within=nrv_within, nrv_between=nrv_between,
            # Effective temperature. If log p = -E/kT_eff + c, regressing log p on
            # x = -E/kT_eval returns slope = kT_eval/kT_eff, so kT_eff =
            # kT_eval/slope. This is invariant to the (arbitrary) kT_eval used for
            # the report, and directly answers "at what temperature is the model
            # Boltzmann across basins?".
            kT_eff_pointwise_eV=(args.kT_eV / slope_pt
                                 if abs(slope_pt) > 1e-9 else float("nan")),
            kT_eff_basin_mass_eV=(args.kT_eV / slope_mass
                                  if abs(slope_mass) > 1e-9 else float("nan")),
            is_ess_frac_ref=float(np.mean(is_ess_r)),
            is_ess_frac_model=float(np.mean(is_ess_t)),
            dF_err_mass_eV=dF_err_mass_eV, dF_true_mass_eV=dF_true_mass_eV,
            dF_err_mass_ratio=(dF_err_mass_eV / dF_true_mass_eV
                               if dF_true_mass_eV > 1e-9 else float("nan")),
            dF_err_min_eV=dF_err_min_eV, dF_true_min_eV=dF_true_min_eV,
            dF_err_min_ratio=(dF_err_min_eV / dF_true_min_eV
                              if dF_true_min_eV > 1e-9 else float("nan")),
            dF_err_eV=dF_err_eV, dF_true_spread_eV=dF_true_eV,
            dF_err_ratio=(dF_err_eV / dF_true_eV if dF_true_eV > 1e-9 else float("nan")),
            frac_var_u_between=(var_u_between / (var_u_between + var_u_within)
                                if (var_u_between + var_u_within) > 0 else float("nan")),
        ))

    if not rows:
        print("[ge-analyze] no systems survived the filters", flush=True)
        return 1

    def agg(k):
        v = np.array([r[k] for r in rows], dtype=float)
        v = v[np.isfinite(v)]
        return dict(mean=float(v.mean()), median=float(np.median(v)),
                    sem=float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0,
                    n=int(len(v)))

    slope_pool, _, r_pool = _lin(pooled_pair_x, pooled_pair_y)
    summary = {
        "kT_note": log_kT_note,
        "kT_eV": args.kT_eV,
        "checkpoint": blob.get("checkpoint"),
        "n_systems": len(rows),
        "sigma": sigma,
        "pooled_basin_pair_slope": slope_pool,
        "pooled_basin_pair_r": r_pool,
        "pooled_basin_pair_n": len(pooled_pair_x),
        "frac_top1_match": float(np.mean([r["top1_match"] for r in rows])),
    }
    for key in ["model_top1_is_data_basin", "ref_top1_is_data_basin"]:
        vals = [r[key] for r in rows if r.get(key) is not None]
        summary[f"frac_{key}"] = float(np.mean(vals)) if vals else None
    for k in ["r_pointwise", "slope_pointwise", "nrv_pointwise",
              "r_pointwise_partial_Rg", "r_logp_vs_Rg",
              "r_basin_mass", "slope_basin_mass", "nrv_basin_mass",
              "kl_ref_model", "kl_model_ref", "tv", "w1_energy_eV",
              "ess_frac_basins", "mean_within_basin_r", "mean_within_basin_nrv",
              "nrv_within", "nrv_between", "frac_var_u_between",
              "var_u_within", "var_u_between", "n_basins", "basin_E_spread_eV",
              "dF_err_mass_eV", "dF_true_mass_eV", "dF_err_mass_ratio",
              "is_ess_frac_ref", "is_ess_frac_model",
              "dF_err_min_eV", "dF_true_min_eV", "dF_err_min_ratio",
              "dF_err_eV", "dF_true_spread_eV", "dF_err_ratio",
              "kT_eff_pointwise_eV", "kT_eff_basin_mass_eV"]:
        summary[k] = agg(k)

    json.dump({"summary": summary, "per_system": rows},
              open(args.out_dir / "global_ensemble_results.json", "w"), indent=2)
    import csv
    with open(args.out_dir / "global_ensemble_per_system.csv", "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    # per-record dump so downstream re-analysis (other kT, other estimators)
    # never needs to re-run xtb
    with open(args.out_dir / "global_ensemble_records.csv", "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["system_id", "basin_id", "pert_id", "n_atoms", "charge",
                      "disp_sq", "log_p_theta", "log_p_std", "E_eV"])
        for r in recs:
            wtr.writerow([r["system_id"], r["basin_id"], r["pert_id"],
                          len(r["atomic_numbers"]), r["charge"], r["disp_sq"],
                          r["log_p_theta"], r.get("log_p_std", ""), r["E_eV"]])
    # ---- figure: pooled basin-pair Delta log Z scatter ---------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 4))
        ax[0].scatter(pooled_pair_x, pooled_pair_y, s=8, alpha=0.5)
        lim = max(1e-6, float(np.percentile(np.abs(pooled_pair_x), 99)))
        ax[0].plot([-lim, lim], [-lim, lim], "k--", lw=1, label="ideal (slope 1)")
        ax[0].set_xlabel(r"$\Delta \log \hat{Z}_{\rm ref}$ (xTB, nats)")
        ax[0].set_ylabel(r"$\Delta \log \hat{Z}_{\theta}$ (model, nats)")
        ax[0].set_title(f"basin-pair mass  slope={slope_pool:.3f} r={r_pool:.3f}")
        ax[0].legend(fontsize=8)
        wi = [r["mean_within_basin_r"] for r in rows]
        be = [r["r_basin_mass"] for r in rows]
        ax[1].scatter(wi, be, s=18)
        ax[1].axhline(0, color="k", lw=0.5); ax[1].axvline(0, color="k", lw=0.5)
        ax[1].plot([-1, 1], [-1, 1], "k--", lw=1)
        ax[1].set_xlabel("within-basin r (the OLD metric)")
        ax[1].set_ylabel("between-basin r (basin mass)")
        ax[1].set_title("local vs global Boltzmann fidelity")
        fig.tight_layout()
        fig.savefig(args.out_dir / "global_ensemble.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"[ge-analyze] figure skipped: {exc}")

    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)

    p1 = sub.add_parser("logp")
    p1.add_argument("--checkpoint", type=Path, required=True)
    p1.add_argument("--config", type=Path, required=True)
    p1.add_argument("--ensembles_dir", type=Path, required=True)
    p1.add_argument("--eval_data", type=Path,
                    default=Path("/n/holylabs/woo_lab/Lab/yulili/bgfm/processed_data/omol25_4m_processed"))
    p1.add_argument("--out_dir", type=Path, required=True)
    p1.add_argument("--n_systems", type=int, default=0)
    p1.add_argument("--n_perturb", type=int, default=8)
    p1.add_argument("--sigma", type=float, default=0.15)
    p1.add_argument("--n_ode_steps", type=int, default=12)
    p1.add_argument("--n_hutchinson", type=int, default=2)
    p1.add_argument("--n_repeats", type=int, default=2)
    p1.add_argument("--batch_geoms", type=int, default=36)
    p1.add_argument("--device", default=None)
    p1.add_argument("--seed", type=int, default=0)
    p1.add_argument("--no_patch", action="store_true")
    p1.add_argument("--no_crn", action="store_true",
                    help="disable common-random-number Hutchinson probes "
                         "(control: shows how much CRN buys)")
    p1.add_argument("--exact_div", action="store_true", default=True,
                    help="exact blocked divergence (3N backward passes per ODE "
                         "step, batch-size independent, zero stochastic error)")
    p1.add_argument("--stochastic_div", dest="exact_div", action="store_false",
                    help="use the Hutchinson estimator instead of exact")
    p1.add_argument("--restart", action="store_true",
                    help="ignore basin_logp_partial.json and recompute everything")

    p2 = sub.add_parser("analyze")
    p2.add_argument("--logp_json", type=Path, required=True)
    p2.add_argument("--out_dir", type=Path, required=True)
    p2.add_argument("--kT_eV", type=float, default=1.0)
    p2.add_argument("--min_basins", type=int, default=3)
    p2.add_argument("--n_workers", type=int, default=8)
    p2.add_argument("--recompute_xtb", action="store_true")

    a = ap.parse_args()
    return stage_logp(a) if a.stage == "logp" else stage_analyze(a)


if __name__ == "__main__":
    raise SystemExit(main())
