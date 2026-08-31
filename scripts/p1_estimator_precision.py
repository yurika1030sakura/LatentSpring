"""P1 -- numerical precision of the log-density estimator (no training).

Two questions, both about the ESTIMATOR, not the model:

  TASK A  How much error does the production 2-probe Hutchinson divergence carry
          relative to the exact autograd Jacobian diagonal?
          -> bias, repeat variance, relative error, per ODE node AND accumulated
             into log q.

  TASK B  If the same geometries are scored several times with independent
          probes, how much does log q move, and how does that compare with the
          within-parent spread of log q that the per-parent ordering metric
          (Pearson r of log q against -E/kT over the 8 displaced geometries)
          actually depends on?

Everything runs on FROZEN geometries taken verbatim from the reported
population (runs/eval_ours/wide_*/boltz1/boltzmann_samples.json), at the
production estimator setting n_ode_steps=12 / n_hutchinson=2 / rademacher.

Key exactness note for TASK B
-----------------------------
In ``log_density_via_flow`` the reverse Euler state update uses only the
velocity, computed under ``torch.no_grad()``; the Hutchinson probes never touch
the trajectory.  The state path is therefore a deterministic function of the
model and x_1.  Consequently R independent probe draws evaluated along ONE
trajectory are distributionally IDENTICAL to R independent full calls of
``log_density_via_flow`` -- we exploit that to make repeats cheap.  The
``--verify_production`` flag checks it empirically against the real function.

Usage (envs/flowmol, GPU):
  python scripts/p1_estimator_precision.py --task all \
      --checkpoint <ckpt> --config configs/sweep/a3_energy_only.yaml \
      --eval_data <processed_dir> \
      --samples_json runs/eval_ours/wide_a3_energy_only_s2/boltz1/boltzmann_samples.json \
      --records_csv runs/eval_ours/wide_a3_energy_only_s2/boltz_records.csv \
      --disjoint_json paper/figures/out/primary_endpoint.json \
      --out_json runs/eval_ours/p1_precision/<tag>.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for p in (_REPO, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# 0.  Toy analytic control (CPU, no model, no checkpoint)
# ---------------------------------------------------------------------------
def task_toy(n_atoms: int = 6, n_repeat: int = 20000, seed: int = 0) -> dict:
    """Linear field v(x) = (A x_flat).reshape(N,3): trace is known in closed form.

    Checks (i) divergence_exact_atomwise reproduces tr(A) to machine precision,
    (ii) the 2-probe Rademacher Hutchinson estimator is unbiased with the
    analytic variance  Var = (1/k) * sum_{i<j} (A_ij + A_ji)^2 .
    """
    from cfm_mol.bgfm_loss import divergence_exact_atomwise, divergence_hutchinson

    torch.manual_seed(seed)
    d = 3 * n_atoms
    A = torch.randn(d, d, dtype=torch.float64) / math.sqrt(d)
    x = torch.randn(n_atoms, 3, dtype=torch.float64, requires_grad=True)
    n_apg = torch.tensor([n_atoms])

    def v_fn(z):
        return (A @ z.reshape(-1)).reshape(n_atoms, 3)

    tr_analytic = float(torch.diagonal(A).sum())
    tr_exact = float(divergence_exact_atomwise(v_fn, x, n_apg, create_graph=False)[0])

    var_analytic = 0.0
    for i in range(d):
        for j in range(i + 1, d):
            var_analytic += float((A[i, j] + A[j, i]) ** 2)

    est = []
    for _ in range(n_repeat):
        est.append(float(divergence_hutchinson(
            v_fn, x.detach().requires_grad_(True), n_apg,
            n_samples=2, rademacher=True, create_graph=False)[0]))
    est = np.asarray(est)
    return {
        "n_atoms": n_atoms, "dim": d, "n_repeat": n_repeat,
        "trace_analytic": tr_analytic,
        "trace_exact_autograd": tr_exact,
        "exact_abs_error": abs(tr_exact - tr_analytic),
        "hutch2_mean": float(est.mean()),
        "hutch2_bias": float(est.mean() - tr_analytic),
        "hutch2_bias_sem": float(est.std(ddof=1) / math.sqrt(n_repeat)),
        "hutch2_sd_measured": float(est.std(ddof=1)),
        "hutch2_sd_analytic": float(math.sqrt(var_analytic / 2.0)),
    }


# ---------------------------------------------------------------------------
# 1.  Model / data plumbing (mirrors scripts/eval_boltzmann_stage1.py)
# ---------------------------------------------------------------------------
def load_model_and_val(config: Path, checkpoint: Path, eval_data: Path, device: str):
    from flowmol.model_utils.load import model_from_config, read_config_file
    from flowmol.data_processing.dataset import MoleculeDataset

    cfg = read_config_file(config)
    cfg.get("mol_fm", {}).pop("bgfm", None)
    cfg["dataset"]["processed_data_dir"] = str(eval_data)
    atom_map = cfg["dataset"]["atom_map"]

    model = model_from_config(cfg)
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
    patch_flowmol(model, d_min, tangent=True, retract=True, gluing=True,
                  discrete_projection=not bond_free,
                  train_time_discrete=not bond_free, atom_map=atom_map)

    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    sd = state.get("state_dict", state)
    bad = sum(int((~torch.isfinite(v)).sum()) for v in sd.values()
              if torch.is_tensor(v) and v.is_floating_point())
    if bad:
        raise RuntimeError(f"checkpoint has {bad} non-finite params -- refusing to score")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[p1] ckpt loaded: missing={len(missing)} unexpected={len(unexpected)} "
          f"non_finite_params=0", flush=True)
    model = model.to(device).eval()

    ds_cfg = dict(cfg["dataset"])
    ds_cfg["fake_atom_p"] = 0.0
    ds_cfg["fake_atom_std"] = 1.0
    ds_cfg["explicit_aromaticity"] = cfg["mol_fm"].get("explicit_aromaticity", False)
    val = MoleculeDataset("val", ds_cfg, prior_config=cfg["mol_fm"]["prior_config"])
    return model, val, cfg, atom_map


def _sig(P: torch.Tensor):
    """Geometry signature -- verbatim from paper/figures/recompute_disjoint_noreference.py."""
    P = P - P.mean(0, keepdim=True)
    d = torch.sort(P.pow(2).sum(1).sqrt()).values
    return (P.shape[0], tuple((d * 1e4).round().long().tolist()))


def build_group_to_val(samples_json: Path, eval_data: Path) -> dict:
    val = torch.load(f"{eval_data}/val_data_processed.pt", map_location="cpu",
                     weights_only=False)
    pos = val["positions"].double()
    table = {}
    for i, (a, b) in enumerate(val["node_idx_array"].tolist()):
        table.setdefault(_sig(pos[a:b]), []).append(i)
    out = {}
    for r in json.load(open(samples_json)):
        if r["pert_id"]:
            continue
        hit = table.get(_sig(torch.tensor(r["positions"], dtype=torch.float64)))
        if hit and len(hit) == 1:
            out[int(r["group_id"])] = hit[0]
    return out


def make_group_batch(val, di: int, positions: list, device: str):
    """(1 + n_perturb) copies of val molecule di, positions overridden verbatim."""
    import dgl
    from flowmol.data_processing.utils import get_batch_idxs, get_upper_edge_mask
    g0 = val[di].to(device)
    graphs = []
    for pos in positions:
        gp = g0.clone()
        p = torch.tensor(pos, dtype=gp.ndata['x_1_true'].dtype, device=device)
        assert p.shape == gp.ndata['x_1_true'].shape, (p.shape, gp.ndata['x_1_true'].shape)
        gp.ndata['x_1_true'] = p
        graphs.append(gp)
    gb = dgl.batch(graphs).to(device)
    gb.ndata['x_t'] = gb.ndata['x_1_true']
    gb.ndata['a_t'] = gb.ndata['a_1_true']
    gb.ndata['c_t'] = gb.ndata['c_1_true']
    gb.edata['e_t'] = gb.edata['e_1_true']
    nbi, _ = get_batch_idxs(gb)
    uem = get_upper_edge_mask(gb)
    return gb, nbi, uem


# ---------------------------------------------------------------------------
# 2.  Instrumented reverse trajectory
# ---------------------------------------------------------------------------
def trajectory_divergences(model, gb, nbi, uem, n_ode_steps: int, n_repeat: int,
                           n_hutch: int, exact: bool, seed: int):
    """Walk the SAME reverse trajectory as log_density_via_flow, recording at each
    node the exact divergence (optional) and ``n_repeat`` independent
    n_hutch-probe Hutchinson estimates.

    Returns dict with div_exact (n_steps, B), div_hutch (n_repeat, n_steps, B),
    logp0 (B,), logq_exact (B,), logq_hutch (n_repeat, B).
    """
    from cfm_mol.bgfm_density import (make_position_velocity_fn,
                                      gaussian_prior_log_density,
                                      _n_atoms_per_graph)
    from cfm_mol.bgfm_loss import divergence_hutchinson, divergence_exact_atomwise

    device = gb.device
    B = gb.batch_size
    x = gb.ndata['x_1_true'].detach().clone()
    dt = 1.0 / n_ode_steps
    n_apg = _n_atoms_per_graph(gb, nbi)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    def probe(_k, xx):
        return (torch.randint(0, 2, xx.shape, device=device, generator=gen,
                              dtype=xx.dtype) * 2 - 1)

    div_exact = np.zeros((n_ode_steps, B)) if exact else None
    div_h = np.zeros((n_repeat, n_ode_steps, B))

    for step in range(n_ode_steps):
        t_val = 1.0 - (step + 0.5) * dt
        t_scalar = torch.full((B,), t_val, device=device, dtype=torch.float32)

        if exact:
            xr = x.detach().requires_grad_(True)
            v_fn = make_position_velocity_fn(model, gb, t_scalar, nbi, uem)
            with torch.enable_grad():
                d = divergence_exact_atomwise(v_fn, xr, n_apg, create_graph=False)
            div_exact[step] = d.detach().cpu().numpy()
            del d, xr, v_fn

        for r in range(n_repeat):
            xr = x.detach().requires_grad_(True)
            v_fn = make_position_velocity_fn(model, gb, t_scalar, nbi, uem)
            with torch.enable_grad():
                d = divergence_hutchinson(v_fn, xr, n_apg, n_samples=n_hutch,
                                          rademacher=True, create_graph=False,
                                          xi_provider=probe)
            div_h[r, step] = d.detach().cpu().numpy()
            del d, xr, v_fn

        with torch.no_grad():
            gb.ndata['x_t'] = x
            v = model.vector_field(gb, t_scalar, node_batch_idx=nbi,
                                   upper_edge_mask=uem)['x']
            x = x - dt * v

    logp0 = gaussian_prior_log_density(x, n_apg, prior_std=1.0).detach().cpu().numpy()
    out = {"logp0": logp0, "div_hutch": div_h,
           "logq_hutch": logp0[None, :] - dt * div_h.sum(axis=1),
           "n_atoms": n_apg.cpu().numpy()}
    if exact:
        out["div_exact"] = div_exact
        out["logq_exact"] = logp0 - dt * div_exact.sum(axis=0)
    return out


# ---------------------------------------------------------------------------
# 3.  Statistics helpers
# ---------------------------------------------------------------------------
def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)      # drop records whose xTB energy failed
    x, y = x[m], y[m]
    if x.size < 3 or x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def describe(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {}
    return {"n": int(a.size), "mean": float(a.mean()),
            "median": float(np.median(a)), "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "min": float(a.min()), "max": float(a.max()),
            "p25": float(np.percentile(a, 25)), "p75": float(np.percentile(a, 75))}


# ---------------------------------------------------------------------------
# 4.  main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all", choices=["all", "toy", "A", "B"])
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--eval_data", type=Path)
    ap.add_argument("--samples_json", type=Path)
    ap.add_argument("--records_csv", type=Path)
    ap.add_argument("--disjoint_json", type=Path)
    ap.add_argument("--n_ode_steps", type=int, default=12)
    ap.add_argument("--n_hutchinson", type=int, default=2)
    ap.add_argument("--taskA_max_atoms", type=int, default=22)
    ap.add_argument("--taskA_n_parents", type=int, default=4)
    ap.add_argument("--taskA_n_geoms", type=int, default=3)
    ap.add_argument("--taskA_repeat", type=int, default=12)
    ap.add_argument("--taskB_n_parents", type=int, default=24)
    ap.add_argument("--taskB_repeat", type=int, default=12)
    ap.add_argument("--taskB_max_atoms", type=int, default=0,
                    help="0 = no cap (use the reported population as-is)")
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--device", default=None)
    ap.add_argument("--verify_production", action="store_true",
                    help="also call log_density_via_flow directly on one parent")
    ap.add_argument("--out_json", type=Path, required=True)
    args = ap.parse_args()

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    report = {"argv": sys.argv, "settings": {
        "n_ode_steps": args.n_ode_steps, "n_hutchinson": args.n_hutchinson,
        "checkpoint": str(args.checkpoint), "config": str(args.config),
        "samples_json": str(args.samples_json), "seed": args.seed}}

    if args.task in ("all", "toy"):
        t0 = time.time()
        report["toy"] = task_toy()
        print(f"[p1] toy control done in {time.time()-t0:.1f}s: "
              f"{json.dumps(report['toy'], indent=1)}", flush=True)
        json.dump(report, open(args.out_json, "w"), indent=1)

    if args.task == "toy":
        return 0

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[p1] device={device}", flush=True)
    model, val, cfg, atom_map = load_model_and_val(
        args.config, args.checkpoint, args.eval_data, device)

    # frozen geometries + energies from the reported population
    recs = json.load(open(args.samples_json))
    by_group = {}
    for r in recs:
        by_group.setdefault(int(r["group_id"]), {})[int(r["pert_id"])] = r
    gid2val = build_group_to_val(args.samples_json, args.eval_data)
    disjoint = set(json.load(open(args.disjoint_json))["disjoint_group_ids"])
    negE = {}
    if args.records_csv and args.records_csv.exists():
        for row in csv.DictReader(open(args.records_csv)):
            negE[(int(row["group_id"]), int(row["pert_id"]))] = float(row["negE_kT"])
    usable = sorted(g for g in disjoint if g in gid2val)
    sizes = {g: len(by_group[g][0]["atomic_numbers"]) for g in usable}
    print(f"[p1] disjoint parents usable: {len(usable)} "
          f"(sizes {min(sizes.values())}-{max(sizes.values())})", flush=True)

    # ---------------- TASK A ----------------
    if args.task in ("all", "A"):
        small = [g for g in usable if sizes[g] <= args.taskA_max_atoms]
        small.sort(key=lambda g: (sizes[g], g))
        # spread over the small-molecule size range rather than taking the very smallest
        pick = [small[int(round(i * (len(small) - 1) / max(1, args.taskA_n_parents - 1)))]
                for i in range(args.taskA_n_parents)] if small else []
        pick = sorted(set(pick))
        print(f"[p1][A] parents {pick} sizes {[sizes[g] for g in pick]}", flush=True)
        perstep, perlogq, groups_out = [], [], []
        for gi, g in enumerate(pick):
            n_av = max(by_group[g]) + 1
            pert_ids = list(range(n_av if args.taskA_n_geoms <= 0
                                  else min(args.taskA_n_geoms, n_av)))
            positions = [by_group[g][p]["positions"] for p in pert_ids]
            gb, nbi, uem = make_group_batch(val, gid2val[g], positions, device)
            t0 = time.time()
            res = trajectory_divergences(model, gb, nbi, uem, args.n_ode_steps,
                                         n_repeat=args.taskA_repeat,
                                         n_hutch=args.n_hutchinson, exact=True,
                                         seed=args.seed + 1000 * gi)
            D = res["div_exact"]                     # (S, B)
            H = res["div_hutch"]                     # (R, S, B)
            bias = H.mean(axis=0) - D
            sd = H.std(axis=0, ddof=1)
            for s in range(D.shape[0]):
                for b in range(D.shape[1]):
                    perstep.append({
                        "group": g, "pert": pert_ids[b], "step": s,
                        "t": 1.0 - (s + 0.5) / args.n_ode_steps,
                        "exact": float(D[s, b]), "hutch_mean": float(H[:, s, b].mean()),
                        "bias": float(bias[s, b]), "sd": float(sd[s, b]),
                        "rel_bias": float(bias[s, b] / abs(D[s, b])) if abs(D[s, b]) > 1e-9 else float("nan"),
                        "rel_sd": float(sd[s, b] / abs(D[s, b])) if abs(D[s, b]) > 1e-9 else float("nan"),
                    })
            LE = res["logq_exact"]; LH = res["logq_hutch"]
            # ordering metric with an EXACT-divergence reference, reference
            # geometry dropped (primary-endpoint convention)
            keepA = [b for b, p_ in enumerate(pert_ids) if p_ != 0]
            r_block = {}
            if len(keepA) >= 3 and negE:
                eA = np.array([negE.get((g, pert_ids[b]), float("nan")) for b in keepA])
                r_ex = pearson(LE[keepA], eA)
                r_hu = [pearson(LH[rr, keepA], eA) for rr in range(args.taskA_repeat)]
                r_block = {"r_exact_divergence": r_ex,
                           "r_hutch2_mean": float(np.nanmean(r_hu)),
                           "r_hutch2_sd": float(np.nanstd(r_hu, ddof=1)),
                           "r_hutch2_minus_exact": float(np.nanmean(r_hu) - r_ex)}
            for b in range(LE.shape[0]):
                perlogq.append({
                    "group": g, "pert": pert_ids[b], "n_atoms": int(res["n_atoms"][b]),
                    "logq_exact": float(LE[b]),
                    "logq_hutch_mean": float(LH[:, b].mean()),
                    "logq_bias": float(LH[:, b].mean() - LE[b]),
                    "logq_sd": float(LH[:, b].std(ddof=1)),
                    "logq_rel_sd": float(LH[:, b].std(ddof=1) / abs(LE[b])),
                })
            groups_out.append({"group": g, "n_atoms": sizes[g],
                               "n_geoms": len(pert_ids),
                               "seconds": round(time.time() - t0, 1), **r_block})
            print(f"[p1][A] group {g} (N={sizes[g]}) done in "
                  f"{time.time()-t0:.1f}s", flush=True)
            del gb
            if device == "cuda":
                torch.cuda.empty_cache()
        report["taskA"] = {
            "n_repeat": args.taskA_repeat, "parents": groups_out,
            "per_step": perstep, "per_logq": perlogq,
            "summary": {
                "div_rel_sd": describe([d["rel_sd"] for d in perstep]),
                "div_rel_bias": describe([d["rel_bias"] for d in perstep]),
                "div_bias_over_sd": describe(
                    [d["bias"] / d["sd"] for d in perstep if d["sd"] > 0]),
                "logq_sd_nats": describe([d["logq_sd"] for d in perlogq]),
                "logq_bias_nats": describe([d["logq_bias"] for d in perlogq]),
                "logq_bias_over_sd": describe(
                    [d["logq_bias"] / d["logq_sd"] for d in perlogq if d["logq_sd"] > 0]),
                "logq_rel_sd": describe([d["logq_rel_sd"] for d in perlogq]),
                "r_exact_divergence": describe(
                    [d["r_exact_divergence"] for d in groups_out if "r_exact_divergence" in d]),
                "r_hutch2_mean": describe(
                    [d["r_hutch2_mean"] for d in groups_out if "r_hutch2_mean" in d]),
                "r_hutch2_minus_exact": describe(
                    [d["r_hutch2_minus_exact"] for d in groups_out if "r_hutch2_minus_exact" in d]),
                "r_hutch2_sd": describe(
                    [d["r_hutch2_sd"] for d in groups_out if "r_hutch2_sd" in d]),
            }}
        json.dump(report, open(args.out_json, "w"), indent=1)

    # ---------------- TASK B ----------------
    if args.task in ("all", "B"):
        pool = [g for g in usable
                if args.taskB_max_atoms <= 0 or sizes[g] <= args.taskB_max_atoms]
        pool.sort(key=lambda g: (sizes[g], g))
        k = min(args.taskB_n_parents, len(pool))
        pick = [pool[int(round(i * (len(pool) - 1) / max(1, k - 1)))] for i in range(k)]
        pick = sorted(set(pick))
        print(f"[p1][B] {len(pick)} parents, sizes {[sizes[g] for g in pick]}", flush=True)
        per_record, per_group, skipped = [], [], []
        R = args.taskB_repeat
        for gi, g in enumerate(pick):
            n_pert = max(by_group[g]) + 1
            pert_ids = list(range(n_pert))
            positions = [by_group[g][p]["positions"] for p in pert_ids]
            t0 = time.time()
            try:
                gb, nbi, uem = make_group_batch(val, gid2val[g], positions, device)
                res = trajectory_divergences(model, gb, nbi, uem, args.n_ode_steps,
                                             n_repeat=R, n_hutch=args.n_hutchinson,
                                             exact=False, seed=args.seed + 7919 * (gi + 1))
            except torch.cuda.OutOfMemoryError as exc:
                # Skipped for MEMORY only -- recorded explicitly so the reported
                # coverage is auditable. Never silently dropped.
                skipped.append({"group": g, "n_atoms": sizes[g], "reason": "cuda_oom"})
                print(f"[p1][B] group {g} N={sizes[g]} SKIPPED (CUDA OOM)", flush=True)
                gb = None
                if device == "cuda":
                    torch.cuda.empty_cache()
                continue
            LH = res["logq_hutch"]                    # (R, B)
            sd_rec = LH.std(axis=0, ddof=1)
            mean_rec = LH.mean(axis=0)
            for b, p in enumerate(pert_ids):
                per_record.append({
                    "group": g, "pert": p, "n_atoms": int(res["n_atoms"][b]),
                    "logq_repeat_mean": float(mean_rec[b]),
                    "logq_repeat_sd": float(sd_rec[b]),
                    "logq_reported": float(by_group[g][p]["log_p_theta"]),
                    "negE_kT": negE.get((g, p), float("nan")),
                })
            # within-parent statistics, reference (pert_id 0) DROPPED -- the
            # primary endpoint's convention.
            keep = [b for b, p in enumerate(pert_ids) if p != 0]
            sig_sd_raw = float(mean_rec[keep].std(ddof=1))
            noise_sd = float(np.sqrt((sd_rec[keep] ** 2).mean()))
            # mean-of-R estimate carries noise^2/R; remove it to get the signal SD
            sig_var = sig_sd_raw ** 2 - noise_sd ** 2 / R
            sig_sd = float(math.sqrt(sig_var)) if sig_var > 0 else float("nan")
            e = np.array([negE.get((g, pert_ids[b]), float("nan")) for b in keep])
            r_rep = [pearson(LH[rr, keep], e) for rr in range(R)]
            r_mean_logq = pearson(mean_rec[keep], e)
            r_reported = pearson(
                [by_group[g][pert_ids[b]]["log_p_theta"] for b in keep], e)
            per_group.append({
                "group": g, "n_atoms": sizes[g], "n_kept": len(keep),
                "signal_sd_raw": sig_sd_raw, "signal_sd_corrected": sig_sd,
                "noise_sd": noise_sd,
                "noise_over_signal": (noise_sd / sig_sd) if sig_sd == sig_sd and sig_sd > 0 else float("nan"),
                "r_per_repeat_mean": float(np.nanmean(r_rep)),
                "r_per_repeat_sd": float(np.nanstd(r_rep, ddof=1)),
                "r_per_repeat": [float(v) for v in r_rep],
                "r_repeat_mean_logq": r_mean_logq,
                "r_reported_single_draw": r_reported,
                "seconds": round(time.time() - t0, 1),
            })
            print(f"[p1][B] group {g} N={sizes[g]} noise_sd={noise_sd:.3f} "
                  f"signal_sd={sig_sd:.3f} r_rep={np.nanmean(r_rep):+.3f}"
                  f"+-{np.nanstd(r_rep, ddof=1):.3f} ({time.time()-t0:.1f}s)", flush=True)
            if args.verify_production and gi == 0:
                from cfm_mol.bgfm_density import log_density_via_flow
                prod = []
                for s in range(4):
                    torch.manual_seed(90000 + s)
                    gb2, nbi2, uem2 = make_group_batch(val, gid2val[g], positions, device)
                    with torch.enable_grad():
                        lp = log_density_via_flow(model, gb2, nbi2, uem2,
                                                  n_ode_steps=args.n_ode_steps,
                                                  n_hutchinson=args.n_hutchinson,
                                                  prior_std=1.0)
                    prod.append(lp.detach().cpu().numpy().tolist())
                    del gb2
                prod = np.array(prod)
                report["production_check"] = {
                    "group": g,
                    "log_density_via_flow_sd": prod.std(axis=0, ddof=1).tolist(),
                    "log_density_via_flow_mean": prod.mean(axis=0).tolist(),
                    "instrumented_sd": sd_rec.tolist(),
                    "instrumented_mean": mean_rec.tolist(),
                }
                print(f"[p1][B] production check: "
                      f"{json.dumps(report['production_check'])}", flush=True)
            del gb
            if device == "cuda":
                torch.cuda.empty_cache()

        ns = [d["noise_over_signal"] for d in per_group]
        atten = [1.0 / math.sqrt(1.0 + v ** 2) for v in ns
                 if v == v and math.isfinite(v)]
        report["taskB"] = {
            "n_repeat": R, "per_record": per_record, "per_group": per_group,
            "n_parents_requested": len(pick), "n_parents_scored": len(per_group),
            "skipped_parents": skipped,
            "summary": {
                "logq_repeat_sd_nats": describe([d["logq_repeat_sd"] for d in per_record]),
                "logq_repeat_sd_relative": describe(
                    [abs(d["logq_repeat_sd"] / d["logq_repeat_mean"]) for d in per_record]),
                "within_parent_signal_sd": describe([d["signal_sd_corrected"] for d in per_group]),
                "within_parent_noise_sd": describe([d["noise_sd"] for d in per_group]),
                "noise_over_signal": describe(ns),
                "implied_r_attenuation_factor": describe(atten),
                "r_sd_across_repeats": describe([d["r_per_repeat_sd"] for d in per_group]),
                "mean_r_single_draw": describe([d["r_per_repeat_mean"] for d in per_group]),
                "mean_r_repeat_mean_logq": describe([d["r_repeat_mean_logq"] for d in per_group]),
                "mean_r_reported": describe([d["r_reported_single_draw"] for d in per_group]),
            }}
        # population-level: mean per-parent r for each independent probe draw.
        # Its SD is the estimator-induced uncertainty on an ARM's headline
        # number at this population size, to be read against the between-seed SEM.
        RM = np.array([[g["r_per_repeat"][rr] for g in per_group] for rr in range(R)])
        pop = np.nanmean(RM, axis=1)
        report["taskB"]["summary"]["population_mean_r_per_repeat"] = {
            "values": [float(v) for v in pop],
            "mean": float(pop.mean()), "sd": float(pop.std(ddof=1)),
            "n_parents": int(RM.shape[1]), "n_repeats": int(R)}
        json.dump(report, open(args.out_json, "w"), indent=1)

    json.dump(report, open(args.out_json, "w"), indent=1)
    print(f"[p1] wrote {args.out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
