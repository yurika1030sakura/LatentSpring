"""E2: measure convergence rate vs step-size Δt.

Compare two generators on the SAME task (generate coord conformers for
QM9 atom compositions):
  - Reflected SDE (Euler-Maruyama): expected rate $O(\\sqrt{\\Delta t})$
  - Probability-flow ODE (same score net, deterministic): expected
    $O(\\Delta t)$
  - (optionally also our patched FlowMol3's coordinate channel if ckpt
    provided)

Metric: 1-Wasserstein distance between sampled and reference pairwise-
distance histograms. Computed per atom-type pair (C-C, C-H, ...) and
averaged — this is a standard proxy for TV in coord-only benchmarks.

Output: CSV with columns [nfe, dt, W1_sde, W1_pfode, W1_ref] that
produces a log-log plot showing the slope difference.

Usage:
    PYTHONPATH=. python scripts/e2_convergence_rate.py \\
        --reflected_ckpt runs/reflected_sde/best.pt \\
        --processed_dir data/qm9_processed \\
        --out runs/e2_convergence.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

from cfm_mol.baselines.reflected_sde import CoordScoreNet, ReflectedSDEModel
from cfm_mol.domain import default_d_min_table


def _pair_distance_histogram(
    r: torch.Tensor,                                  # (B, N, 3) or (N, 3)
    a: torch.Tensor,                                  # (B, N) or (N,)
    n_atom_types: int,
    bins: int = 50,
    range_max: float = 6.0,
    lengths: torch.Tensor | None = None,
) -> dict[tuple[int, int], np.ndarray]:
    """Histograms of pair distances split by atom-type pair. Returns a
    dict mapping (min(a_i, a_j), max(a_i, a_j)) -> normalized histogram."""
    if r.dim() == 2:
        r = r.unsqueeze(0)
        a = a.unsqueeze(0)
        if lengths is not None:
            lengths = lengths.unsqueeze(0)
    B, N, _ = r.shape
    out: dict[tuple[int, int], list] = {}
    for b in range(B):
        n = int(lengths[b]) if lengths is not None else N
        rb = r[b, :n]
        ab = a[b, :n]
        diffs = rb.unsqueeze(0) - rb.unsqueeze(1)
        dists = diffs.norm(dim=-1)
        for i in range(n):
            for j in range(i + 1, n):
                key = (int(min(ab[i], ab[j])), int(max(ab[i], ab[j])))
                out.setdefault(key, []).append(float(dists[i, j]))
    hists = {}
    for key, vals in out.items():
        h, _ = np.histogram(vals, bins=bins, range=(0.0, range_max))
        s = h.sum()
        hists[key] = h / s if s > 0 else h.astype(float)
    return hists


def _wasserstein_1d(h1: np.ndarray, h2: np.ndarray) -> float:
    """1-Wasserstein between two normalized 1D histograms (equal bin-width)."""
    c1 = np.cumsum(h1)
    c2 = np.cumsum(h2)
    return float(np.abs(c1 - c2).sum())


def _compare(h_a: dict, h_b: dict) -> float:
    keys = set(h_a) | set(h_b)
    if not keys:
        return float("nan")
    total = 0.0
    cnt = 0
    for k in keys:
        if k in h_a and k in h_b:
            total += _wasserstein_1d(h_a[k], h_b[k])
            cnt += 1
    return total / max(cnt, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reflected_ckpt", type=Path, required=True)
    ap.add_argument("--processed_dir", type=Path, required=True)
    ap.add_argument("--n_atom_types", type=int, default=5)
    ap.add_argument("--n_max", type=int, default=30)
    ap.add_argument("--n_samples", type=int, default=512)
    ap.add_argument("--nfe_list", default="10,20,50,100,200")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    # Reference = real QM9 val-set pair-distance histograms.
    from scripts.train_reflected_sde import QM9CoordDataset
    ds = QM9CoordDataset(args.processed_dir, args.n_atom_types,
                          args.n_max, split="val")
    print(f"reference set: {len(ds)} mols")
    pos = ds.positions[:args.n_samples]
    atm = ds.atom_types[:args.n_samples]
    lns = ds.lengths[:args.n_samples]
    h_ref = _pair_distance_histogram(pos, atm, args.n_atom_types,
                                      lengths=lns)

    # Load trained score net.
    d_min = default_d_min_table(args.n_atom_types)
    score = CoordScoreNet(n_atom_types=args.n_atom_types)
    model = ReflectedSDEModel(score, d_min).to(args.device)
    state = torch.load(args.reflected_ckpt, map_location=args.device)
    model.load_state_dict(state, strict=False)
    model.eval()

    # Sample atom-type templates from the reference so conditioning matches.
    a_sample = atm.to(args.device)
    lens_sample = lns.to(args.device)

    nfe_values = [int(x) for x in args.nfe_list.split(",")]
    rows = []
    for nfe in nfe_values:
        print(f"NFE={nfe}")
        with torch.no_grad():
            r_sde = model.sample_euler_maruyama(a_sample, n_steps=nfe)
            r_ode = model.sample_probability_flow_ode(a_sample, n_steps=nfe)

        h_sde = _pair_distance_histogram(r_sde.cpu(), atm,
                                          args.n_atom_types, lengths=lns)
        h_ode = _pair_distance_histogram(r_ode.cpu(), atm,
                                          args.n_atom_types, lengths=lns)
        w1_sde = _compare(h_sde, h_ref)
        w1_ode = _compare(h_ode, h_ref)
        dt = 1.0 / nfe
        rows.append(dict(nfe=nfe, dt=dt,
                         W1_sde=w1_sde, W1_pfode=w1_ode))
        print(f"  dt={dt:.4f}  W1_sde={w1_sde:.4f}  W1_pfode={w1_ode:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}")

    # Quick slope estimate: log(W1) vs log(dt). Slope should be ~1 for PF-ODE
    # and ~0.5 for SDE.
    dt_arr = np.array([r["dt"] for r in rows])
    sde_arr = np.array([r["W1_sde"] for r in rows])
    ode_arr = np.array([r["W1_pfode"] for r in rows])
    slope_sde = float(np.polyfit(np.log(dt_arr), np.log(sde_arr), 1)[0])
    slope_ode = float(np.polyfit(np.log(dt_arr), np.log(ode_arr), 1)[0])
    print(f"\nEMPIRICAL SLOPES (log-log fit):")
    print(f"  SDE (Euler-Maruyama)       : {slope_sde:.3f}  (theory: 0.5)")
    print(f"  PF-ODE (same score, Euler) : {slope_ode:.3f}  (theory: 1.0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
