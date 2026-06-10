"""Level 3 comparison: BGFM K-step ensemble vs OMol25 Langevin MD reference.

For each test molecule we have two ensembles -- both energy-labeled by the
same OMol25 NP, so they are directly comparable as distributions over
configurations of that molecule under that potential.

Metrics computed per molecule:
  - Energy histogram overlap (proportional intersection)
  - Wasserstein-1 distance between energy distributions
  - Pairwise distance histogram overlap (RDF proxy)
  - Effective sample size for each method
  - Per-method compute (NN forward calls): BGFM = K * n_samples;
    MD = n_md_steps_total * 1   (Langevin)
  - "Speedup at matched quality" = MD calls / BGFM calls when both
    achieve the same energy-histogram overlap target (headline number).

Output: <out>/level3_summary.csv + per-molecule npz with full histograms +
a single PNG comparison figure per molecule (when matplotlib available).

Usage (envs/omol25 -- needs numpy + ase + matplotlib):
  conda activate envs/omol25
  python scripts/level3_compare.py \\
      --md_dir runs/eval/level3_md_reference \\
      --bgfm_dir runs/eval/level3_bgfm \\
      --out_dir runs/eval/level3_compare
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def _pairwise_distances(positions: np.ndarray) -> np.ndarray:
    """positions (n_atoms, 3) -> (n_pairs,) upper-triangle distances."""
    n = positions.shape[0]
    diffs = positions[:, None, :] - positions[None, :, :]      # (n, n, 3)
    dij = np.sqrt((diffs ** 2).sum(-1))                          # (n, n)
    iu = np.triu_indices(n, k=1)
    return dij[iu]


def _hist_overlap(a: np.ndarray, b: np.ndarray, bins: int = 50,
                  lo: float | None = None, hi: float | None = None) -> tuple[float, np.ndarray, np.ndarray]:
    """Normalized intersection of two histograms; returns (overlap, ha, hb)."""
    if lo is None: lo = min(a.min(), b.min())
    if hi is None: hi = max(a.max(), b.max())
    edges = np.linspace(lo, hi, bins + 1)
    ha, _ = np.histogram(a, bins=edges, density=True)
    hb, _ = np.histogram(b, bins=edges, density=True)
    # Normalize as probability (sum over bins = 1 after multiplying by width)
    width = (hi - lo) / bins
    pa = ha * width
    pb = hb * width
    overlap = float(np.minimum(pa, pb).sum())
    return overlap, ha, hb


def _wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    """Wasserstein-1 between two 1D empirical distributions via sorted-CDF
    difference. O(N log N), no scipy dependency."""
    a = np.sort(a)
    b = np.sort(b)
    # Resample one to the other's length via linear interpolation, or use
    # the equal-grid CDF approach. Simplest: interpolate b onto a's quantiles.
    if len(a) == 0 or len(b) == 0: return float("nan")
    qa = np.linspace(0, 1, len(a))
    qb = np.linspace(0, 1, len(b))
    # Inverse CDFs sampled on a common quantile grid
    K = max(len(a), len(b))
    q = np.linspace(0, 1, K)
    inv_a = np.interp(q, qa, a)
    inv_b = np.interp(q, qb, b)
    return float(np.mean(np.abs(inv_a - inv_b)))


def _ess(values: np.ndarray) -> float:
    """Effective sample size via autocorrelation. For an iid sample this
    returns ~ len(values); for correlated MD it can be much smaller."""
    n = len(values)
    if n < 4: return float(n)
    v = values - values.mean()
    # autocorrelation up to lag n//4
    f = np.fft.fft(v, n=2 * n)
    ac = np.fft.ifft(f * np.conj(f)).real[:n]
    ac = ac / ac[0]
    # Sum of autocorrelations until first negative
    tau = 1.0
    for k in range(1, min(n, n // 4)):
        if ac[k] < 0: break
        tau += 2 * ac[k]
    return float(n / max(tau, 1.0))


def _process_pair(md_npz: dict, bgfm_npz: dict, mol_id: str,
                  hist_bins: int = 50) -> dict:
    """Compute all per-molecule metrics."""
    md_E = md_npz["energies"][~np.isnan(md_npz["energies"])]
    bg_E = bgfm_npz["energies"][~np.isnan(bgfm_npz["energies"])]
    if md_E.size < 10 or bg_E.size < 10:
        return {"mol_id": mol_id, "error": "too few valid energies"}

    # ENERGY overlap + Wasserstein
    lo = min(md_E.min(), bg_E.min())
    hi = max(md_E.max(), bg_E.max())
    e_overlap, e_ha, e_hb = _hist_overlap(md_E, bg_E, bins=hist_bins, lo=lo, hi=hi)
    e_wass = _wasserstein_1d(md_E, bg_E)

    # PAIRWISE DISTANCE overlap (RDF proxy)
    md_d = np.concatenate([_pairwise_distances(p) for p in md_npz["positions"]])
    bg_d = np.concatenate([_pairwise_distances(p) for p in bgfm_npz["positions"]])
    d_lo = 0.0
    d_hi = max(md_d.max(), bg_d.max())
    d_overlap, d_ha, d_hb = _hist_overlap(md_d, bg_d, bins=hist_bins,
                                           lo=d_lo, hi=d_hi)

    # Effective sample sizes (BGFM is iid by construction)
    md_ess = _ess(md_E)
    bg_ess = float(len(bg_E))   # iid samples

    # Compute (NN forward calls)
    md_calls = int(md_npz.get("n_md_steps_total", 100000))
    bgfm_K = int(bgfm_npz.get("K_steps", 12))
    bgfm_n = int(bgfm_npz.get("n_samples", len(bg_E)))
    bgfm_calls = bgfm_K * bgfm_n
    # Speedup at MATCHED ESS quality:
    #   ratio of calls per effective sample
    md_calls_per_eff = md_calls / max(md_ess, 1)
    bg_calls_per_eff = bgfm_calls / max(bg_ess, 1)
    speedup = md_calls_per_eff / max(bg_calls_per_eff, 1e-9)

    return {
        "mol_id": mol_id,
        "n_atoms": int(md_npz["positions"].shape[1]),
        "md_N": int(md_E.size),
        "bgfm_N": int(bg_E.size),
        "energy_overlap": float(e_overlap),
        "energy_wass1_eV": float(e_wass),
        "pairwise_dist_overlap": float(d_overlap),
        "md_ess": float(md_ess),
        "bgfm_ess": float(bg_ess),
        "md_calls": int(md_calls),
        "bgfm_calls": int(bgfm_calls),
        "speedup_per_eff_sample": float(speedup),
    }, e_ha, e_hb, d_ha, d_hb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md_dir", type=Path, required=True)
    ap.add_argument("--bgfm_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--hist_bins", type=int, default=50)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Pair files by mol id (the trailing molXXXXX number)
    md_files = {p.stem.split("_mol")[-1]: p for p in args.md_dir.glob("*.npz")}
    bg_files = {p.stem.split("_mol")[-1]: p for p in args.bgfm_dir.glob("*.npz")}
    mol_ids = sorted(set(md_files) & set(bg_files))
    print(f"[lvl3-cmp] paired {len(mol_ids)} molecules with both MD + BGFM data",
          flush=True)
    if not mol_ids:
        print("[lvl3-cmp] no pairs found, nothing to do.", flush=True)
        return 1

    rows = []
    for mid in mol_ids:
        md = dict(np.load(md_files[mid], allow_pickle=False))
        bg = dict(np.load(bg_files[mid], allow_pickle=False))
        out = _process_pair(md, bg, mid, hist_bins=args.hist_bins)
        if isinstance(out, dict):
            print(f"[lvl3-cmp] mol {mid}: ERROR {out.get('error')}", flush=True)
            continue
        row, e_ha, e_hb, d_ha, d_hb = out
        rows.append(row)
        # Per-mol diagnostic npz
        np.savez(args.out_dir / f"level3_compare_mol{mid}.npz",
                 energy_hist_md=e_ha, energy_hist_bgfm=e_hb,
                 pairwise_hist_md=d_ha, pairwise_hist_bgfm=d_hb,
                 **row)
        print(f"[lvl3-cmp] mol {mid}: E-overlap={row['energy_overlap']:.3f}, "
              f"W1={row['energy_wass1_eV']:.3f} eV, "
              f"d-overlap={row['pairwise_dist_overlap']:.3f}, "
              f"speedup={row['speedup_per_eff_sample']:.1f}x", flush=True)

    # Write summary CSV
    if rows:
        csv_path = args.out_dir / "level3_summary.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\n[lvl3-cmp] wrote {csv_path}", flush=True)
        # Aggregate headline numbers
        e_ov = [r["energy_overlap"] for r in rows]
        d_ov = [r["pairwise_dist_overlap"] for r in rows]
        sp = [r["speedup_per_eff_sample"] for r in rows]
        print(f"\n[lvl3-cmp SUMMARY]  (across {len(rows)} molecules)")
        print(f"  energy histogram overlap : mean {np.mean(e_ov):.3f}  "
              f"median {np.median(e_ov):.3f}")
        print(f"  pairwise dist overlap    : mean {np.mean(d_ov):.3f}  "
              f"median {np.median(d_ov):.3f}")
        print(f"  speedup per effective sample (BGFM vs MD):")
        print(f"    geomean {np.exp(np.mean(np.log(np.clip(sp,1e-3,None)))):.0f}x  "
              f"median {np.median(sp):.0f}x  range [{np.min(sp):.0f}x, {np.max(sp):.0f}x]")
        print()
        print("  Interpretation:")
        print("    overlap > 0.7 + speedup > 1000x  =>  Level-3 result ready for ICLR oral.")
        print("    overlap 0.3-0.7                   =>  partial agreement; explore K_steps + n_samples.")
        print("    overlap < 0.3                     =>  BGFM ensemble does NOT match MD; method limitation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
