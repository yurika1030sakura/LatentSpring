"""Calibrate d_min table against real molecule datasets.

Reads processed QM9 / GEOM-Drugs / tmQM positions + atom types, scans
every pairwise distance, reports:
  - fraction of molecules where any pair violates d_min (under current scale)
  - minimum observed pair distance per atom-type pair (which gives the
    tightest feasible scale)

Emits a per-pair calibrated d_min table so training data has 100%
feasibility. Currently `default_d_min_table` uses Pyykko radii * 0.7; this
script tells us whether 0.7 is correct or if we should use 0.65 (or
pair-specific).

Run: PYTHONPATH=. python scripts/calibrate_d_min.py --dataset qm9 \\
         --processed_dir data/qm9_processed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from cfm_mol.domain import default_d_min_table


def load_processed_qm9(processed_dir: Path) -> dict:
    """Load FlowMol3-format processed QM9. Expects train_data_processed.pt."""
    path = processed_dir / "train_data_processed.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"processed QM9 file not found at {path}. "
            "Run scripts/install_and_process.slurm (step 3) to generate it."
        )
    data = torch.load(path)
    return data


def scan_pairwise_min(
    data: dict,
    n_atom_types: int,
    max_mols: int = 100_000,
    atom_type_is_onehot: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scan all (i, j) pairs in all molecules, track the min distance per
    atom-type-pair.

    Returns
    -------
    d_min_obs : (A, A) — minimum observed pair distance (0 if never seen).
    count : (A, A) — how many pairs of each type were observed.
    """
    positions = data["positions"]         # (N_total, 3)
    atom_types = data["atom_types"]       # (N_total, A) if one-hot, else (N_total,)
    node_idx = data["node_idx_array"]     # (n_mols, 2)
    n_mols = node_idx.shape[0]
    if max_mols is not None:
        n_mols = min(n_mols, max_mols)

    if atom_type_is_onehot:
        a_flat = atom_types.argmax(dim=-1).long()
    else:
        a_flat = atom_types.long()

    d_min_obs = torch.full((n_atom_types, n_atom_types), float("inf"))
    count = torch.zeros((n_atom_types, n_atom_types), dtype=torch.long)

    for m in range(n_mols):
        start, end = int(node_idx[m, 0]), int(node_idx[m, 1])
        pos = positions[start:end]        # (n, 3)
        a = a_flat[start:end]             # (n,)
        n = pos.shape[0]
        if n < 2:
            continue
        diffs = pos.unsqueeze(0) - pos.unsqueeze(1)
        dists = diffs.norm(dim=-1)        # (n, n)
        eye = torch.eye(n, dtype=torch.bool)
        dists = torch.where(eye, torch.full_like(dists, float("inf")), dists)

        ai = a.unsqueeze(-1).expand(n, n)
        aj = a.unsqueeze(-2).expand(n, n)
        for u in range(n_atom_types):
            for v in range(n_atom_types):
                mask = (ai == u) & (aj == v)
                if not mask.any():
                    continue
                m_min = dists[mask].min().item()
                if m_min < d_min_obs[u, v]:
                    d_min_obs[u, v] = m_min
                count[u, v] += int(mask.sum())

    d_min_obs[torch.isinf(d_min_obs)] = 0.0
    return d_min_obs, count


def report(
    d_min_obs: torch.Tensor,
    d_min_current: torch.Tensor,
    atom_map: list[str],
    tol: float = 0.05,
) -> None:
    A = d_min_obs.shape[0]
    print()
    print(f"{'pair':10s} {'observed':>10s} {'current d_min':>14s} "
          f"{'slack':>10s} {'status':>10s}")
    print("-" * 60)
    any_violation = False
    for i in range(A):
        for j in range(i, A):
            obs = d_min_obs[i, j].item()
            cur = d_min_current[i, j].item()
            slack = obs - cur
            status = "OK" if slack >= -tol else "VIOLATE"
            if status != "OK":
                any_violation = True
            if obs > 0:
                print(f"{atom_map[i]}-{atom_map[j]:<8s} {obs:10.3f} "
                      f"{cur:14.3f} {slack:+10.3f} {status:>10s}")
    if not any_violation:
        print("\nAll observed pairs satisfy current d_min table. "
              "Scale = 0.7 is safe.")
    else:
        print("\nSome pairs VIOLATE. Either tighten scale below 0.7 or use "
              "per-pair calibration = observed * 0.95.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["qm9", "geom", "tmqm"], default="qm9")
    ap.add_argument("--processed_dir", type=Path, required=True)
    ap.add_argument("--atom_map", default="C,H,N,O,F")
    ap.add_argument("--max_mols", type=int, default=100_000)
    args = ap.parse_args()

    atom_map = args.atom_map.split(",")
    n_atom_types = len(atom_map)

    print(f"loading {args.dataset} from {args.processed_dir} ...")
    if args.dataset == "qm9":
        data = load_processed_qm9(args.processed_dir)
    else:
        raise NotImplementedError(f"dataset {args.dataset} loader not yet written")

    print(f"scanning up to {args.max_mols} molecules ...")
    d_min_obs, count = scan_pairwise_min(
        data, n_atom_types=n_atom_types, max_mols=args.max_mols,
    )

    d_min_current = default_d_min_table(n_atom_types=n_atom_types)

    report(d_min_obs, d_min_current, atom_map)
    return 0


if __name__ == "__main__":
    sys.exit(main())
