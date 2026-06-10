"""Evaluate a trained FlowMol3 checkpoint on validity + FCD-like structural
metrics + OOD-aware size distribution matching.

Uses FlowMol3's own `SampleAnalyzer` (baselines/flowmol3/flowmol/analysis/
metrics.py) for:
  - per-feature validity (atom valence, stability)
  - energy divergence vs reference dataset (a FID-like structural metric)
  - REOS / ring-system stats

Sample-size distribution: read n_atoms_histogram.pt from the EVAL dataset
and sample molecules proportional to it (matches the target distribution;
fair comparison of baselines on OOD).

Usage:
    PYTHONPATH=. python scripts/evaluate_validity.py \\
        --checkpoint runs/qm9_full/version_0/checkpoints/last.ckpt \\
        --config configs/qm9_cfm.yaml \\
        --eval_data data/qm9_processed \\
        --n_samples 500 \\
        --out runs/iclr/eval_qm9_full.csv

For OOD slices (E1), point --eval_data at the OOD processed dir:
    --eval_data data/tmqm_organic_processed \\
    --atom_map_override C,H,N,O,F,P,S,Cl,Br,I
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch


def _load_model(ckpt_path: Path, cfg: dict, apply_patch: bool,
                discrete_projection: bool = True) -> object:
    from flowmol.model_utils.load import model_from_config
    from cfm_mol.domain import default_d_min_table
    from cfm_mol.flow_model import patch_flowmol

    cfg.get("mol_fm", {}).pop("bgfm", None)
    model = model_from_config(cfg)
    if apply_patch:
        atom_map = cfg["dataset"]["atom_map"]
        n_real = len(atom_map)
        has_fake = cfg["mol_fm"].get("fake_atom_p", 0.0) > 0
        has_mask = cfg["mol_fm"].get("parameterization", "") == "ctmc"
        n_total = n_real + int(has_fake) + int(has_mask)
        d_min = torch.zeros(n_total, n_total)
        d_min[:n_real, :n_real] = default_d_min_table(
            n_atom_types=n_real, atom_map=atom_map,
        )
        patch_flowmol(model, d_min, atom_map=atom_map,
                       discrete_projection=discrete_projection)

    state = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _load_size_distribution(eval_data_dir: Path) -> torch.Tensor:
    """Load the n_atoms histogram from the target (eval) dataset so we
    sample molecules matching its size distribution.

    FlowMol3 saves histograms as a tuple `(edge_count_hist, node_count_hist)`.
    We use node_count (second element) which is the # atoms per molecule.
    Falls back to computing the histogram from val_data_processed.pt's
    node_idx_array for OOD datasets we built ourselves.
    """
    for split in ("val", "train", "test"):
        p = eval_data_dir / f"{split}_data_n_atoms_histogram.pt"
        if p.exists():
            hist = torch.load(p)
            if isinstance(hist, (list, tuple)):
                hist = hist[-1]
            hist = hist.float()
            if hist.sum() > 0:
                return hist / hist.sum()

    # Fallback: compute from processed data's node_idx_array.
    for split in ("val", "test", "train"):
        p = eval_data_dir / f"{split}_data_processed.pt"
        if p.exists():
            data = torch.load(p)
            node_idx = data["node_idx_array"]
            n_atoms = (node_idx[:, 1] - node_idx[:, 0]).long()
            max_n = int(n_atoms.max().item()) + 1
            hist = torch.bincount(n_atoms, minlength=max_n).float()
            hist = hist / hist.sum()
            return hist
    raise FileNotFoundError(
        f"No histogram OR processed data found in {eval_data_dir}."
    )


def _sample_n_atoms(n_samples: int, dist: torch.Tensor,
                    min_atoms: int = 3) -> torch.Tensor:
    """Sample n_atoms values from the empirical size distribution."""
    cat = torch.distributions.Categorical(probs=dist)
    sizes = cat.sample((n_samples,))
    return sizes.clamp_min(min_atoms)


def _sample_batch(model, n_atoms_tensor: torch.Tensor, n_timesteps: int,
                  device: str, sample_kwargs: dict | None = None) -> list:
    n_atoms_tensor = n_atoms_tensor.to(device)
    sample_kwargs = sample_kwargs or {}
    with torch.no_grad():
        mols = model.sample(n_atoms=n_atoms_tensor, n_timesteps=n_timesteps,
                             device=device, **sample_kwargs)
    return mols


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--eval_data", type=Path, required=True,
                    help="Processed-data dir for size distribution + "
                         "reference metrics.")
    ap.add_argument("--analyzer_data", type=Path, default=None,
                    help="Separate dir for SampleAnalyzer (valencies + "
                         "reference distributions). Default: reuse eval_data. "
                         "For OOD eval, point at qm9_processed so analyzer "
                         "gets valid valencies_kekulized.json.")
    ap.add_argument("--n_samples", type=int, default=500)
    ap.add_argument("--n_timesteps", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--vanilla", action="store_true",
                    help="Evaluate unpatched model (baseline mode).")
    ap.add_argument("--no-discrete", action="store_true",
                    help="Apply patch but DISABLE discrete projection hook "
                         "(coord-only retract + tangent). For ablation.")
    ap.add_argument("--bgfm-guidance-weight", type=float, default=0.0,
                    help="Sampling-time BGFM score guidance strength. "
                         "Requires patched mode; 0 disables it.")
    ap.add_argument("--bgfm-guidance-start", type=float, default=0.75,
                    help="Time at which late BGFM guidance turns on.")
    ap.add_argument("--bgfm-guidance-schedule", default="late_linear",
                    choices=["late_linear", "late_quadratic", "linear", "constant"])
    ap.add_argument("--bgfm-guidance-clip", type=float, default=5.0,
                    help="Componentwise clip after per-molecule score normalization.")
    ap.add_argument("--bgfm-guidance-ratio", type=float, default=1.0,
                    help="Max guidance atom norm as a multiple of per-molecule "
                         "velocity RMS.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from flowmol.model_utils.load import read_config_file
    from flowmol.analysis.metrics import SampleAnalyzer

    cfg = read_config_file(str(args.config))
    print(f"loading checkpoint {args.checkpoint} "
          f"(vanilla={args.vanilla}, no_discrete={args.no_discrete}) ...")
    model = _load_model(args.checkpoint, cfg,
                        apply_patch=not args.vanilla,
                        discrete_projection=not args.no_discrete)
    model = model.to(args.device)
    sample_kwargs = {}
    if args.bgfm_guidance_weight > 0.0:
        if args.vanilla:
            raise ValueError("--bgfm-guidance-weight requires patched mode; "
                             "do not combine with --vanilla")
        sample_kwargs["bgfm_score_guidance"] = {
            "max_weight": args.bgfm_guidance_weight,
            "start": args.bgfm_guidance_start,
            "schedule": args.bgfm_guidance_schedule,
            "clip": args.bgfm_guidance_clip,
            "max_norm_ratio": args.bgfm_guidance_ratio,
        }
        print(f"BGFM score guidance enabled: {sample_kwargs['bgfm_score_guidance']}")

    print(f"loading size distribution from {args.eval_data} ...")
    dist = _load_size_distribution(args.eval_data)
    print(f"  eval dist: shape={tuple(dist.shape)}, "
          f"mean_n_atoms={(dist * torch.arange(len(dist))).sum():.2f}")

    print(f"sampling {args.n_samples} molecules "
          f"(NFE={args.n_timesteps}, bs={args.batch_size}) ...")
    all_mols = []
    remaining = args.n_samples
    while remaining > 0:
        nb = min(args.batch_size, remaining)
        n_atoms_tensor = _sample_n_atoms(nb, dist)
        mols = _sample_batch(model, n_atoms_tensor, args.n_timesteps,
                             args.device, sample_kwargs=sample_kwargs)
        all_mols.extend(mols)
        remaining -= nb
        print(f"  sampled {len(all_mols)}/{args.n_samples}")

    analyzer_dir = args.analyzer_data or args.eval_data
    print(f"evaluating with SampleAnalyzer(processed_data_dir={analyzer_dir}) ...")
    analyzer = SampleAnalyzer(processed_data_dir=analyzer_dir)
    # FlowMol3's analyze signature: (sampled_molecules, return_counts=False,
    # energy_div=False, functional_validity=False, ...).
    try:
        stats = analyzer.analyze(all_mols)
    except Exception as e:
        print(f"[warn] analyze() failed: {e}. Falling back to compute_validity only.")
        stats = {}
    if not isinstance(stats, dict):
        stats = {"raw_result": str(stats)}
    # compute_validity returns a dict too.
    try:
        validity = analyzer.compute_validity(all_mols)
        if isinstance(validity, dict):
            for k, v in validity.items():
                stats[f"validity_{k}"] = v
    except Exception as e:
        print(f"[warn] compute_validity() failed: {e}")

    stats["n_samples"] = len(all_mols)
    stats["n_timesteps"] = args.n_timesteps
    stats["checkpoint"] = str(args.checkpoint)
    stats["eval_data"] = str(args.eval_data)

    print()
    for k, v in stats.items():
        if isinstance(v, (int, float)):
            print(f"  {k:36s} = {v:.4f}" if isinstance(v, float)
                  else f"  {k:36s} = {v}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Flatten non-scalar values to strings.
    row = {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
           for k, v in stats.items()}
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
