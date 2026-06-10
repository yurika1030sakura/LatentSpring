"""Sample molecules from a FlowMol3 checkpoint and export as JSON for
external OMol25 energy evaluation.

Runs in the flowmol env (torch 2.2). Produces a JSON file with a list of
{atomic_numbers, positions, charge, spin} entries. The JSON is then fed to
scripts/compute_omol25_energy.py (in the omol25 env).

This subprocess-bridged approach avoids torch-version conflict between
fairchem-core (needs torch >= 2.5) and FlowMol3 (uses torch 2.2).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


# Periodic table up to Bi, matching the current OMol25 atom map coverage.
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--eval_data", type=Path, required=True,
                    help="Eval data dir to sample size distribution from")
    ap.add_argument("--n_samples", type=int, default=100)
    ap.add_argument("--n_timesteps", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--vanilla", action="store_true",
                    help="Skip our patch (pure FlowMol3).")
    ap.add_argument("--discrete-projection", action="store_true",
                    help="Enable valence/connectivity projection during sampling. "
                         "Default is off for bond-free Paper 1 eval.")
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

    from flowmol.model_utils.load import read_config_file, model_from_config
    from cfm_mol.domain import default_d_min_table
    from cfm_mol.flow_model import patch_flowmol

    cfg = read_config_file(str(args.config))
    cfg.get("mol_fm", {}).pop("bgfm", None)
    model = model_from_config(cfg)

    atom_map = cfg["dataset"]["atom_map"]
    if not args.vanilla:
        n_real = len(atom_map)
        has_fake = cfg["mol_fm"].get("fake_atom_p", 0.0) > 0
        has_mask = cfg["mol_fm"].get("parameterization", "") == "ctmc"
        n_total = n_real + int(has_fake) + int(has_mask)
        d_min = torch.zeros(n_total, n_total)
        d_min[:n_real, :n_real] = default_d_min_table(
            n_atom_types=n_real, atom_map=atom_map,
        )
        patch_flowmol(model, d_min, atom_map=atom_map,
                      discrete_projection=args.discrete_projection,
                      train_time_discrete=False)
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

    ckpt = torch.load(str(args.checkpoint), map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Load size distribution from eval_data (fallback to computing)
    n_atoms_path = args.eval_data / "val_data_n_atoms_histogram.pt"
    proc_path = args.eval_data / "val_data_processed.pt"
    if n_atoms_path.exists():
        hist = torch.load(n_atoms_path)
        if isinstance(hist, (list, tuple)):
            hist = hist[-1]
        size_dist = hist.float() / hist.float().sum()
    elif proc_path.exists():
        data = torch.load(proc_path)
        node_idx = data["node_idx_array"]
        n_atoms = (node_idx[:, 1] - node_idx[:, 0]).long()
        max_n = int(n_atoms.max().item()) + 1
        hist = torch.bincount(n_atoms, minlength=max_n).float()
        size_dist = hist / hist.sum()
    else:
        raise FileNotFoundError(f"no size distribution info in {args.eval_data}")

    cat = torch.distributions.Categorical(probs=size_dist)

    print(f"sampling {args.n_samples} molecules on {device} ...")
    all_samples: list[dict] = []
    remaining = args.n_samples
    while remaining > 0:
        nb = min(args.batch_size, remaining)
        sizes = cat.sample((nb,)).clamp_min(3).to(device)
        with torch.no_grad():
            mols = model.sample(
                n_atoms=sizes, n_timesteps=args.n_timesteps, device=device,
                **sample_kwargs,
            )
        for mol in mols:
            try:
                atom_syms = list(mol.atom_types)
                unknown = sorted({s for s in atom_syms if s not in SYMBOL_TO_Z})
                if unknown:
                    raise KeyError(f"unknown element symbol(s): {unknown}")
                z = [SYMBOL_TO_Z[s] for s in atom_syms]
                positions = mol.positions.detach().cpu().numpy().tolist()
                all_samples.append({
                    "atomic_numbers": z,
                    "positions": positions,
                    "charge": 0,
                    "spin": 1,
                })
            except Exception as e:
                print(f"  skip 1 mol (extract error: {e})")
        remaining -= nb
        print(f"  sampled {len(all_samples)}/{args.n_samples}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_samples, f)
    print(f"wrote {len(all_samples)} samples to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
