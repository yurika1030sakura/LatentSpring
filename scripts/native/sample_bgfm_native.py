#!/usr/bin/env python
"""Sample BGFM-Native: energy-coupled flow proposal + residual corrector.

This script is deliberately strict about compute accounting.  The default
path uses the learned residual strain head only; it does not call OMol25 or
xTB during sampling.  Samples are serialized to JSON for the EBMol/FlowMol
benchmark and the independent xTB/MMFF/DFT evaluation scripts.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

_Z = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9,
    "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17,
    "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25,
    "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33,
    "Se": 34, "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41,
    "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49,
    "Sn": 50, "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57,
    "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65,
    "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71, "Hf": 72, "Ta": 73,
    "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80, "Tl": 81,
    "Pb": 82, "Bi": 83,
}


def _sample_to_payload(sample, atom_map, positions_override=None):
    # SampledMolecule exposes `positions`, `atom_types` (symbols), and
    # `atom_charges` after RDKit conversion.  The override is used for
    # corrected coordinates.
    pos = positions_override if positions_override is not None else sample.positions
    symbols = list(sample.atom_types)
    charges = sample.atom_charges
    if charges is None:
        charges = torch.zeros(len(symbols), dtype=torch.long)
    return {
        "atomic_numbers": [int(_Z.get(str(s), 6)) for s in symbols],
        "atom_symbols": [str(s) for s in symbols],
        "positions": pos.detach().cpu().tolist(),
        "charge": int(charges.sum().item()) if torch.is_tensor(charges) else 0,
        "spin": 1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--n_samples", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--flow_nfe", type=int, default=100)
    ap.add_argument("--corrector_steps", type=int, default=50)
    ap.add_argument("--kT", type=float, default=0.025)
    ap.add_argument("--eta_init", type=float, default=1.0e-3)
    ap.add_argument("--eta_final", type=float, default=1.0e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    from flowmol.model_utils.load import model_from_config, read_config_file
    from cfm_mol.native.train_hook import patch_bgfm_native
    from cfm_mol.native.strain_head import residual_energy_and_force
    from cfm_mol.refinement import langevin_corrector, Accounting

    cfg = read_config_file(args.config)
    bgfm_cfg = cfg.get("mol_fm", {}).pop("bgfm", None)
    native_cfg = (bgfm_cfg or {}).get("native", {})
    model = model_from_config(cfg).to(device)
    if native_cfg and native_cfg.get("enabled", False):
        native_cfg.setdefault("n_atom_types", len(cfg["dataset"]["atom_map"]))
        patch_bgfm_native(model, native_cfg)
    ckpt = torch.load(str(args.checkpoint), map_location="cpu")
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    model.eval()

    records = []
    acc = Accounting()
    t0 = time.time()
    n_done = 0
    atom_map = cfg["dataset"]["atom_map"]
    while n_done < args.n_samples:
        bs = min(args.batch_size, args.n_samples - n_done)
        with torch.no_grad():
            samples = model.sample_random_sizes(bs, device=device, n_timesteps=args.flow_nfe)
        acc.flow_nfe += args.flow_nfe * bs
        for sample in samples:
            corrected = None
            if args.corrector_steps > 0 and hasattr(model, "_bgfm_native_strain_head"):
                # Use graph tensors rather than RDKit strings for head inputs.
                g = sample.g.to(device)
                pos = g.ndata["x_1"].to(device)
                atom_types = g.ndata["a_1"].argmax(dim=-1).to(device)
                charges = g.ndata["c_1"].argmax(dim=-1).to(device) if "c_1" in g.ndata else torch.zeros_like(atom_types)
                node_batch_idx = torch.zeros(pos.shape[0], dtype=torch.long, device=device)
                def efun(r):
                    total, strain, base, force = residual_energy_and_force(
                        model._bgfm_native_strain_head,
                        r,
                        atom_types,
                        charges,
                        node_batch_idx,
                        n_graphs=1,
                        create_graph=False,
                        detach_positions=True,
                    )
                    return strain, force
                corrected, acc = langevin_corrector(
                    efun, pos, args.corrector_steps, beta=1.0 / args.kT,
                    eta_init=args.eta_init, eta_final=args.eta_final,
                    metropolis=False, accounting=acc,
                )
            records.append(_sample_to_payload(sample, atom_map, corrected))
            n_done += 1
    acc.wall_clock_s = time.time() - t0
    acc.n_samples = n_done
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"samples": records, "accounting": acc.to_dict(), "args": vars(args)}, f)
    print(f"[bgfm-native-sample] wrote {len(records)} samples -> {args.out}")
    print(f"[bgfm-native-sample] accounting: {acc.to_dict()}")


if __name__ == "__main__":
    main()
