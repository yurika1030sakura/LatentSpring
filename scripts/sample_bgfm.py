"""BGFM sampler: flow proposal + learned-energy Langevin corrector.

This script is intentionally conservative: the external OMol25 oracle is
not called during sampling by default. The flow produces de novo samples;
if the checkpoint contains the BGFM calibrated energy head, a short
learned-energy Langevin corrector refines coordinates while keeping atom
types/charges fixed. All compute is accounted explicitly.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch


def _state_dict_from_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(str(path), map_location="cpu")
    return state.get("state_dict", state)


def _as_list(x: Any) -> list:
    if x is None:
        return []
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _symbols_to_atomic_numbers(symbols: list[str]) -> list[int]:
    try:
        from rdkit import Chem
        pt = Chem.GetPeriodicTable()
        return [int(pt.GetAtomicNumber(str(s))) for s in symbols]
    except Exception:
        # Small fallback for smoke tests without RDKit.
        fallback = {"H": 1, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9,
                    "P": 15, "S": 16, "Cl": 17, "Br": 35, "I": 53}
        return [fallback.get(str(s), 0) for s in symbols]


def _attach_energy_head(model, bgfm_cfg: dict, device: str) -> None:
    """Instantiate the energy head before loading checkpoint keys."""
    if not bool(bgfm_cfg.get("energy_head_enabled", False)):
        return
    from cfm_mol.energy_head import EnergyHead
    head = EnergyHead(
        n_atom_types=int(bgfm_cfg.get("n_atom_types", getattr(model, "n_atom_types", 83))),
        n_charge_classes=int(bgfm_cfg.get("n_charge_classes", 6)),
        hidden_dim=int(bgfm_cfg.get("energy_head_hidden_dim", 128)),
        n_layers=int(bgfm_cfg.get("energy_head_layers", 4)),
        cutoff=float(bgfm_cfg.get("energy_head_cutoff", 6.0)),
    ).to(device)
    model._bgfm_energy_head = head


def _iter_samples(model, batch_size: int, flow_nfe: int, device: str):
    """Use FlowMol3's actual public sampling API."""
    if hasattr(model, "sample_random_sizes"):
        return model.sample_random_sizes(
            n_molecules=batch_size, device=device, n_timesteps=flow_nfe)
    raise AttributeError(
        "Expected FlowMol-style model.sample_random_sizes(...). The previous "
        "sample(n_samples=..., n_steps=...) call path was not compatible with "
        "the vendored FlowMol3 backbone.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--n_samples", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--flow_nfe", type=int, default=100,
                    help="Number of flow ODE steps (K_flow).")
    ap.add_argument("--corrector_steps", type=int, default=50,
                    help="Number of learned-energy Langevin steps. Set 0 to disable.")
    ap.add_argument("--corrector_eta_init", type=float, default=1.0e-3)
    ap.add_argument("--corrector_eta_final", type=float, default=1.0e-4)
    ap.add_argument("--metropolis", action="store_true",
                    help="Single-molecule Metropolis correction; disabled in main experiments unless reported.")
    ap.add_argument("--kT", type=float, default=0.025,
                    help="Temperature in eV; used by the learned-energy corrector.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    from flowmol.model_utils.load import model_from_config, read_config_file
    from cfm_mol.refinement import langevin_corrector, Accounting
    from cfm_mol.energy_head import energy_and_force

    cfg = read_config_file(args.config)
    mol_fm_cfg = cfg.get("mol_fm", {})
    bgfm_cfg = dict(mol_fm_cfg.pop("bgfm", {}) or {})
    atom_map = list(mol_fm_cfg.get("atom_map", []))

    model = model_from_config(cfg).to(device).eval()
    _attach_energy_head(model, bgfm_cfg, device)
    sd = _state_dict_from_checkpoint(args.checkpoint)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[bgfm-sample] missing checkpoint keys: {len(missing)}", flush=True)
    if unexpected:
        print(f"[bgfm-sample] unexpected checkpoint keys: {len(unexpected)}", flush=True)
    model.eval()

    head = getattr(model, "_bgfm_energy_head", None)
    if head is not None:
        head.eval()
    if args.corrector_steps > 0 and head is None:
        print("[bgfm-sample] WARNING: corrector requested but checkpoint/config "
              "has no _bgfm_energy_head; sampling flow-only.", flush=True)

    out_records: list[dict] = []
    accounting = Accounting()
    beta = 1.0 / float(args.kT)
    t_start = time.time()

    n_done = 0
    while n_done < args.n_samples:
        batch_size = min(args.batch_size, args.n_samples - n_done)
        with torch.no_grad():
            samples = _iter_samples(model, batch_size, args.flow_nfe, device)
        accounting.flow_nfe += args.flow_nfe * batch_size

        for mol in samples:
            symbols = [str(s) for s in _as_list(getattr(mol, "atom_types", None))]
            positions = getattr(mol, "positions", None)
            if positions is None:
                continue
            positions_t = torch.as_tensor(positions, dtype=torch.float32, device=device)
            charges_raw = getattr(mol, "atom_charges", None)
            if charges_raw is None:
                charges_t = torch.zeros(positions_t.shape[0], dtype=torch.long, device=device)
            else:
                charges_t = torch.as_tensor(charges_raw, dtype=torch.long, device=device).view(-1)
                if charges_t.numel() != positions_t.shape[0]:
                    charges_t = torch.zeros(positions_t.shape[0], dtype=torch.long, device=device)

            if args.corrector_steps > 0 and head is not None and positions_t.numel() > 0:
                atom_idx = torch.tensor(
                    [atom_map.index(sym) if sym in atom_map else 0 for sym in symbols],
                    dtype=torch.long, device=device)
                # FlowMol charge one-hot in training uses classes roughly shifted by +2.
                charge_idx = (charges_t + 2).clamp(min=0, max=int(bgfm_cfg.get("n_charge_classes", 6)) - 1)
                node_batch_idx = torch.zeros(positions_t.shape[0], dtype=torch.long, device=device)

                def _energy_force(r: torch.Tensor):
                    return energy_and_force(
                        head, r, atom_idx, charge_idx, node_batch_idx, 1,
                        create_graph=False)

                positions_t, accounting = langevin_corrector(
                    _energy_force,
                    positions_t,
                    n_steps=args.corrector_steps,
                    beta=beta,
                    eta_init=args.corrector_eta_init,
                    eta_final=args.corrector_eta_final,
                    metropolis=args.metropolis,
                    accounting=accounting,
                    recenter=True,
                )

            out_records.append({
                "atomic_numbers": _symbols_to_atomic_numbers(symbols),
                "atom_types": symbols,
                "positions": positions_t.detach().cpu().tolist(),
                "charge": int(charges_t.detach().cpu().sum().item()),
                "spin": 1,
            })
            n_done += 1
            if n_done >= args.n_samples:
                break

    accounting.wall_clock_s = time.time() - t_start
    accounting.n_samples = n_done
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "samples": out_records,
        "accounting": accounting.to_dict(),
        "config": {
            "checkpoint": str(args.checkpoint),
            "config": str(args.config),
            "flow_nfe": args.flow_nfe,
            "corrector_steps": args.corrector_steps,
            "kT": args.kT,
            "seed": args.seed,
        },
    }
    with open(args.out, "w") as f:
        json.dump(payload, f)
    print(f"[bgfm-sample] wrote {len(out_records)} samples to {args.out}", flush=True)
    print(f"[bgfm-sample] accounting: {accounting.to_dict()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
