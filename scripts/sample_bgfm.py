"""BGFM sampler: flow proposal + learned-energy Langevin corrector.

Produces N molecules from a trained BGFM checkpoint and writes them
to a JSON file in the same format as the standard generation
benchmarks consume. Every sampling configuration reports an
accounting record with:

  flow_nfe          number of flow ODE steps
  energy_head_nfe   number of E_psi forward passes
  oracle_nfe        number of external OMol25 or xTB calls (default 0)
  wall_clock_s      sampling wall-clock seconds
  rejected          Metropolis rejection count (if enabled)
  n_samples         number of accepted samples produced

This is the entry point used by Experiments 1 (standard generation),
2 (independent physical quality), and 3 (quality-diversity-compute
Pareto). For Pareto sweeps, run with varying --corrector_steps and
--flow_nfe.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--n_samples", type=int, default=10000)
    ap.add_argument("--flow_nfe", type=int, default=100,
                    help="Number of flow ODE steps (K_flow).")
    ap.add_argument("--corrector_steps", type=int, default=50,
                    help="Number of Langevin refinement steps (J). "
                         "Set to 0 to disable the corrector.")
    ap.add_argument("--corrector_eta_init", type=float, default=1.0e-3)
    ap.add_argument("--corrector_eta_final", type=float, default=1.0e-4)
    ap.add_argument("--metropolis", action="store_true")
    ap.add_argument("--kT", type=float, default=0.025,
                    help="Temperature in eV; used by the corrector.")
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
    model = model_from_config(cfg).to(device).eval()
    state = torch.load(str(args.checkpoint), map_location="cpu")
    sd = state.get("state_dict", state)
    model.load_state_dict(sd, strict=False)

    out_records: list[dict] = []
    accounting = Accounting()
    beta = 1.0 / float(args.kT)

    t_start = time.time()
    n_done = 0
    while n_done < args.n_samples:
        batch_size = min(64, args.n_samples - n_done)

        # ---- Flow proposal ----
        with torch.no_grad():
            samples = model.sample(
                n_samples=batch_size,
                n_steps=args.flow_nfe,
            )
        accounting.flow_nfe += args.flow_nfe * batch_size

        # ---- Optional Langevin corrector under E_psi ----
        if args.corrector_steps > 0 and hasattr(model, "energy_head"):
            head = model.energy_head
            for sample in samples:
                positions = sample["positions"].to(device).clone()
                node_batch_idx = torch.zeros(
                    positions.shape[0], dtype=torch.long, device=device,
                )
                n_graphs = 1

                def _energy_force(r: torch.Tensor):
                    feats = model.scalar_features_at(sample, r)
                    return energy_and_force(
                        head, lambda r_: feats, r, node_batch_idx, n_graphs,
                        create_graph=False,
                    )

                positions, accounting = langevin_corrector(
                    _energy_force,
                    positions,
                    n_steps=args.corrector_steps,
                    beta=beta,
                    eta_init=args.corrector_eta_init,
                    eta_final=args.corrector_eta_final,
                    metropolis=args.metropolis,
                    accounting=accounting,
                )
                sample["positions"] = positions.detach().cpu()

        # ---- Serialize ----
        for sample in samples:
            out_records.append({
                "atomic_numbers": sample["atomic_numbers"].tolist(),
                "positions": sample["positions"].tolist(),
                "charge": int(sample.get("charge", 0)),
                "spin": int(sample.get("spin", 1)),
            })
            n_done += 1

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
    print(f"[bgfm-sample] wrote {len(out_records)} samples to {args.out}",
          flush=True)
    print(f"[bgfm-sample] accounting: {accounting.to_dict()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
