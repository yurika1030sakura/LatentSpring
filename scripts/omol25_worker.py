"""Persistent OMol25 worker exposing an XML-RPC API.

Runs in the omol25 conda env (torch 2.8 + fairchem-core 2.19).
The flowmol env (torch 2.2) connects via xmlrpc.client to compute energy
+ forces per batch during sampling, without importing fairchem directly.

Protocol: single method `compute_batch(samples)` where samples is a list
of dicts {atomic_numbers, positions, charge, spin}. Returns a list of
dicts {energy_eV, forces_eV_per_A (N x 3)}.

Start:
    conda activate envs/omol25
    python scripts/omol25_worker.py --port 5900 --device cpu

Client usage (in flowmol env):
    import xmlrpc.client
    srv = xmlrpc.client.ServerProxy("http://localhost:5900", allow_none=True)
    result = srv.compute_batch([{...}, ...])
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from xmlrpc.server import SimpleXMLRPCServer


CKPT_DEFAULT = (
    "/n/netscratch/ryl_lab/Lab/hf_cache/models--facebook--OMol25/"
    "snapshots/039b7070e59d1537e56c93a3a455263d062ed9c8/checkpoints/"
    "esen_sm_conserving_all.pt"
)


def _load_calc(ckpt: str, device: str = "cpu"):
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit
    pred = load_predict_unit(ckpt, device=device)
    return FAIRChemCalculator(pred)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5900)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--ckpt", default=CKPT_DEFAULT)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    print(f"loading OMol25 model on {args.device} ...")
    calc = _load_calc(args.ckpt, device=args.device)
    print("loaded")

    import ase
    import numpy as np

    def compute_batch(samples):
        """Process a batch of molecules; return [{energy_eV, forces}, ...]."""
        out = []
        for s in samples:
            try:
                atoms = ase.Atoms(
                    numbers=s["atomic_numbers"],
                    positions=np.array(s["positions"], dtype=float),
                )
                atoms.info["charge"] = int(s.get("charge", 0))
                atoms.info["spin"] = int(s.get("spin", 1))
                atoms.calc = calc
                e = float(atoms.get_potential_energy())
                f = atoms.get_forces().tolist()
                out.append({"energy_eV": e, "forces": f, "ok": True})
            except Exception as exc:
                out.append({
                    "energy_eV": None, "forces": None, "ok": False,
                    "error": f"{type(exc).__name__}: {str(exc)[:140]}",
                })
        return out

    def health():
        return {"status": "ok", "t": time.time()}

    server = SimpleXMLRPCServer(
        (args.host, args.port),
        allow_none=True,
        logRequests=False,
    )
    server.register_function(compute_batch, "compute_batch")
    server.register_function(health, "health")
    print(f"serving on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    sys.exit(main())
