"""Tiny end-to-end check of the cross-env teacher path used by Experiment A:
flowmol env -> xmlrpc -> omol25 worker -> eSEN forces, through
cfm_mol/physics_drift.py::OMol25DriftClient (the same client the on-policy
training branch uses). Prints one JSON line."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--rpc_url", default="http://127.0.0.1:28951")
a = ap.parse_args()

from cfm_mol.physics_drift import OMol25DriftClient, DriftConfig, _PERIODIC_SYMBOLS

atom_map = list(_PERIODIC_SYMBOLS[1:84])
cli = OMol25DriftClient(DriftConfig(rpc_url=a.rpc_url, clip_force=1e9), atom_map)

# Two tiny molecules: water and methane, laid out as a DGL-style flat batch.
# atom index into atom_map: H=0, C=5, O=7
pos = torch.tensor([
    [0.000, 0.000, 0.119], [0.000, 0.763, -0.477], [0.000, -0.763, -0.477],   # H2O
    [0.000, 0.000, 0.000], [0.629, 0.629, 0.629], [-0.629, -0.629, 0.629],
    [0.629, -0.629, -0.629], [-0.629, 0.629, -0.629],                          # CH4
], dtype=torch.float32)
idx = torch.tensor([7, 0, 0, 5, 0, 0, 0, 0])
nbi = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1])
onehot = torch.zeros(8, len(atom_map)); onehot[torch.arange(8), idx] = 1.0

t0 = time.time()
F, valid = cli.compute_forces_dgl(pos, onehot, nbi,
                                  mol_charge=torch.zeros(2), return_valid=True)
# same geometry displaced -> the "x_t" side of the mismatch measurement
pos2 = pos + 0.25 * torch.randn_like(pos)
F2, valid2 = cli.compute_forces_dgl(pos2, onehot, nbi,
                                    mol_charge=torch.zeros(2), return_valid=True)
print(json.dumps({
    "rpc_url": a.rpc_url,
    "seconds_for_2_calls": round(time.time() - t0, 2),
    "n_valid_x1": int(valid.sum()), "n_valid_xt": int(valid2.sum()),
    "F_x1_norms": [round(v, 4) for v in F.norm(dim=-1).tolist()],
    "F_xt_norms": [round(v, 4) for v in F2.norm(dim=-1).tolist()],
    "mismatch_norms": [round(v, 4) for v in (F2 - F).norm(dim=-1).tolist()],
    "all_finite": bool(torch.isfinite(F).all() and torch.isfinite(F2).all()),
}))
