"""Client-side drift utility used in the sampling loop (flowmol env).

Connects to a persistent OMol25 worker over XML-RPC to fetch energy +
force for each sample during the Euler integration. The velocity field
output is augmented in-place with the force-based drift term:

    v_hybrid = v_theta + lambda(t) * F / ||F||_inf_per_mol

The normalization keeps drift magnitude at the same scale as v_theta
regardless of absolute energy scale.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import math
import torch
import xmlrpc.client


# Atom-index symbol tables: FlowMol3 QM9 uses indices into atom_map.
# We convert to atomic numbers for OMol25.
_ATOM_MAP_TO_Z: dict[str, int] = {
    "C": 6, "H": 1, "N": 7, "O": 8, "F": 9,
    "P": 15, "S": 16, "Cl": 17, "Br": 35, "I": 53,
    "B": 5, "Si": 14,
    # 3d TMs
    "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25,
    "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
    # 4d/5d
    "Mo": 42, "Ru": 44, "Rh": 45, "Pd": 46, "Pt": 78, "Ir": 77,
}


@dataclass
class DriftConfig:
    """Schedule + weighting for the physics drift."""
    max_weight: float = 0.5          # lambda(1) = max drift strength
    schedule: str = "late_only"      # 'constant', 'linear', 'cosine', 'late_only'
    late_threshold: float = 0.7      # for 'late_only': drift active only for t >= this
    clip_force: float = 10.0         # force clip (eV/A) to prevent blowups
    rpc_url: str = "http://127.0.0.1:5900"


def lambda_schedule(t: float, cfg: DriftConfig) -> float:
    if cfg.schedule == "constant":
        return cfg.max_weight
    if cfg.schedule == "linear":
        return cfg.max_weight * t
    if cfg.schedule == "cosine":
        return cfg.max_weight * 0.5 * (1 - math.cos(math.pi * t))
    if cfg.schedule == "late_only":
        if t < cfg.late_threshold:
            return 0.0
        return cfg.max_weight * (t - cfg.late_threshold) / (1 - cfg.late_threshold)
    raise ValueError(f"unknown schedule: {cfg.schedule}")


class OMol25DriftClient:
    """Thin client over XML-RPC to the OMol25 worker."""

    def __init__(self, cfg: DriftConfig, atom_map: Sequence[str]):
        self.cfg = cfg
        self.srv = xmlrpc.client.ServerProxy(cfg.rpc_url, allow_none=True)
        self.atom_map = list(atom_map)
        self._atomic_z: list[int] = [
            _ATOM_MAP_TO_Z.get(s, 6) for s in self.atom_map
        ]
        # Connectivity check (fails fast if worker not up).
        try:
            h = self.srv.health()
            assert h["status"] == "ok", h
        except Exception as e:
            raise RuntimeError(
                f"cannot reach OMol25 worker at {cfg.rpc_url}: {e}"
            )

    def compute_forces_dgl(
        self,
        x_t: torch.Tensor,
        atom_type_logits: torch.Tensor,
        node_batch_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Compute OMol25 force per atom for a DGL-batched graph.

        Args
        ----
        x_t : (total_atoms, 3) coordinates on device
        atom_type_logits : (total_atoms, A) one-hot/simplex; we argmax
        node_batch_idx : (total_atoms,) molecule index per atom

        Returns
        -------
        forces : (total_atoms, 3) same device, eV/A, clipped to
        ``cfg.clip_force``.
        """
        device = x_t.device
        x_cpu = x_t.detach().cpu().numpy()
        a_hard = atom_type_logits.argmax(dim=-1).cpu().numpy()
        nbi = node_batch_idx.cpu().numpy()

        # Build per-molecule sample dicts.
        n_mols = int(nbi.max()) + 1
        samples = []
        index_map: list[list[int]] = []
        for b in range(n_mols):
            atom_idxs = [int(i) for i, bi in enumerate(nbi) if bi == b]
            if not atom_idxs:
                samples.append(None)
                index_map.append([])
                continue
            z = [int(self._atomic_z[int(a_hard[i])]) for i in atom_idxs]
            pos = [x_cpu[i].tolist() for i in atom_idxs]
            samples.append({
                "atomic_numbers": z,
                "positions": pos,
                "charge": 0,
                "spin": 1,
            })
            index_map.append(atom_idxs)

        # Filter out None placeholders before RPC call.
        samples_valid = [s for s in samples if s is not None]
        try:
            results = self.srv.compute_batch(samples_valid)
        except Exception as e:
            # On failure, return zero drift (graceful degradation).
            return torch.zeros_like(x_t)

        # Re-scatter forces to (total_atoms, 3).
        forces = torch.zeros_like(x_t.cpu())
        r_iter = iter(results)
        for b, idxs in enumerate(index_map):
            if not idxs:
                continue
            r = next(r_iter)
            if not r.get("ok"):
                continue
            f_arr = r["forces"]
            for j, atom_i in enumerate(idxs):
                forces[atom_i] = torch.tensor(f_arr[j])
        # Clip to prevent large-force blowups.
        forces = forces.clamp(-self.cfg.clip_force, self.cfg.clip_force)
        return forces.to(device)
