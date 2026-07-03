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


# Symbol -> atomic number over the full periodic table. OMol25 covers 83
# elements (H..Bi). The previous table listed only ~30 symbols and silently
# mapped every unknown element to carbon (``.get(s, 6)``), which corrupts the
# teacher's energy/force for any element outside that set -- see
# notes/appendix_hbc.tex, Assumption 3 (well-posed conditional energy).
import logging as _logging

_LOG = _logging.getLogger(__name__)

# Index == atomic number. Entry 0 is a placeholder / fake-atom token.
_PERIODIC_SYMBOLS: list[str] = [
    "X",
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt",
    "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
]
_ATOM_MAP_TO_Z: dict[str, int] = {s: z for z, s in enumerate(_PERIODIC_SYMBOLS)}


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
        self._atomic_z = []
        for s in self.atom_map:
            z = _ATOM_MAP_TO_Z.get(s)
            if z is None:
                _LOG.warning(
                    "physics_drift: unknown element symbol %r in atom_map; "
                    "falling back to carbon (Z=6)", s,
                )
                z = 6
            self._atomic_z.append(z)
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
        mol_charge: torch.Tensor | None = None,
        mol_spin: torch.Tensor | None = None,
        return_valid: bool = False,
    ):
        """Compute OMol25 force per atom for a DGL-batched graph.

        Args
        ----
        x_t : (total_atoms, 3) coordinates on device
        atom_type_logits : (total_atoms, A) one-hot/simplex; we argmax
        node_batch_idx : (total_atoms,) molecule index per atom
        mol_charge : (n_mols,) optional total charge per molecule (default 0)
        mol_spin : (n_mols,) optional spin multiplicity per molecule (default 1)

        Returns
        -------
        forces : (total_atoms, 3) same device, eV/A, clipped to
        ``cfg.clip_force``.
        """
        device = x_t.device
        x_cpu = x_t.detach().cpu().numpy()
        a_hard = atom_type_logits.argmax(dim=-1).cpu().numpy()
        nbi = node_batch_idx.cpu().numpy()
        q_cpu = None if mol_charge is None else mol_charge.detach().cpu().tolist()
        sp_cpu = None if mol_spin is None else mol_spin.detach().cpu().tolist()

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
            q = 0 if (q_cpu is None or b >= len(q_cpu)) else int(round(q_cpu[b]))
            sp = 1 if (sp_cpu is None or b >= len(sp_cpu)) else int(round(sp_cpu[b]))
            samples.append({
                "atomic_numbers": z,
                "positions": pos,
                "charge": q,
                "spin": sp,
            })
            index_map.append(atom_idxs)

        # Filter out None placeholders before RPC call.
        samples_valid = [s for s in samples if s is not None]
        try:
            results = self.srv.compute_batch(samples_valid)
        except Exception as e:
            # On failure, return zero drift (graceful degradation).
            if return_valid:
                return (torch.zeros_like(x_t),
                        torch.zeros(x_t.shape[0], dtype=torch.bool, device=device))
            return torch.zeros_like(x_t)

        # Re-scatter forces to (total_atoms, 3). Track which atoms carry a VALID
        # teacher force (molecule RPC returned ok); atoms of failed molecules keep
        # zero force AND valid=False, so the caller can mask them out rather than
        # training toward a spurious F/kT = 0 target.
        forces = torch.zeros_like(x_t.cpu())
        valid = torch.zeros(x_t.shape[0], dtype=torch.bool)
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
                valid[atom_i] = True
        # Clip to prevent large-force blowups.
        forces = forces.clamp(-self.cfg.clip_force, self.cfg.clip_force)
        if return_valid:
            return forces.to(device), valid.to(device)
        return forces.to(device)
