"""Physics-informed drift via OMol25 neural network potential.

At sampling time, augment the learned flow-matching velocity with the
negative gradient of a DFT-quality energy function learned by OMol25
(Meta FAIR, Levine et al. 2025, arXiv:2505.08762). This pushes samples
toward low-energy configurations without retraining the flow model.

Mathematical form:
    dr/dt = v_theta(r, t, a, b) - lambda(t) * grad_r E_OMol25(r, a, b)

where:
    - v_theta is FlowMol3's learned velocity field
    - E_OMol25 is the energy of the molecule at level
      DFT(wB97M-V/def2-TZVPD) predicted by eSEN-sm-conserving
    - lambda(t) is a schedule that ramps up physics drift toward t -> 1
      (dominating near the end when we want realistic geometries)

Model choice: **eSEN-sm-conserving**
  - Energy-conserving (F = -dE/dx via autograd) — required for
    differentiable drift
  - Small enough for batched inference on a single A100 (~80-250 ms per
    gradient call on 30-50 atom molecules)
  - Trained on full OMol25: 83 elements, including transition metals
    (Pd/Ni/Rh/Ir/Cu/Fe/Zn) — enables future organometallic catalyst work
    without retraining the potential
"""
from __future__ import annotations

from typing import Callable

import torch


# Mapping from FlowMol3's 5-atom QM9 map to atomic numbers used by OMol25.
QM9_ATOM_MAP_TO_Z = {
    "C": 6, "H": 1, "N": 7, "O": 8, "F": 9,
}
# Extended map for GEOM-Drugs 10-organic.
ORGANIC_ATOM_MAP_TO_Z = {
    "C": 6, "H": 1, "N": 7, "O": 8, "F": 9,
    "P": 15, "S": 16, "Cl": 17, "Br": 35, "I": 53,
}
# Full broad map including transition metals for tmQM.
FULL_ATOM_MAP_TO_Z = {
    **ORGANIC_ATOM_MAP_TO_Z,
    "B": 5, "Si": 14, "As": 33, "Se": 34,
    "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26,
    "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
    "Mo": 42, "Ru": 44, "Rh": 45, "Pd": 46,
    "Ir": 77, "Pt": 78,
}


class OMol25Drift:
    """Wraps a pretrained OMol25 ML potential to compute energy gradients
    for physics-informed drift during flow matching sampling.

    Lazily loads the model on first call (avoids cold-start for tests).
    """

    def __init__(
        self,
        model_name: str = "esen_sm_conserving_all",
        atom_map: list[str] | None = None,
        device: str = "cuda",
        charge: int = 0,
        spin: int = 1,
    ):
        self.model_name = model_name
        self.atom_map = atom_map or ["C", "H", "N", "O", "F"]
        # Fall back to full map lookup if requested atom not in QM9 subset.
        self._symbol_to_z = {
            sym: FULL_ATOM_MAP_TO_Z.get(sym, 6)
            for sym in self.atom_map
        }
        self.device = device
        self.charge = charge
        self.spin = spin
        self._calc = None   # lazy-initialised FAIRChemCalculator

    def _init_calc(self):
        if self._calc is not None:
            return
        from fairchem.core import FAIRChemCalculator
        from fairchem.core.units.mlip_unit import load_predict_unit
        pred = load_predict_unit(self.model_name, device=self.device)
        self._calc = FAIRChemCalculator(pred)

    def energy_and_grad(
        self,
        positions: torch.Tensor,         # (N, 3) coords
        atom_indices: torch.LongTensor,  # (N,) atom-map indices
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute E(r) and dE/dr for a single molecule.

        Returns (energy_scalar, gradient_(N,3)).
        """
        self._init_calc()

        # Convert atom-map indices -> atomic numbers.
        z = torch.tensor(
            [self._symbol_to_z[self.atom_map[int(i)]]
             for i in atom_indices],
            dtype=torch.long, device=positions.device,
        )

        # Build an ASE-compatible Atoms object for the calculator.
        # fairchem-core's FAIRChemCalculator expects ASE-style input.
        import ase
        atoms = ase.Atoms(
            numbers=z.cpu().numpy(),
            positions=positions.detach().cpu().numpy(),
            charges=None,    # we use total-charge via info
        )
        atoms.info["charge"] = self.charge
        atoms.info["spin"] = self.spin
        atoms.calc = self._calc

        # Energy + forces (forces = -dE/dx).
        e = atoms.get_potential_energy()
        f = atoms.get_forces()   # (N, 3)

        energy = torch.tensor(float(e), device=positions.device)
        gradient = -torch.tensor(f, dtype=positions.dtype,
                                  device=positions.device)
        return energy, gradient

    def drift_velocity(
        self,
        positions: torch.Tensor,
        atom_indices: torch.LongTensor,
        lambda_t: float,
    ) -> torch.Tensor:
        """Return -lambda(t) * dE/dr, ready to add to velocity."""
        _, grad = self.energy_and_grad(positions, atom_indices)
        return -lambda_t * grad


def linear_schedule(t: float, lam_start: float = 0.0,
                     lam_end: float = 0.1) -> float:
    """Default lambda(t) schedule: ramp from 0 at t=0 to lam_end at t=1.
    Early in sampling (t near 0), we trust the learned velocity; late
    (t near 1), we trust the physics prior more.
    """
    return lam_start + (lam_end - lam_start) * t


def cosine_schedule(t: float, lam_end: float = 0.1) -> float:
    """Smoother: lambda(t) = lam_end * (1 - cos(pi * t)) / 2."""
    import math
    return lam_end * (1.0 - math.cos(math.pi * t)) / 2.0
