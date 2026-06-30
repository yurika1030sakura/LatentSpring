"""BGFM-Native: energy-coupled Boltzmann flow matching components.

The modules in this package implement the high-risk/high-upside version
of BGFM described in the ICLR framing:

    v_BGFM = v_gen + alpha(t) P[- beta * grad_r DeltaU_psi]

where DeltaU_psi is a composition-conditioned residual strain energy
calibrated from an external neural potential.  The package is designed
as an add-on to the existing FlowMol3-style code path: it can be enabled
from the `mol_fm.bgfm.native` config block without replacing the base
FlowMol3 data/model construction pipeline.
"""

from .strain_head import ResidualStrainEnergyHead, residual_energy_and_force
from .boltzmann_bridge import local_boltzmann_bridge_loss
from .energy_drift import patch_energy_coupled_vector_field
from .corrector_in_loop import corrector_in_loop_loss
from .train_hook import patch_bgfm_native

__all__ = [
    "ResidualStrainEnergyHead",
    "residual_energy_and_force",
    "local_boltzmann_bridge_loss",
    "patch_energy_coupled_vector_field",
    "corrector_in_loop_loss",
    "patch_bgfm_native",
]
