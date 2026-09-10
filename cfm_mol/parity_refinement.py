"""Explicit inversion averaging for sources and physical energy targets.

This changes the declared source/target. Never silently reinterpret raw-eSEN
results or apply the volume-only entropy identity to a stochastic mixture step.
The exact refinement identity uses the fixed augmented source q0_plus and an
inversion-equivariant deterministic adapter T.
"""
import torch


def randomize_inversion(positions, *, generator):
    """One independent sign per source parent; return positions and lineage signs."""
    if positions.ndim != 3 or not torch.isfinite(positions).all():
        raise ValueError('Finite molecular coordinate batches required')
    signs = 2*torch.randint(2, (len(positions),), generator=generator, device=generator.device)-1
    signs = signs.to(device=positions.device)
    return positions*signs[:, None, None], signs


def evaluate_even_potential(oracle, positions, *, max_request=32):
    """E_plus(x)=(E(x)+E(-x))/2 and its conservative force.

    Raw oracle counts include BOTH orientations. Returns raw components for
    provenance and source-energy replay; this is no surrogate/critic estimate.
    """
    positions = torch.as_tensor(positions).detach().cpu().double()
    if positions.ndim != 3 or len(positions) < 1 or not torch.isfinite(positions).all():
        raise ValueError('Nonempty finite molecular batch required')
    energies, forces = oracle.evaluate_chunked(torch.cat([positions, -positions]), max_request=max_request)
    energy, mirrored_energy = energies.chunk(2)
    force, mirrored_force = forces.chunk(2)
    # Derivative of E(-x) includes the sign from the inversion map.
    even_energy = .5*(energy+mirrored_energy)
    even_force = .5*(force-mirrored_force)
    return even_energy, even_force, {'raw_energy_eV': energy, 'inverted_energy_eV': mirrored_energy,
        'odd_energy_eV': .5*(energy-mirrored_energy), 'odd_force_eV_A': .5*(force+mirrored_force)}


def parity_work_change(raw_work, raw_energy, even_energy, *, kT):
    """Work for an explicit uniform-sign-augmented path and even target.

    The target auxiliary also contains a uniform sign and the original reverse
    path conditioned on the unflipped endpoint. Uniform sign factors cancel.
    This does not require the original source or reverse path to be O(3)-invariant.
    """
    if kT <= 0 or not torch.isfinite(torch.as_tensor(kT)):
        raise ValueError('Positive finite target temperature required')
    if not (raw_work.shape == raw_energy.shape == even_energy.shape):
        raise ValueError('One matched work and energy value per parent required')
    if not all(torch.isfinite(x).all() for x in [raw_work, raw_energy, even_energy]):
        raise ValueError('Finite work and energy values required')
    return raw_work+(even_energy-raw_energy)/kT
