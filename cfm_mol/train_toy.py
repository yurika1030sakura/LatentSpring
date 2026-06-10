"""Toy CFM training loop -- proves the pipeline works end-to-end.

WHAT THIS DOES:
  1. Generate a synthetic "dataset" of feasible small molecules (random
     Gaussian clouds, retracted onto the steric fibre) -- these are our r_1.
  2. Train LinearVelocityNet by minimising cfm_loss for N steps.
  3. Log the loss curve.
  4. After training, sample new molecules via the constrained Euler ODE and
     check that 100% of samples land on the fibre.

WHAT THIS DOES NOT DO:
  - Learn real chemistry. The MLP ignores atom types and bonds, has no
    equivariance, has no chance of matching real QM9 distributions.
  - Use a real dataset. The "target distribution" is just Gaussian clouds.
  - Train the discrete channel.

PURPOSE:
  - Prove the training pipeline has no bugs end-to-end: loss goes down,
    backprop works, Euler integrator emits feasible molecules.
  - If anything breaks here, it's a cfm_mol bug, not a FlowMol3 integration bug.
    Catches pipeline errors cheaply before Week 1-2 FlowMol3 fork.

RUN: python -m cfm_mol.train_toy
"""
from __future__ import annotations

import time

import torch

from cfm_mol.domain import default_d_min_table, steric_all_ok
from cfm_mol.fibre import (
    euler_step_on_fibre,
    retract,
    sample_prior,
)
from cfm_mol.flow import LinearVelocityNet, cfm_loss


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

def make_dataset(
    n_molecules: int,
    n_atoms: int,
    n_atom_types: int = 10,
    scale: float = 1.8,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.LongTensor, torch.LongTensor]:
    """Generate a synthetic "training set" of feasible molecules.

    Coords: Gaussian clouds, retracted onto the steric fibre.
    Atom types: random in {1, ..., 4} (C/N/O/F in MiDi convention).
    Bond orders: all zero (placeholder -- toy network ignores them anyway).
    """
    torch.manual_seed(seed)
    d_min_table = default_d_min_table(n_atom_types=n_atom_types)
    a = torch.randint(1, 5, (n_molecules, n_atoms))
    r = torch.randn(n_molecules, n_atoms, 3) * scale
    r, converged = retract(r, a, d_min_table)
    assert converged.all(), f"{(~converged).sum()} / {n_molecules} failed to retract"
    b = torch.zeros(n_molecules, n_atoms, n_atoms, dtype=torch.long)
    return r, a, b


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    n_molecules: int = 256,
    n_atoms: int = 8,
    batch_size: int = 32,
    n_steps: int = 500,
    lr: float = 1e-3,
    log_every: int = 50,
):
    """Run the toy training loop. Returns (net, losses, d_min_table, a_for_sampling)."""

    # Dataset
    r1, a, b = make_dataset(n_molecules=n_molecules, n_atoms=n_atoms)
    d_min_table = default_d_min_table(n_atom_types=10)
    assert steric_all_ok(r1, a, d_min_table).all().item(), "dataset contains infeasible molecules"
    print(f"dataset: {n_molecules} molecules of {n_atoms} atoms, all feasible")

    # Network + optimizer
    net = LinearVelocityNet(hidden=64)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    losses = []
    t_start = time.time()
    for step in range(1, n_steps + 1):
        idx = torch.randint(0, n_molecules, (batch_size,))
        r1_batch, a_batch, b_batch = r1[idx], a[idx], b[idx]

        loss = cfm_loss(net, r1_batch, a_batch, b_batch, d_min_table)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

        if step % log_every == 0 or step == 1:
            recent = losses[-min(log_every, len(losses)):]
            print(
                f"step {step:4d} | loss {loss.item():8.4f} | "
                f"avg({log_every}) {sum(recent)/len(recent):8.4f} | "
                f"{time.time()-t_start:5.1f}s"
            )

    return net, losses, d_min_table, a


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample(
    net: torch.nn.Module,
    a: torch.LongTensor,
    d_min_table: torch.Tensor,
    n_samples: int = 64,
    n_atoms: int = 8,
    n_steps: int = 50,
) -> tuple[torch.Tensor, float]:
    """Integrate the constrained flow ODE from t=0 to t=1.

    Uses fixed atom types `a` (no discrete flow yet) and a bogus bond tensor.
    Returns samples and the fibre-validity rate.
    """
    # Use first n_samples atom types; or randomly sample if more needed.
    if n_samples > a.shape[0]:
        idx = torch.randint(0, a.shape[0], (n_samples,))
        a_s = a[idx]
    else:
        a_s = a[:n_samples]
    b_s = torch.zeros(n_samples, n_atoms, n_atoms, dtype=torch.long)

    r = sample_prior(
        n_atoms=n_atoms,
        batch_size=n_samples,
        a=a_s,
        d_min_table=d_min_table,
        scale=3.0,
    )
    dt = 1.0 / n_steps
    for k in range(n_steps):
        t = torch.full((n_samples,), k * dt)
        v = net(r, t, a_s, b_s)
        r = euler_step_on_fibre(r, v, dt, a_s, d_min_table)

    validity = steric_all_ok(r, a_s, d_min_table).float().mean().item()
    return r, validity


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("TOY CFM TRAINING")
    print("=" * 60)

    net, losses, d_min_table, a = train()

    loss_start = sum(losses[:10]) / 10
    loss_end = sum(losses[-10:]) / 10
    print()
    print(f"avg loss first 10 steps: {loss_start:.4f}")
    print(f"avg loss last 10 steps:  {loss_end:.4f}")
    print(f"loss ratio end/start:    {loss_end / loss_start:.3f}")
    if loss_end < 0.5 * loss_start:
        print("verdict: loss decreased >50% -> backprop is wired correctly")
    else:
        print("WARNING: loss did not decrease much. check lr, capacity, or bug.")

    print()
    print("=" * 60)
    print("SAMPLING (constrained Euler flow, 50 steps)")
    print("=" * 60)
    r_samples, validity = sample(net, a, d_min_table, n_samples=64)
    print(f"generated {r_samples.shape[0]} molecules of {r_samples.shape[1]} atoms")
    print(f"fibre validity: {100*validity:.1f}%")
    if validity >= 0.99:
        print("verdict: 100% of Euler-integrated samples are on the fibre -- "
              "constrained inference works end-to-end.")
    else:
        print("WARNING: some samples fell off the fibre; check margin / max_iters "
              "in euler_step_on_fibre.")
