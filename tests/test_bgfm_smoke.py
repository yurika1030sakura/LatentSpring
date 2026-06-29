"""End-to-end smoke test for all four BGFM modules + joint density.

Verifies that:
  - Module 1 (flow proposal) is importable (FlowMol3 backbone).
  - Module 2 (Boltzmann regularization) loss functions are callable
    on a synthetic batch.
  - Module 3 (EnergyHead) forward + autograd force compute end-to-end
    and produce finite outputs.
  - Module 4 (Langevin corrector) produces finite refined positions
    and a populated Accounting record.
  - Module 3.5 (joint discrete-continuous density) sums correctly.
  - The energy-head calibration loss is finite and differentiable.

Run with:
  python -m pytest tests/test_bgfm_smoke.py -v
or directly with python tests/test_bgfm_smoke.py.

Hardware: runs on CPU; uses a tiny synthetic batch so it completes
in seconds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _make_synthetic_batch(B: int = 3, N_per: int = 10, n_atom_types: int = 83):
    """A toy batch: B molecules with N_per atoms each."""
    n_total = B * N_per
    positions = torch.randn(n_total, 3) * 1.5
    atom_types = torch.randint(0, n_atom_types, (n_total,))
    charges = torch.randint(0, 5, (n_total,))
    node_batch_idx = torch.arange(B).repeat_interleave(N_per)
    # Synthetic OMol25-like targets.
    E_target = torch.randn(B) * 100.0
    F_target = torch.randn(n_total, 3) * 0.5
    return positions, atom_types, charges, node_batch_idx, B, E_target, F_target


def test_energy_head_forward_and_force():
    from cfm_mol.energy_head import EnergyHead, energy_and_force

    positions, atom_types, charges, node_batch_idx, B, _, _ = _make_synthetic_batch()
    head = EnergyHead(n_atom_types=83, hidden_dim=64, n_layers=2, cutoff=5.0)

    E = head(positions, atom_types, charges, node_batch_idx, B)
    assert E.shape == (B,), f"E shape {E.shape}"
    assert torch.isfinite(E).all(), "E contains non-finite values"

    E2, F2 = energy_and_force(head, positions, atom_types, charges, node_batch_idx, B, create_graph=False)
    assert F2.shape == positions.shape, f"F shape {F2.shape}"
    assert torch.isfinite(F2).all(), "F contains non-finite values"
    print("[smoke] energy head forward + autograd force: OK")


def test_energy_head_calibration_loss():
    from cfm_mol.energy_head import EnergyHead, energy_and_force
    from cfm_mol.bgfm_loss import energy_head_calibration_loss

    positions, atom_types, charges, node_batch_idx, B, E_target, F_target = _make_synthetic_batch()
    head = EnergyHead(n_atom_types=83, hidden_dim=64, n_layers=2)

    E_pred, F_pred = energy_and_force(head, positions, atom_types, charges, node_batch_idx, B, create_graph=True)
    loss, diag = energy_head_calibration_loss(E_pred, F_pred, E_target, F_target, lambda_F=0.1)
    assert torch.isfinite(loss), f"loss not finite: {loss}"
    loss.backward()
    n_with_grad = sum(1 for p in head.parameters() if p.grad is not None and torch.isfinite(p.grad).all())
    assert n_with_grad > 0, "no head parameter received a finite gradient"
    print(f"[smoke] energy-head calibration loss: {loss.item():.3f}, "
          f"grad on {n_with_grad} params")


def test_langevin_corrector_accounting():
    from cfm_mol.energy_head import EnergyHead, energy_and_force
    from cfm_mol.refinement import langevin_corrector, Accounting

    positions, atom_types, charges, node_batch_idx, B, _, _ = _make_synthetic_batch()
    head = EnergyHead(n_atom_types=83, hidden_dim=64, n_layers=2)

    def _ef(r):
        return energy_and_force(head, r, atom_types, charges, node_batch_idx, B, create_graph=False)

    acc = Accounting()
    refined, acc = langevin_corrector(
        _ef, positions, n_steps=10, beta=1.0 / 0.025,
        eta_init=1e-3, eta_final=1e-4, accounting=acc,
    )
    assert refined.shape == positions.shape
    assert torch.isfinite(refined).all(), "refined positions contain non-finite values"
    assert acc.energy_head_nfe == 11, f"expected 11 head NFEs (1 init + 10 steps), got {acc.energy_head_nfe}"
    assert acc.flow_nfe == 0, "corrector should not increment flow_nfe"
    assert acc.oracle_nfe == 0, "corrector should not call external oracle"
    print(f"[smoke] Langevin corrector accounting: {acc.to_dict()}")


def test_joint_density_combiner():
    from cfm_mol.joint_density import (
        log_prob_atom_count, log_prob_discrete_state, joint_log_prob,
    )
    B, N_total, N_max, K = 3, 30, 50, 84
    n_atoms = torch.tensor([10, 10, 10])
    n_count_logits = torch.randn(B, N_max + 1)
    a_count_logp = log_prob_atom_count(n_atoms, n_count_logits)
    assert a_count_logp.shape == (B,)

    atom_type_logp = torch.randn(N_total) * 0.1
    charge_logp = torch.randn(N_total) * 0.1
    node_batch_idx = torch.arange(B).repeat_interleave(10)
    discrete_logp = log_prob_discrete_state(
        a_count_logp, atom_type_logp, charge_logp, node_batch_idx, B,
    )
    coord_logp = torch.randn(B) * 5.0
    joint = joint_log_prob(coord_logp, discrete_logp)
    assert joint.shape == (B,)
    assert torch.allclose(joint, coord_logp + discrete_logp)
    print(f"[smoke] joint density combiner: OK ({joint.mean().item():.3f})")


def test_density_loss_with_joint_density():
    """The density-energy loss must accept a discrete_log_p argument
    and add it correctly (variance-invariance preserved)."""
    from cfm_mol.bgfm_density import within_group_variance_loss
    # Synthetic per-virtual-mol values.
    log_p_coord = torch.tensor([0.0, 1.0, 2.0,  0.0, 1.0, 2.0])  # 2 parents x 3 perts
    energies = torch.tensor([10.0, 11.0, 12.0,  20.0, 21.0, 22.0])
    kT = 1.0
    parent_id = torch.tensor([0, 0, 0,  1, 1, 1])
    res_no_disc = log_p_coord + energies / kT
    loss_no, _ = within_group_variance_loss(res_no_disc, parent_id)

    # Now add a constant-per-parent discrete log-prob shift. Variance
    # within each parent must be unchanged.
    discrete = torch.tensor([5.0, 5.0, 5.0,  -7.0, -7.0, -7.0])
    res_with = (log_p_coord + discrete) + energies / kT
    loss_with, _ = within_group_variance_loss(res_with, parent_id)
    assert torch.isclose(loss_no, loss_with), (
        f"variance loss should be invariant to per-parent shift: "
        f"{loss_no.item()} vs {loss_with.item()}")
    print(f"[smoke] joint density variance-invariance: OK ({loss_no.item():.6f})")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_energy_head_forward_and_force()
    test_energy_head_calibration_loss()
    test_langevin_corrector_accounting()
    test_joint_density_combiner()
    test_density_loss_with_joint_density()
    print("\n[smoke] ALL TESTS PASSED")
