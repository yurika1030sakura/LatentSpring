"""CPU smoke test for the cfm_mol.native package (BGFM-Native).

Covers the four components installed by patch_bgfm_native:
    - ResidualStrainEnergyHead forward + energy_and_force autograd
    - residual_head_loss is finite and differentiable
    - local_boltzmann_bridge_loss is non-negative, parent-shift invariant,
      and reduces to 0 when log p exactly matches -beta * E
    - compute_energy_drift returns an SE(3)-centered (per-graph) drift
      that is zero before t_on and finite after
    - corrector_in_loop_loss returns finite, non-negative loss with
      diagnostics

Run with:
    python tests/test_native_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _batch(B: int = 3, N_per: int = 10):
    n_total = B * N_per
    positions = torch.randn(n_total, 3) * 1.2
    atom_types = torch.randint(0, 83, (n_total,))
    charges = torch.randint(0, 6, (n_total,))
    node_batch_idx = torch.arange(B).repeat_interleave(N_per)
    return positions, atom_types, charges, node_batch_idx, B


def test_residual_strain_head_forward_and_force():
    from cfm_mol.native.strain_head import (
        ResidualStrainEnergyHead, residual_energy_and_force,
    )
    positions, atom_types, charges, node_batch_idx, B = _batch()
    head = ResidualStrainEnergyHead(
        n_atom_types=83, hidden_dim=64, n_rbf=16, cutoff=5.0, n_layers=2,
        baseline_hidden_dim=64,
    )
    total, strain, base = head.forward_parts(
        positions, atom_types, charges, node_batch_idx, B,
    )
    assert total.shape == (B,)
    assert strain.shape == (B,)
    assert base.shape == (B,)
    assert torch.allclose(total, base + strain, atol=1e-5)

    total_, strain_, base_, force = residual_energy_and_force(
        head, positions, atom_types, charges, node_batch_idx, B,
        create_graph=False, detach_positions=True,
    )
    assert force.shape == positions.shape
    assert torch.isfinite(force).all()
    # The strain readout is zero-initialized so initial strain ~ 0 and the
    # force should also be near zero. Allow a small numerical slack.
    assert force.abs().max().item() < 1.0
    print("[native] residual strain head forward + force: OK")


def test_residual_head_loss():
    from cfm_mol.native.strain_head import (
        ResidualStrainEnergyHead, residual_head_loss,
    )
    positions, atom_types, charges, node_batch_idx, B = _batch()
    head = ResidualStrainEnergyHead(
        n_atom_types=83, hidden_dim=64, n_rbf=16, cutoff=5.0, n_layers=2,
        baseline_hidden_dim=64,
    )
    e_tgt = torch.randn(B) * 20.0
    f_tgt = torch.randn_like(positions) * 0.3
    loss, diag = residual_head_loss(
        head, positions, atom_types, charges, node_batch_idx, B,
        e_tgt, f_tgt, lambda_force=0.1,
    )
    assert torch.isfinite(loss)
    loss.backward()
    n_with_grad = sum(1 for p in head.parameters()
                      if p.grad is not None and torch.isfinite(p.grad).all())
    assert n_with_grad > 0
    print(f"[native] residual head loss: L={loss.item():.3f}, "
          f"grad on {n_with_grad} params, "
          f"E_mae_res={diag['native_head_energy_mae_residual']:.3f}")


def test_boltzmann_bridge_loss():
    from cfm_mol.native.boltzmann_bridge import local_boltzmann_bridge_loss
    B, K = 3, 6
    parent_id = torch.arange(B).repeat_interleave(K)
    energies = torch.randn(B * K) * 0.5
    logp = torch.randn(B * K) * 0.3
    beta = 1.0 / 0.025
    loss, diag = local_boltzmann_bridge_loss(
        logp, energies, parent_id, kT=0.025,
    )
    assert torch.isfinite(loss)
    assert loss.item() >= -1e-6
    assert diag["bridge_groups_used"] == float(B)
    # Per-parent constant shifts in logp and energy do not change the
    # softmaxes -> KL is invariant.
    shift_logp = torch.tensor([5.0, -3.0, 7.0]).repeat_interleave(K)
    shift_E = torch.tensor([-2.0, 4.0, -6.0]).repeat_interleave(K)
    loss_shift, _ = local_boltzmann_bridge_loss(
        logp + shift_logp, energies + shift_E, parent_id, kT=0.025,
    )
    assert torch.isclose(loss, loss_shift, atol=1e-4), (
        f"bridge KL not shift-invariant: {loss.item()} vs {loss_shift.item()}"
    )
    # Ideal calibration logp = -beta * E gives ~0 KL.
    loss_ideal, _ = local_boltzmann_bridge_loss(
        -beta * energies, energies, parent_id, kT=0.025,
    )
    assert loss_ideal.item() < 1e-4, (
        f"ideal-calibration KL should be ~0, got {loss_ideal.item()}"
    )
    # Gradient flows.
    logp_g = logp.clone().requires_grad_(True)
    loss_g, _ = local_boltzmann_bridge_loss(
        logp_g, energies, parent_id, kT=0.025,
    )
    loss_g.backward()
    assert logp_g.grad is not None and torch.isfinite(logp_g.grad).all()
    print(f"[native] Boltzmann bridge: KL={loss.item():.4f}, "
          f"ideal-KL={loss_ideal.item():.2e}")


def test_compute_energy_drift_se3():
    from cfm_mol.native.strain_head import ResidualStrainEnergyHead
    from cfm_mol.native.energy_drift import (
        EnergyDriftConfig, alpha_schedule, compute_energy_drift,
    )
    positions, atom_types, charges, node_batch_idx, B = _batch()
    head = ResidualStrainEnergyHead(
        n_atom_types=83, hidden_dim=64, n_rbf=16, cutoff=5.0, n_layers=2,
        baseline_hidden_dim=64,
    )
    cfg = EnergyDriftConfig(
        enabled=True, alpha_max=0.2, t_on=0.65, power=2.0,
        beta=40.0, force_clip=10.0, normalize_force=True,
        detach_force=False,
    )
    # Before t_on -> drift exactly zero.
    drift_early, _ = compute_energy_drift(
        head, positions, atom_types, charges, node_batch_idx, B,
        t=torch.full((B,), 0.1), cfg=cfg, create_graph=False,
    )
    assert torch.allclose(drift_early, torch.zeros_like(drift_early)), \
        "drift should be zero before t_on"
    # After t_on -> finite and per-graph centroid removed.
    drift_late, diag = compute_energy_drift(
        head, positions, atom_types, charges, node_batch_idx, B,
        t=torch.full((B,), 0.95), cfg=cfg, create_graph=False,
    )
    assert torch.isfinite(drift_late).all()
    centroid = torch.zeros(B, 3)
    cnt = torch.zeros(B)
    centroid.index_add_(0, node_batch_idx, drift_late)
    cnt.index_add_(0, node_batch_idx,
                   torch.ones_like(node_batch_idx, dtype=drift_late.dtype))
    centroid = centroid / cnt.clamp_min(1.0).unsqueeze(-1)
    assert centroid.abs().max().item() < 1e-4, "per-graph centroid not removed"

    # alpha schedule monotone in [t_on, 1].
    ts = torch.tensor([0.0, 0.3, 0.65, 0.8, 1.0])
    a = alpha_schedule(ts, cfg)
    assert a[0].item() == 0.0
    assert a[2].item() == 0.0  # t == t_on
    assert a[3].item() > 0.0
    assert a[4].item() >= a[3].item()
    print(f"[native] energy drift: zero pre-t_on, "
          f"late drift norm mean={float(diag['native_drift_norm_mean'].item()):.4f}")


def test_corrector_in_loop_loss():
    from cfm_mol.native.strain_head import ResidualStrainEnergyHead
    from cfm_mol.native.corrector_in_loop import corrector_in_loop_loss
    torch.manual_seed(7)
    positions, atom_types, charges, node_batch_idx, B = _batch()
    head = ResidualStrainEnergyHead(
        n_atom_types=83, hidden_dim=64, n_rbf=16, cutoff=5.0, n_layers=2,
        baseline_hidden_dim=64,
    )
    loss, diag = corrector_in_loop_loss(
        head, positions, atom_types, charges, node_batch_idx, B,
        beta=40.0, j_max=3, eta_init=1e-3, eta_final=1e-4,
        delta_max=0.25, stochastic=True,
    )
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
    print(f"[native] corrector-in-loop: L={loss.item():.4f}, "
          f"J={diag['native_corrector_steps']}, "
          f"energy_inc={diag['native_stab_energy_inc']:.4f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_residual_strain_head_forward_and_force()
    test_residual_head_loss()
    test_boltzmann_bridge_loss()
    test_compute_energy_drift_se3()
    test_corrector_in_loop_loss()
    print("\n[native] ALL TESTS PASSED")
