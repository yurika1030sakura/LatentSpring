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


def test_deterministic_corrector():
    """SPEC item D: deterministic corrector noise mode -> zero displacement
    std and zero added noise term across the J unrolled steps.

    We monkey-patch torch.randn_like inside the corrector module to record
    every call. If the deterministic path is honoured, no randn_like calls
    should occur during the corrector unroll, so the recorded count is 0
    and the implied added-noise contribution is exactly zero. We also
    verify that two seeded calls produce bit-identical losses (no RNG
    consumption => deterministic in the noise sense).
    """
    import importlib
    import random as _random

    from cfm_mol.native.strain_head import ResidualStrainEnergyHead
    from cfm_mol.native import corrector_in_loop as cil

    positions, atom_types, charges, node_batch_idx, B = _batch()
    head = ResidualStrainEnergyHead(
        n_atom_types=83, hidden_dim=64, n_rbf=16, cutoff=5.0, n_layers=2,
        baseline_hidden_dim=64,
    )

    # Record every randn_like call inside the corrector module.
    randn_calls = {"n": 0, "max_abs": 0.0}
    _orig_randn_like = torch.randn_like

    def _spy_randn_like(*args, **kwargs):
        out = _orig_randn_like(*args, **kwargs)
        randn_calls["n"] += 1
        randn_calls["max_abs"] = max(randn_calls["max_abs"], float(out.abs().max().item()))
        return out

    # The corrector calls `torch.randn_like` via the `torch` reference in
    # its module namespace -- patch that reference.
    cil.torch.randn_like = _spy_randn_like
    try:
        # Try the spec keyword first; if the code agent renamed it, fall
        # back to discovered aliases.
        # Probe candidate deterministic-mode keywords introduced by the
        # temperature refactor. `beta` is included as a positional-style
        # kwarg because the legacy signature requires it; it is harmless
        # under deterministic mode (noise term is gated off).
        kw_attempts = [
            {"corrector_noise_mode": "deterministic", "beta": 1.0 / 0.02569},
            {"noise_mode": "deterministic", "beta": 1.0 / 0.02569},
            {"stochastic": False, "beta": 1.0 / 0.02569},
        ]
        last_err = None
        loss_a = None
        sig = None
        # Pin RNG before each call so that *if* the corrector were noisy,
        # the two losses would still match -- but we additionally check
        # randn_calls["n"] == 0, which is the strong condition.
        _random.seed(0)
        torch.manual_seed(123)
        for kw in kw_attempts:
            try:
                randn_calls["n"] = 0
                randn_calls["max_abs"] = 0.0
                loss_a, diag_a = cil.corrector_in_loop_loss(
                    head, positions, atom_types, charges, node_batch_idx, B,
                    j_max=3, eta_init=1e-3, eta_final=1e-4, delta_max=0.25,
                    **kw,
                )
                sig = kw
                break
            except TypeError as e:
                last_err = e
                continue
        assert loss_a is not None, (
            f"corrector_in_loop_loss did not accept any deterministic "
            f"keyword (tried {kw_attempts}); last error: {last_err}"
        )

        # Strong assertion: deterministic mode must not have drawn any
        # Gaussian samples during the J unrolled steps.
        assert randn_calls["n"] == 0, (
            f"deterministic corrector consumed {randn_calls['n']} randn_like "
            f"draws (max|eps|={randn_calls['max_abs']:.3g}); noise term not zero"
        )

        # Reproducibility: same RNG state -> bit-identical loss (since no
        # randomness is consumed in the noise path; only the python random
        # int J still gates the unroll length).
        _random.seed(0)
        torch.manual_seed(123)
        randn_calls["n"] = 0
        loss_b, _ = cil.corrector_in_loop_loss(
            head, positions, atom_types, charges, node_batch_idx, B,
            j_max=3, eta_init=1e-3, eta_final=1e-4, delta_max=0.25,
            **sig,
        )
        assert randn_calls["n"] == 0, "second deterministic call drew noise"
        assert torch.isclose(loss_a, loss_b, atol=0.0, rtol=0.0), (
            f"deterministic corrector not bit-reproducible: "
            f"{loss_a.item()} vs {loss_b.item()}"
        )
        print(
            f"[native] deterministic corrector OK: keyword={sig}, "
            f"loss={loss_a.item():.6f}, randn_calls=0, displacement_std=0"
        )
    finally:
        cil.torch.randn_like = _orig_randn_like


def test_drift_beta_scaling():
    """SPEC item A: with alpha fixed and a fixed perturbation, doubling
    beta (i.e. halving drift_kT_eV) must exactly double the magnitude of
    the energy-drift contribution to the vector field.
    """
    from cfm_mol.native.strain_head import ResidualStrainEnergyHead
    from cfm_mol.native.energy_drift import (
        EnergyDriftConfig, compute_energy_drift,
    )
    torch.manual_seed(31)
    positions, atom_types, charges, node_batch_idx, B = _batch()
    head = ResidualStrainEnergyHead(
        n_atom_types=83, hidden_dim=64, n_rbf=16, cutoff=5.0, n_layers=2,
        baseline_hidden_dim=64,
    )
    # Give the strain readout a small but non-zero perturbation so the
    # force is finite (zero-init readout produces zero force). The force
    # is grad_r DeltaU, which only depends on strain_readout (not on the
    # composition baseline), so we must perturb strain_readout[-1].
    with torch.no_grad():
        sro = head.strain_readout
        last_lin = sro[-1]
        last_lin.weight.normal_(mean=0.0, std=1e-2)
        if last_lin.bias is not None:
            last_lin.bias.zero_()

    # Build two configs that differ ONLY in beta (factor 2). Disable
    # normalize_force so the magnitude is linear in beta.
    def _mk_cfg(beta_val: float) -> EnergyDriftConfig:
        kw = dict(
            enabled=True, alpha_max=0.2, t_on=0.65, power=2.0,
            beta=beta_val, force_clip=1.0e9, normalize_force=False,
            detach_force=True,
        )
        # Honour the new drift_kT_eV field if the dataclass exposes it.
        try:
            return EnergyDriftConfig(drift_kT_eV=1.0 / beta_val, **kw)
        except TypeError:
            return EnergyDriftConfig(**kw)

    cfg1 = _mk_cfg(20.0)
    cfg2 = _mk_cfg(40.0)  # 2x beta of cfg1

    t = torch.full((B,), 0.95)
    drift1, _ = compute_energy_drift(
        head, positions, atom_types, charges, node_batch_idx, B,
        t=t, cfg=cfg1, create_graph=False,
    )
    drift2, _ = compute_energy_drift(
        head, positions, atom_types, charges, node_batch_idx, B,
        t=t, cfg=cfg2, create_graph=False,
    )
    n1 = drift1.norm(dim=-1).mean().item()
    n2 = drift2.norm(dim=-1).mean().item()
    assert n1 > 0.0, f"drift1 magnitude is zero: {n1}"
    ratio = n2 / max(n1, 1e-20)
    assert abs(ratio - 2.0) < 1e-3, (
        f"drift should scale linearly with beta: expected ratio ~2.0, "
        f"got {ratio:.6f} (|drift1|={n1:.4g}, |drift2|={n2:.4g})"
    )
    # Per-element check: drift2 ~ 2 * drift1 elementwise.
    assert torch.allclose(drift2, 2.0 * drift1, atol=1e-5, rtol=1e-4), (
        "drift2 not elementwise equal to 2 * drift1"
    )
    print(f"[native] drift beta scaling OK: |d1|={n1:.4g}, |d2|={n2:.4g}, "
          f"ratio={ratio:.6f} (~2.0)")


def test_native_config_temperature_keys():
    """SPEC item config: v1 YAML carries the new temperature block; the
    four ablation configs exist under configs/native/ablations/ and load.
    """
    import yaml
    cfg_path = Path(
        "/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/configs/native/"
        "omol25_4m_bgfm_native_v1.yaml"
    )
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    temp = cfg["mol_fm"]["bgfm"]["native"]["temperature"]
    assert abs(float(temp["physical_kT_eV"]) - 0.02569) < 1e-9, (
        f"physical_kT_eV must be exactly 0.02569 (room-T anchor), "
        f"got {temp['physical_kT_eV']}"
    )
    # Each of the four ablation configs exists and loads.
    ablation_dir = Path(
        "/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/configs/native/ablations"
    )
    expected_tags = ("0p02569", "0p05", "0p10", "0p25")
    for tag in expected_tags:
        p = ablation_dir / f"omol25_4m_bgfm_native_T_{tag}.yaml"
        assert p.exists(), f"missing ablation config: {p}"
        with open(p) as f:
            ablation_cfg = yaml.safe_load(f)
        assert ablation_cfg is not None and isinstance(ablation_cfg, dict), (
            f"ablation config did not parse to a dict: {p}"
        )
        # It must also carry the temperature block.
        a_temp = ablation_cfg["mol_fm"]["bgfm"]["native"]["temperature"]
        assert "bridge_kT_final_eV" in a_temp and "drift_kT_eV" in a_temp, (
            f"ablation {p} missing required temperature keys"
        )
    print(
        f"[native] config temperature block OK: physical_kT={temp['physical_kT_eV']} eV; "
        f"4 ablations present and parse"
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    test_residual_strain_head_forward_and_force()
    test_residual_head_loss()
    test_boltzmann_bridge_loss()
    test_compute_energy_drift_se3()
    test_corrector_in_loop_loss()
    test_deterministic_corrector()
    test_drift_beta_scaling()
    test_native_config_temperature_keys()
    print("\n[native] ALL TESTS PASSED")
