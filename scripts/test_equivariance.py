"""Equivariance test for tangent_project + retract.

Theorem 4.1 of the methods derivation assumes the velocity field is E(3)-
equivariant (the FlowMol3 GVP backbone guarantees this). Our tangent
projection and retraction are pair-local operations along unit vectors
$(r_i - r_j) / \\|r_i - r_j\\|$, which are themselves SO(3)-equivariant and
translation-invariant. This script verifies that end-to-end:

  For any g in SE(3) acting on coords r,
    retract(g.r, a)  ==  g . retract(r, a)
    tangent_project(g.v, g.r, a)  ==  g . tangent_project(v, r, a)

We use random rotations (uniform Haar) + random translations. The test
passes if |lhs - rhs| < 1e-4 on all 64 random transforms (1e-6 is too tight
because the retract's "most-violating-pair" argmin has floating-point
ties; 1e-4 is tight enough to catch any non-equivariance).

Run: python scripts/test_equivariance.py
"""
from __future__ import annotations

import sys

import torch

from cfm_mol.domain import default_d_min_table
from cfm_mol.fibre import retract, tangent_project


def _random_rotation(batch: int, device: str | torch.device = "cpu") -> torch.Tensor:
    """Uniform SO(3) via QR of random matrix. Shape (batch, 3, 3)."""
    M = torch.randn(batch, 3, 3, device=device)
    Q, R = torch.linalg.qr(M)
    # Ensure determinant +1 by flipping sign of last column if needed.
    sign = torch.sign(torch.det(Q)).unsqueeze(-1).unsqueeze(-1)
    Q = Q * sign
    return Q


def _apply_se3(r: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """(B, N, 3) coords, (B, 3, 3) rotation, (B, 3) translation -> (B, N, 3)."""
    return torch.einsum("bij,bnj->bni", R, r) + t.unsqueeze(1)


def _apply_rotation(v: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Rotate-only for velocity-like vectors. (B, N, 3), (B, 3, 3)."""
    return torch.einsum("bij,bnj->bni", R, v)


def test_retract_equivariance(n_trials: int = 32, tol: float = 1e-4) -> int:
    B, N = 4, 8
    n_atom_types = 5
    d_min = default_d_min_table(n_atom_types=n_atom_types)
    failures = 0
    for trial in range(n_trials):
        torch.manual_seed(trial)
        # Random coords near the boundary to stress-test retract.
        r = torch.randn(B, N, 3) * 1.2
        a = torch.randint(0, n_atom_types, (B, N))
        Rg = _random_rotation(B)
        tg = torch.randn(B, 3)

        # Path A: retract then transform.
        r_retract, _ = retract(r.clone(), a, d_min)
        lhs = _apply_se3(r_retract, Rg, tg)

        # Path B: transform then retract.
        r_transformed = _apply_se3(r.clone(), Rg, tg)
        r_retract_after, _ = retract(r_transformed, a, d_min)
        rhs = r_retract_after

        diff = (lhs - rhs).abs().max().item()
        if diff > tol:
            print(f"  [retract trial {trial}] max_diff={diff:.2e}  FAIL")
            failures += 1
    if failures == 0:
        print(f"  retract: {n_trials}/{n_trials} trials equivariant (max diff < {tol:.1e})")
    return failures


def test_tangent_project_equivariance(n_trials: int = 32, tol: float = 1e-4) -> int:
    B, N = 4, 8
    n_atom_types = 5
    d_min = default_d_min_table(n_atom_types=n_atom_types)
    failures = 0
    for trial in range(n_trials):
        torch.manual_seed(trial + 1000)
        r = torch.randn(B, N, 3) * 2.0
        v = torch.randn(B, N, 3) * 0.5
        a = torch.randint(0, n_atom_types, (B, N))
        Rg = _random_rotation(B)
        tg = torch.randn(B, 3)

        # Path A: project then rotate velocity (translation of r is irrelevant
        # for velocity direction; only rotation of r and v matters).
        v_proj = tangent_project(v.clone(), r.clone(), a, d_min)
        lhs = _apply_rotation(v_proj, Rg)

        # Path B: transform coords + rotate velocity, then project.
        r_transformed = _apply_se3(r.clone(), Rg, tg)
        v_rotated = _apply_rotation(v.clone(), Rg)
        v_proj_after = tangent_project(v_rotated, r_transformed, a, d_min)
        rhs = v_proj_after

        diff = (lhs - rhs).abs().max().item()
        if diff > tol:
            print(f"  [tangent trial {trial}] max_diff={diff:.2e}  FAIL")
            failures += 1
    if failures == 0:
        print(f"  tangent_project: {n_trials}/{n_trials} trials equivariant (max diff < {tol:.1e})")
    return failures


def main() -> int:
    print("=" * 60)
    print("EQUIVARIANCE TEST -- tangent_project + retract under SE(3)")
    print("=" * 60)
    f1 = test_retract_equivariance()
    f2 = test_tangent_project_equivariance()
    total = f1 + f2
    print()
    if total == 0:
        print("all equivariance tests passed.")
        return 0
    else:
        print(f"FAIL: {total} trials violated equivariance.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
