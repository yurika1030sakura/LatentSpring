"""endpoint -> velocity conversion used by force_endpoint_to_velocity.

Under the ctmc parameterization the vector_field returns an ENDPOINT
prediction x_1_hat, but score_from_fm_velocity documents itself as taking a
VELOCITY.  The training hook converts with

    v = (x_1_hat - x_t) / (1 - t)

These tests pin both the broadcasting (the conversion is applied to (N, 3)
coordinates with per-atom (N,) times) and the resulting score identity.
"""
import torch
from cfm_mol.bgfm_loss import score_from_fm_velocity


def _convert(x1_hat, x_t, t_atom):
    """Exactly the expression used in bgfm_train_hook.py."""
    return (x1_hat - x_t) / (1.0 - t_atom).clamp_min(1e-4).unsqueeze(-1)


def test_conversion_broadcasts_over_atoms():
    N = 615                      # the shape that crashed job 44402032
    x1_hat = torch.randn(N, 3)
    x_t = torch.randn(N, 3)
    t_atom = torch.full((N,), 0.85)
    v = _convert(x1_hat, x_t, t_atom)
    assert v.shape == (N, 3)
    # per-atom time must scale the matching atom, not mix atoms
    t2 = torch.rand(N) * 0.9
    v2 = _convert(x1_hat, x_t, t2)
    for i in (0, 17, N - 1):
        expected = (x1_hat[i] - x_t[i]) / (1.0 - t2[i])
        assert torch.allclose(v2[i], expected, atol=1e-5)


def test_score_matches_endpoint_formula():
    """Converting then reading out must equal the endpoint score

        s = (t * x_1_hat - x_t) / ((1 - t)^2 * sigma^2)

    which is the marginal score with E[x_1 | x_t] = x_1_hat.
    """
    N, sigma = 64, 1.0
    x1_hat = torch.randn(N, 3)
    x_t = torch.randn(N, 3) * 0.5
    for tv in (0.85, 0.92, 0.97):
        t_atom = torch.full((N,), tv)
        s = score_from_fm_velocity(_convert(x1_hat, x_t, t_atom), x_t, t_atom,
                                   prior_std=sigma)
        expected = (tv * x1_hat - x_t) / ((1.0 - tv) ** 2 * sigma ** 2)
        # readout caps per-atom norm at 1000; compare only uncapped atoms
        keep = expected.norm(dim=-1) < 1000.0
        assert keep.any()
        assert torch.allclose(s[keep], expected[keep], rtol=1e-4, atol=1e-3)


def test_conversion_is_inverse_of_docstring_relation():
    """score_from_fm_velocity documents E[x_1|x] = x_t + (1-t) v."""
    N = 32
    x1_hat = torch.randn(N, 3)
    x_t = torch.randn(N, 3)
    t_atom = torch.full((N,), 0.9)
    v = _convert(x1_hat, x_t, t_atom)
    recovered = x_t + (1.0 - t_atom).unsqueeze(-1) * v
    assert torch.allclose(recovered, x1_hat, atol=1e-5)
