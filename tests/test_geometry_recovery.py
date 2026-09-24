import torch
from cfm_mol.geometry_recovery import local_geometry_errors, recovery_noise


def example():
    x = torch.tensor([[[0., 0., 0.], [1.4, 0., 0.], [0., 1.4, 0.],
                       [0., 0., 1.1], [-1., -.3, 0.]]], dtype=torch.float64)
    return x, torch.tensor([[6, 6, 8, 1, 1]])


def test_geometry_objective_preserves_three_dimensional_reference():
    x, z = example()
    pair, angle = local_geometry_errors(x, x, z)
    assert pair == 0 and angle == 0
    distorted = x.clone()
    distorted[0, 2] += torch.tensor([.25, -.25, .35])
    distorted.requires_grad_()
    pair, angle = local_geometry_errors(distorted, x, z)
    loss = pair + angle
    assert pair > 0 and angle > 0
    loss.backward()
    assert torch.isfinite(distorted.grad).all()
    improved = distorted.detach() - .01 * distorted.grad
    after = sum(local_geometry_errors(improved, x, z))
    assert after < loss


def test_local_geometry_invariance_and_nonradial_signal():
    x, z = example()
    y = x.clone(); y[0, 2, 2] += .2
    q, _ = torch.linalg.qr(torch.tensor([[1., 2., 3.], [2., -1., 1.],
                                        [1., 0., -2.]], dtype=x.dtype))
    permutation = torch.tensor([2, 0, 4, 1, 3])
    expected = local_geometry_errors(y, x, z)
    rotated = local_geometry_errors(y @ q + 3, x @ q - 2, z)
    permuted = local_geometry_errors(y[:, permutation], x[:, permutation], z[:, permutation])
    for a, b, c in zip(expected, rotated, permuted):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-10)
        torch.testing.assert_close(a, c, atol=1e-12, rtol=1e-10)


def test_corruptions_reproducible_centered_and_nonzero():
    x, _ = example(); x = x.expand(24, -1, -1)
    first = recovery_noise(x, torch.Generator().manual_seed(42))
    second = recovery_noise(x, torch.Generator().manual_seed(42))
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert torch.isfinite(first).all() and first.square().sum() > 0
    torch.testing.assert_close(first.mean(1), torch.zeros_like(first.mean(1)), atol=1e-12, rtol=0)


def test_recovery_trains_real_two_pass_backbone_and_restores_context():
    import json
    from pathlib import Path
    from cfm_mol import matched_egnn as base, connectivity_feedback as feedback
    from cfm_mol.geometry_recovery import recovery_loss
    torch.set_num_threads(1)
    spec = json.loads(Path('research/evidence/gaga_feedback_distance_s0_v1.json').read_text())
    model = base.initialize(spec, 'cpu').train()
    feedback.install(model)
    source = base.HarmonicSource(spec['edge_log_width'])
    context = feedback.GeometryContext(source, 'distance')
    x, z = example(); x = x.float()
    calls = []
    handle = model.dynamics.egnn.register_forward_hook(lambda *_: calls.append(1))
    losses = []
    for corrupt, local in [(False, False), (True, False), (True, True)]:
        model.zero_grad(set_to_none=True)
        objective, diagnostics = recovery_loss(model, x, z, spec, source, context,
            74101, corrupt=corrupt, local_geometry=local)
        assert torch.isfinite(objective)
        objective.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients) > 0
        losses.append(float(objective.detach()))
        if local:
            assert diagnostics['pair'] > 0 and diagnostics['angle'] > 0
    handle.remove()
    assert len(calls) == 6
    assert losses[0] != losses[1] and losses[2] > losses[1]
    assert all(model.dynamics.egnn._modules[f'e_block_{i}']._geometry_context is None
               for i in range(model.dynamics.egnn.n_layers))
