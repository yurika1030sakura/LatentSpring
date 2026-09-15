import torch
from cfm_mol.endpoint_connectivity import endpoint_support_loss


def test_global_penalty_detects_two_locally_connected_fragments():
    x = torch.tensor([[0., 0., 0.], [1., 0., 0.], [5., .1, 0.], [6., .1, 0.]], dtype=torch.double)
    radii = torch.full((4,), .5, dtype=torch.double)
    assert endpoint_support_loss(x, radii, 'local') == 0
    assert endpoint_support_loss(x, radii, 'tree') > 0
    connected = torch.tensor([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [3., 0., 0.]], dtype=torch.double)
    assert endpoint_support_loss(connected, radii, 'tree') == 0


def test_gradient_symmetry_and_descent_away_from_ties():
    x = torch.tensor([[0., .1, .2], [1., -.1, 0.], [5., .3, .1], [6., .7, .4]], dtype=torch.double, requires_grad=True)
    radii = torch.tensor([.5, .6, .7, .8], dtype=torch.double)
    assert torch.autograd.gradcheck(lambda y: endpoint_support_loss(y, radii), (x,), atol=1e-5)
    loss = endpoint_support_loss(x, radii)
    grad = torch.autograd.grad(loss, x)[0]
    torch.testing.assert_close(grad.sum(0), torch.zeros(3, dtype=torch.double), atol=1e-12, rtol=0)
    assert endpoint_support_loss(x - .01 * grad, radii) < loss
    q, _ = torch.linalg.qr(torch.tensor([[1., 2., 3.], [2., -1., 1.], [1., 1., -1.]], dtype=torch.double))
    p = torch.tensor([3, 1, 0, 2])
    torch.testing.assert_close(endpoint_support_loss((x @ q + 17)[p], radii[p]), loss)


def test_overlap_remains_penalized_in_both_modes():
    x = torch.tensor([[0., 0., 0.], [.2, 0., 0.]], dtype=torch.double)
    for mode in ['tree', 'local']:
        assert endpoint_support_loss(x, torch.ones(2, dtype=torch.double), mode) > 0
