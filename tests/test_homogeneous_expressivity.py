"""The adapter must stay configuration-dependent on single-element systems.

Before the split-group repair, an `internal` layer replaced every active atom by
its own centroid.  On a homogeneous system every atom is active, so the context
collapsed onto the origin, every direction vector was exactly zero, every
sigmoid feature vanished, and a 19k-parameter conditional flow degenerated into
a fixed two-parameter isotropic radial map with no learned directional context.
That is exactly the LJ/DW homonuclear-cluster family the field benchmarks on.
"""
import torch

from cfm_mol.species_coupling_adapter import SpeciesCouplingAdapter


def _centered(batch, atoms, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, atoms, 3, generator=g, dtype=torch.float64)
    return x - x.mean(1, keepdim=True)


def _adapter(numbers, **kw):
    a = SpeciesCouplingAdapter(numbers, charge=0, spin_multiplicity=1, kT=0.025851999786435, split_groups=True, **kw)
    return a.double()


def test_homogeneous_context_is_not_degenerate():
    """Some layer must see a nonzero direction vector on a single-element system."""
    a = _adapter([6] * 8)
    x = _centered(4, 8, 11)
    seen = []
    for layer in a.layers:
        context, origin, active, roles, sel_a, sel_b = a.split_context(x, layer)
        seen.append(float((context - origin).abs().max()))
    assert max(seen) > 1e-6, (
        f"every layer's directions collapsed to zero (max |context-origin| = {max(seen):.3e}); "
        "the conditional flow is vacuous on homogeneous systems")


def test_node_head_actually_drives_the_map():
    """The per-atom conditioner output must influence the map.

    With collapsed directions the sigmoid features vanish identically, so
    `node_output[..., 0]` (weights) and `node_output[..., 1]` (offsets) have no
    effect whatsoever and only the two group scalars survive.  Perturbing the
    node head must therefore change the output of a repaired adapter.
    """
    a = _adapter([6] * 8)
    x = _centered(2, 8, 5)
    with torch.no_grad():
        for p in a.conditioner.group_head.parameters():
            p.normal_(0, 0.3)
    before, vol_before = a(x)
    with torch.no_grad():
        for p in a.conditioner.node_head.parameters():
            p.normal_(0, 0.5)
    after, vol_after = a(x)
    moved = float((after - before).abs().max())
    assert moved > 1e-8, (
        f"perturbing the per-atom conditioner changed the output by {moved:.3e}; "
        "the node head is structurally dead on a homogeneous system")


def test_repair_preserves_com_and_invertibility():
    a = _adapter([6] * 8)
    with torch.no_grad():
        for head in (a.conditioner.node_head, a.conditioner.group_head):
            for p in head.parameters():
                p.normal_(0, 0.3)
    x = _centered(3, 8, 17)
    y, vol = a(x)
    assert float(y.mean(1).abs().max()) < 1e-9, "COM-free subspace not preserved"
    back, inv_vol, diag = a.inverse(y)
    assert float((back - x).abs().max()) < 1e-7, "inverse does not reconstruct"
    assert float((vol + inv_vol).abs().max()) < 1e-7, "log-volumes do not cancel"


def test_heteronuclear_still_works():
    a = _adapter([1, 6, 6, 8])
    x = _centered(2, 4, 3)
    y, vol = a(x)
    assert torch.isfinite(y).all() and torch.isfinite(vol).all()
    assert float(y.mean(1).abs().max()) < 1e-9
