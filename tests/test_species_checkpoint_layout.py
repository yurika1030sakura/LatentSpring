"""Checkpoint layout must round-trip, including the historical split metadata."""
import pytest
import torch

from cfm_mol.entropy_adapter_io import build_species_adapter
from cfm_mol.nonequilibrium import centered_orthonormal_basis
from cfm_mol.species_coupling_adapter import SpeciesCouplingAdapter


@pytest.mark.parametrize('split', [False, True])
@pytest.mark.parametrize('affine', [False, True])
def test_layout_roundtrip_and_full_intrinsic_jacobian(split, affine):
    torch.manual_seed(771)
    config = dict(charge=0, spin_multiplicity=1, kT=1., hidden=8, radial=8,
                  sweeps=2, split_groups=split)
    model = build_species_adapter([6]*4, config, affine=affine).double()
    with torch.no_grad():
        for head in [model.conditioner.node_head, model.conditioner.group_head]:
            head.weight.normal_(std=.15); head.bias.normal_(std=.1)
    restored = build_species_adapter([6]*4, model.configuration, affine=affine).double()
    restored.load_state_dict(model.state_dict())
    basis = centered_orthonormal_basis(4)
    z = torch.randn(9, dtype=torch.float64)
    x = (basis@z.reshape(3, 3))[None]
    y, volume = model(x)
    replay, replay_volume = restored(x)
    torch.testing.assert_close(replay, y, atol=0, rtol=0)
    torch.testing.assert_close(replay_volume, volume, atol=0, rtol=0)
    def transform(v):
        positions, _ = model((basis@v.reshape(3, 3))[None])
        return (basis.T@positions[0]).flatten()
    sign, reference = torch.linalg.slogdet(torch.autograd.functional.jacobian(transform, z))
    assert sign == 1
    torch.testing.assert_close(volume[0], reference, atol=1e-9, rtol=1e-9)
    back, inverse_volume, _ = restored.inverse(y)
    torch.testing.assert_close(back, x, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(inverse_volume, -volume, atol=1e-9, rtol=1e-9)
    if split:
        # This is a labelled diagnostic, not a repair of permutation equivariance.
        assert model.configuration['permutation_equivariant'] is False
        torch.testing.assert_close(y[:, :2].mean(1), x[:, :2].mean(1), atol=1e-12, rtol=0)
        permutation = torch.tensor([2, 1, 0, 3])
        changed, _ = model(x[:, permutation])
        assert float((changed-y[:, permutation]).abs().max()) > 1e-5


def test_old_layout_metadata_is_inferred_and_corruption_rejected():
    model = SpeciesCouplingAdapter([6]*8, charge=0, spin_multiplicity=1, kT=1., split_groups=True)
    historical = {k: v for k, v in model.configuration.items() if k not in ['split_groups', 'permutation_equivariant']}
    assert build_species_adapter([6]*8, historical).layers == model.layers
    with pytest.raises(ValueError, match='derived layout differs'):
        build_species_adapter([6]*8, historical | {'internal_blocks': [[0, 2, 4, 6], [1, 3, 5, 7]]})
    with pytest.raises(ValueError, match='Unknown'):
        build_species_adapter([6]*8, historical | {'invented_option': True})
    legacy = {k: v for k, v in model.configuration.items()
              if k not in ['split_groups', 'permutation_equivariant', 'minimum_active', 'internal_blocks']}
    unsplit = build_species_adapter([6]*8, legacy)
    assert unsplit.configuration['split_groups'] is False
    assert unsplit.configuration['permutation_equivariant'] is True
