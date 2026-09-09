"""Energy labels live on separate shards; force labels are not a prerequisite."""
from types import SimpleNamespace
import dgl
import torch
import pytest

from cfm_mol.bgfm_train_hook import patch_flowmol_bgfm


@pytest.mark.parametrize("has_forces", [False, True])
@pytest.mark.parametrize("nonfinite_density", [False, True])
def test_energy_only_hook_reaches_energy_step_without_force_labels(monkeypatch, nonfinite_density, has_forces):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.n_atom_types = 3
            self.vector_field = torch.nn.Identity()
            self.trainer = SimpleNamespace(estimated_stepping_batches=10)
            self.global_step = 10
            self.logged = {}
        def training_step(self, g, batch_idx):
            return self.weight.square()
        def log(self, name, value, **kwargs):
            self.logged[name] = value
    g = dgl.batch([dgl.graph(([0, 1], [1, 0]), num_nodes=2)])
    if has_forces:
        g.ndata['force_1_true'] = torch.zeros(2,3)
    class Loader:
        M, K = 1, 2
        def __init__(self, **kwargs):
            pass
        def next_batch(self):
            return g, torch.tensor([1., 2.]), torch.tensor([0, 0]), torch.tensor([0, 0]), torch.tensor([True, False])
    observed = []
    def energy(model, *args, **kwargs):
        observed.append(kwargs['density_options'])
        if nonfinite_density:
            raise FloatingPointError('non-finite density')
        return 2*model.weight.square(), {'logp_mean': 0., 'residual_within_std': 1., 'n_groups_used': 1}
    monkeypatch.setattr('cfm_mol.perturbation_loader.PerturbationLoader', Loader)
    monkeypatch.setattr('cfm_mol.bgfm_density.energy_consistency_loss_per_mol', energy)
    model = Model()
    patch_flowmol_bgfm(model, {'enabled': True, 'lambda_1': 0., 'lambda_2': 0.5, 'force_diagnostics': False,
        'energy_perturbation_shards': ['unused.pt'],
        'energy_density_options': {'mode': 'clamped_cnf', 'terminal_time': 0.95},
        'warmup_frac': 0., 'ramp_frac': 0.})
    loss = model.training_step(g, 0)
    assert observed == [{'mode': 'clamped_cnf', 'terminal_time': 0.95}]
    assert loss.item() == (1. if nonfinite_density else 2.)
    loss.backward()
    assert model.weight.grad.item() == (2. if nonfinite_density else 4.)
    if nonfinite_density:
        assert model.logged['train_density_nonfinite_skip'] == 1.
        assert model.logged['train_L_energy_nan_skip'] == 1.
