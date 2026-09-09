"""Density identities, true parameter gradients, and the real FlowMol3 API."""
import math
from types import SimpleNamespace

import dgl
import pytest
import torch
from torch import nn

from cfm_mol.clamped_density import (
    center_by_graph, deterministic_field, log_density_clamped_flow, position_velocity,
    sample_clamped_flow,
)


def graph_batch(n_atoms=(3, 4), dtype=torch.float64):
    generator = torch.Generator().manual_seed(814)
    graphs = []
    for n in n_atoms:
        src, dst = torch.where(~torch.eye(n, dtype=torch.bool))
        g = dgl.graph((src, dst), num_nodes=n)
        x = torch.randn(n, 3, generator=generator, dtype=dtype)
        g.ndata['x_1_true'] = x-x.mean(0)
        g.ndata['x_t'] = x.clone()
        for key, width in [('a', 4), ('c', 7)]:
            g.ndata[f'{key}_1_true'] = torch.nn.functional.one_hot(torch.zeros(n, dtype=torch.long), width).to(dtype)
            g.ndata[f'{key}_t'] = torch.zeros(n, width, dtype=dtype)
        g.edata['e_1_true'] = torch.nn.functional.one_hot(torch.zeros(g.num_edges(), dtype=torch.long), 5).to(dtype)
        g.edata['e_t'] = torch.zeros_like(g.edata['e_1_true'])
        graphs.append(g)
    g = dgl.batch(graphs)
    nbi = torch.repeat_interleave(torch.arange(len(n_atoms)), torch.tensor(n_atoms))
    src, dst = g.edges()
    return g, nbi, src < dst


class Schedule:
    feats = ['a', 'x', 'c', 'e']  # position need not be first
    def __init__(self, power=1):
        self.power = power
    def alpha_t(self, t):
        return (t**self.power)[:, None].expand(-1, 4)
    def alpha_t_prime(self, t):
        return (self.power*t**(self.power-1))[:, None].expand(-1, 4)


class LinearHead(nn.Module):
    def __init__(self, a=0.3, endpoint=True, power=1):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(a, dtype=torch.float64))
        self.endpoint = endpoint
        self.interpolant_scheduler = Schedule(power)
        self.dropout = nn.Dropout(0.8)
    def forward(self, g, t, node_batch_idx, **kwargs):
        x = g.ndata['x_t']
        v = self.a*self.dropout(x)
        if self.endpoint:
            s = self.interpolant_scheduler
            ratio = (1-s.alpha_t(t)[:, 1])/s.alpha_t_prime(t)[:, 1]
            return {'x': x+ratio[node_batch_idx, None]*v}
        return {'x': v}


@pytest.mark.parametrize('power', [1, 2])
def test_endpoint_head_converts_to_scheduled_velocity(power):
    g, nbi, uem = graph_batch()
    model = SimpleNamespace(vector_field=LinearHead(power=power))
    x = g.ndata['x_1_true']+torch.tensor([3., 5., -1.])
    with deterministic_field(model.vector_field):
        v = position_velocity(model, g, x, torch.tensor([0.4, 0.8]), nbi, uem)
    assert torch.allclose(v, 0.3*center_by_graph(x, nbi, 2), atol=1e-12)


def test_com_trace_normalization_and_second_order_convergence():
    g, nbi, uem = graph_batch()
    model = SimpleNamespace(vector_field=LinearHead())
    x = g.ndata['x_1_true']
    T, a = 0.95, 0.3
    counts = g.batch_num_nodes().to(x)
    sq = x.new_zeros(2).index_add(0, nbi, x.square().sum(-1))
    exact = -0.5*sq*math.exp(-2*a*T)-1.5*(counts-1)*math.log(2*math.pi)-3*(counts-1)*a*T
    errors = []
    for steps in (8, 16, 32):
        actual = log_density_clamped_flow(model, g, nbi, uem, n_ode_steps=steps, n_hutchinson=0)
        errors.append((actual-exact).abs().max().item())
    assert errors[0]/errors[1] > 3.8
    assert errors[1]/errors[2] > 3.8
    assert errors[-1] < 1e-4


def test_full_discretized_gradient_matches_finite_difference():
    g, nbi, uem = graph_batch()
    head = LinearHead()
    model = SimpleNamespace(vector_field=head)
    def objective():
        p = log_density_clamped_flow(model, g, nbi, uem, n_ode_steps=4,
                                    n_hutchinson=0, for_training=True)
        return ((p-p.mean())**2).mean()
    derivative = torch.autograd.grad(objective(), head.a)[0].item()
    initial, eps = head.a.item(), 1e-5
    with torch.no_grad():
        head.a.fill_(initial+eps)
    upper = objective().item()
    with torch.no_grad():
        head.a.fill_(initial-eps)
    lower = objective().item()
    with torch.no_grad():
        head.a.fill_(initial)
    assert derivative == pytest.approx((upper-lower)/(2*eps), rel=1e-6, abs=1e-7)


def test_prior_path_retains_translation_parameter_gradient():
    # A COM-free shift has zero divergence; only the prior/trajectory gradient
    # can train it. The archived frozen-trajectory estimator cannot do so.
    class Shift(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Parameter(torch.tensor(0.2, dtype=torch.float64))
        def forward(self, g, t, **kwargs):
            direction = torch.arange(g.num_nodes(), dtype=torch.float64)[:, None].expand(-1, 3)
            return {'x': self.a*direction}
    g, nbi, uem = graph_batch()
    model = SimpleNamespace(vector_field=Shift())
    def objective():
        return log_density_clamped_flow(model, g, nbi, uem, parameterization='velocity',
            n_ode_steps=2, n_hutchinson=0, for_training=True).sum()
    grad = torch.autograd.grad(objective(), model.vector_field.a)[0].item()
    eps = 1e-5
    with torch.no_grad():
        model.vector_field.a.add_(eps)
    plus = objective().item()
    with torch.no_grad():
        model.vector_field.a.sub_(2*eps)
    minus = objective().item()
    assert abs(grad) > 1e-3
    assert grad == pytest.approx((plus-minus)/(2*eps), rel=1e-6)


def test_constant_field_and_graph_model_restoration():
    class Zero(nn.Module):
        def forward(self, g, t, **kwargs):
            return {'x': torch.zeros_like(g.ndata['x_t'])}
    g, nbi, uem = graph_batch()
    old = {key: value.clone() for key, value in g.ndata.items()}
    model = SimpleNamespace(vector_field=Zero())
    value = log_density_clamped_flow(model, g, nbi, uem, parameterization='velocity', n_ode_steps=2)
    assert torch.isfinite(value).all()
    assert model.vector_field.training
    for key in old:
        assert torch.equal(g.ndata[key], old[key])
    head = LinearHead()
    head.dropout.eval()  # preserve mixed module modes too
    with deterministic_field(head):
        assert not head.training
    assert head.training and not head.dropout.training


def test_real_ctmc_head_is_endpoint_and_adapter_uses_velocity():
    from flowmol.models.ctmc_vector_field import CTMCVectorField
    from flowmol.models.interpolant_scheduler import InterpolantScheduler
    torch.manual_seed(7)
    scheduler = InterpolantScheduler(['x', 'a', 'c', 'e'], schedule_type='linear')
    vf = CTMCVectorField(n_atom_types=3, canonical_feat_order=['x', 'a', 'c', 'e'],
        interpolant_scheduler=scheduler, n_vec_channels=4, n_hidden_scalars=8,
        n_hidden_edge_feats=8, n_molecule_updates=1, convs_per_update=2,
        n_message_gvps=1, n_update_gvps=1, n_expansion_gvps=1, rbf_dim=4)
    model = SimpleNamespace(vector_field=vf.eval())
    g, nbi, uem = graph_batch((3,), dtype=torch.float32)
    for key in ['a', 'c']:
        g.ndata[f'{key}_t'] = g.ndata[f'{key}_1_true']
    g.edata['e_t'] = g.edata['e_1_true']
    x, t = g.ndata['x_1_true'], torch.tensor([0.4])
    g.ndata['x_t'] = x
    endpoint = vf(g, t, node_batch_idx=nbi, upper_edge_mask=uem)['x']
    expected = center_by_graph((endpoint-x)/(1-t), nbi, 1)
    actual = position_velocity(model, g, x, t, nbi, uem)
    assert torch.allclose(actual, expected, atol=2e-6)
    assert not torch.allclose(actual, endpoint, atol=1e-3)
    logp = log_density_clamped_flow(model, g, nbi, uem, n_ode_steps=2, n_hutchinson=1, for_training=True)
    logp.sum().backward()
    grads = [p.grad for p in vf.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)


def test_trace_replicates_share_one_trajectory_and_use_distinct_probes():
    g, nbi, uem = graph_batch()
    class CountHead(LinearHead):
        def __init__(self):
            super().__init__()
            self.calls = 0
        def forward(self, *args, **kwargs):
            self.calls += 1
            return super().forward(*args, **kwargs)
    head = CountHead()
    model = SimpleNamespace(vector_field=head)
    seen = []
    def probes(step, sample, x):
        seen.append((step, sample))
        return torch.ones_like(x) if sample == 0 else ((torch.arange(x.numel()).reshape(x.shape)%2)*2-1).to(x)
    q = log_density_clamped_flow(model, g, nbi, uem, n_ode_steps=3,
        n_hutchinson=1, n_trace_replicates=2, xi_fn=probes, for_training=True)
    assert q.shape == (2, 2)
    assert head.calls == 6  # two midpoint stages per step, shared by both traces
    assert seen == [(step, sample) for step in range(3) for sample in range(2)]
    assert not torch.allclose(q[0], q[1])
    assert torch.isfinite(torch.autograd.grad((q[0]*q[1]).mean(), head.a)[0])


def test_sampler_and_density_describe_same_analytic_flow():
    g, nbi, uem = graph_batch()
    model = SimpleNamespace(vector_field=LinearHead())
    x0 = g.ndata['x_1_true'].clone()
    exact = x0*math.exp(0.3*0.95)
    errors = []
    for steps in [8, 16, 32]:
        sample = sample_clamped_flow(model,g,nbi,uem,n_ode_steps=steps,x0=x0)
        errors.append(float((sample-exact).abs().max()))
    assert errors[0]/errors[1] > 3.8
    assert errors[1]/errors[2] > 3.8
    assert torch.equal(g.ndata['x_1_true'],x0)
    g.ndata['x_1_true'] = sample
    logq = log_density_clamped_flow(model,g,nbi,uem,n_ode_steps=32,n_hutchinson=0)
    counts = g.batch_num_nodes().to(x0)
    prior = -0.5*x0.new_zeros(2).index_add(0,nbi,x0.square().sum(-1))-1.5*(counts-1)*math.log(2*math.pi)
    assert torch.allclose(logq,prior-3*(counts-1)*0.3*0.95,atol=2e-5,rtol=0)


def test_common_probes_cancel_constant_jacobian_group_noise():
    from cfm_mol.replica_loss import make_grouped_probe_sampler, grouped_replica_residual
    g,nbi,uem = graph_batch((3,3))
    parents = torch.zeros(2,dtype=torch.long)
    sampler = make_grouped_probe_sampler(nbi,parents)
    probe = sampler(0,0,g.ndata['x_1_true'])
    assert torch.equal(probe[:3],probe[3:])
    model = SimpleNamespace(vector_field=LinearHead())
    q = log_density_clamped_flow(model,g,nbi,uem,n_ode_steps=3,n_hutchinson=1,
        n_trace_replicates=2,xi_fn=sampler)
    exact = log_density_clamped_flow(model,g,nbi,uem,n_ode_steps=3,n_hutchinson=0)
    energy = torch.tensor([0.3,-0.4])
    noisy_loss,_ = grouped_replica_residual(q,energy,parents)
    exact_loss,_ = grouped_replica_residual(exact,energy,parents)
    assert noisy_loss.item() == pytest.approx(exact_loss.item(),abs=1e-10)
    _,unequal,_ = graph_batch((3,4))
    with pytest.raises(ValueError,match='same atom count'):
        make_grouped_probe_sampler(unequal,parents)


@pytest.mark.parametrize('common',[False,True])
def test_energy_trace_rng_is_separate_from_fm_rng(common):
    from cfm_mol.bgfm_density import energy_consistency_loss_per_mol
    g,nbi,uem=graph_batch((3,3))
    model=SimpleNamespace(vector_field=LinearHead())
    torch.manual_seed(441)
    before=torch.get_rng_state().clone()
    options={'mode':'clamped_cnf','n_trace_replicates':2,'trace_seed':998,
             'common_trace_within_parent':common,'residual_estimator':'replica_product'}
    kwargs=dict(energies=torch.tensor([0.2,0.5]),parent_id=torch.zeros(2,dtype=torch.long),
                n_ode_steps=2,n_hutchinson=1,density_options=options)
    loss,_=energy_consistency_loss_per_mol(model,g,nbi,uem,**kwargs)
    assert torch.equal(before,torch.get_rng_state())
    repeat,_=energy_consistency_loss_per_mol(model,g,nbi,uem,**kwargs)
    assert loss.item()==repeat.item()


def test_adaptive_reference_matches_analytic_density_for_each_fixed_probe():
    from cfm_mol.clamped_reference import log_density_clamped_reference
    g,nbi,uem=graph_batch()
    x=g.ndata['x_1_true'].clone()
    head=LinearHead();model=SimpleNamespace(vector_field=head)
    result=log_density_clamped_reference(model,g,nbi,uem,rtol=1e-9,atol=1e-11,
        quadrature_orders=(2,4),n_replicates=3,seed=42)
    counts=g.batch_num_nodes().to(x)
    sq=x.new_zeros(2).index_add(0,nbi,x.square().sum(-1))
    prior=-0.5*sq*math.exp(-2*0.3*0.95)-1.5*(counts-1)*math.log(2*math.pi)
    generator=torch.Generator().manual_seed(42)
    expected=[]
    for _ in range(3):
        probe=(2*torch.randint(0,2,x.shape,generator=generator)-1).to(x)
        projected=center_by_graph(probe,nbi,2)
        trace=x.new_zeros(2).index_add(0,nbi,(probe*projected).sum(-1))*0.3
        expected.append(prior-0.95*trace)
    expected=torch.stack(expected)
    for value in result['estimates'].values():
        assert torch.allclose(torch.tensor(value['log_q'],dtype=x.dtype),expected,atol=1e-7,rtol=0)
    assert head.training
    assert torch.equal(g.ndata['x_1_true'],x)
