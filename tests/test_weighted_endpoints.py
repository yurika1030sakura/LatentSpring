"""New endpoint measure + native loss/sampler checks, no mock DGL installation."""
from dataclasses import replace
import json
import math
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from cfm_mol.weighted_endpoints import EndpointGroup, WeightedEndpointStore, composition_key
from cfm_mol.weighted_endpoint_fm import endpoint_fm_loss, checkpoint_source
from cfm_mol.endpoint_diagnostics import TensorGraph, radial_model
from cfm_mol.tree_prior_controls import TreePriorControl
from cfm_mol.clamped_density import sample_clamped_flow
from cfm_mol.curvature_escort import centered_basis, fit_curvature, affine_parameters


def group(**kwargs):
    x = np.array([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],
                  [[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],
                  [[0.,0.,0.],[1.2,0.,0.],[0.,1.2,0.]]])
    values = dict(group_id='test', graph_id='graphA', numbers=(8,1,1), charge=0,
        spin_multiplicity=1, temperature_K=300., mixture_mass=1., minimum_positions=x,
        log_weight=np.log([.3,.5,.2]), success=np.ones(3,dtype=bool),
        basin_id=('A','A','B'), particle_id=('0','1','2'),
        target=dict(oracle_id='synthetic',base_measure='declared',support='all',
                    optimizer='fixed map',weight_scope='graph_global',split='train'), provenance={})
    values.update(kwargs)
    return EndpointGroup(**values)


def test_log_weights_stable_and_dedup_preserves_mass():
    g=group(log_weight=np.log([.3,.5,.2])+10000)
    a=g.aggregate_identical_minima()
    assert a.masses()['A'] == pytest.approx(.8)
    assert a.masses()['B'] == pytest.approx(.2)
    assert len(a.log_weight)==2
    assert len(a.provenance['merged_particle_ids'][0])==2


@pytest.mark.parametrize('weights', [[-np.inf]*3, [0,np.nan,0], [0,np.inf,0]])
def test_invalid_weights_fail(weights):
    with pytest.raises(ValueError): group(log_weight=weights)


def test_failed_mass_is_not_silently_renormalized():
    g=group(success=np.array([True,True,False]))
    with pytest.raises(ValueError, match='failed positive'): WeightedEndpointStore([g])
    s=WeightedEndpointStore([g],failure_policy='condition_within_group')
    assert s.audit()['target_changed_by_conditioning']
    assert s.failed_mass==pytest.approx(.2)
    assert {r.basin_id for r in s.draw(100,np.random.default_rng(0))}=={'A'}


def test_numerically_underflowed_failure_is_still_rejected():
    g=group(log_weight=np.array([0.,0.,-1e6]),success=np.array([True,True,False]))
    with pytest.raises(ValueError): WeightedEndpointStore([g])


def test_per_anchor_weights_not_accepted_as_global():
    t=group().target.copy();t['weight_scope']='anchor_local'
    with pytest.raises(ValueError,match='Per-anchor'): group(target=t)


def test_different_representatives_are_not_blindly_deduplicated():
    x=group().minimum_positions.copy();x[1,0,0]+=.1
    with pytest.raises(ValueError,match='different aligned'): group(minimum_positions=x).aggregate_identical_minima()


def test_composition_condition_has_no_graph_or_basin_or_coordinate():
    a=group();b=group(group_id='other',graph_id='graphB')
    assert a.condition()==b.condition()
    assert not {'graph_id','basin_id','positions','bonds'}.intersection(a.condition())
    assert composition_key((8,1,1),0,1)==composition_key((1,8,1),0,1)


def test_outer_prior_and_basin_sampling_reproduce_declared_mass():
    a=group(mixture_mass=.25)
    b=group(group_id='other',graph_id='graphB',mixture_mass=.75)
    s=WeightedEndpointStore([a,b]);d=s.draw(20000,np.random.default_rng(35))
    assert abs(np.mean([r.group_id=='test' for r in d])-.25)<.015
    assert abs(np.mean([r.basin_id=='A' for r in d])-.8)<.015


def test_centered_endpoint_smoothing_is_reproducible_and_nonzero():
    s=WeightedEndpointStore([group()])
    a=s.draw(20,np.random.default_rng(10),sigma_A=.005)
    b=s.draw(20,np.random.default_rng(10),sigma_A=.005)
    for x,y in zip(a,b):
        np.testing.assert_array_equal(x.positions,y.positions)
        np.testing.assert_allclose(x.positions.mean(0),0,atol=1e-15)
    assert np.std([x.positions[0,2] for x in a])>0


def test_roundtrip_and_checksum(tmp_path):
    s=WeightedEndpointStore([group()]);s.save(tmp_path/'store')
    q=WeightedEndpointStore.load(tmp_path/'store')
    assert s.audit()==q.audit()
    path=tmp_path/'store'/'group_0000.npz';path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='checksum'): WeightedEndpointStore.load(tmp_path/'store')


def test_unsafe_path_rejected(tmp_path):
    s=WeightedEndpointStore([group()]);s.save(tmp_path/'store')
    path=tmp_path/'store'/'manifest.json';v=json.loads(path.read_text())
    v['groups'][0]['array_file']='../elsewhere.npz';path.write_text(json.dumps(v))
    with pytest.raises(ValueError,match='Unsafe'):WeightedEndpointStore.load(tmp_path/'store')


def test_original_fm_loss_harmonic_pairing_and_native_backprop():
    torch.manual_seed(100)
    s=WeightedEndpointStore([group()]);draws=s.draw(4,np.random.default_rng(33),sigma_A=.005)
    graph=TensorGraph(draws);model=radial_model();prior=TreePriorControl('harmonic_tree').double()
    before=graph.ndata['x_1_true'].clone()
    loss=endpoint_fm_loss(model,graph,graph.nbi,graph.edges()[0]<graph.edges()[1],draws,prior,10)
    loss.backward()
    grads=[v.grad for v in model.vector_field.parameters() if v.grad is not None]
    assert torch.isfinite(loss) and grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads)>0
    assert torch.equal(graph.ndata['x_1_true'],before)
    assert model.vector_field.training


def test_native_sampler_does_not_use_reference_endpoint_coordinates():
    s=WeightedEndpointStore([group()]);draws=s.draw(2,np.random.default_rng(4))
    graph=TensorGraph(draws);model=radial_model();prior=TreePriorControl('harmonic_tree').double()
    x0=checkpoint_source(prior,draws,301).float();mask=graph.edges()[0]<graph.edges()[1]
    with torch.no_grad():
        a=sample_clamped_flow(model,graph,graph.nbi,mask,x0=x0,n_ode_steps=4,terminal_time=1.,parameterization='displacement')
        graph.ndata['x_1_true']=torch.randn_like(graph.ndata['x_1_true'])*1000
        b=sample_clamped_flow(model,graph,graph.nbi,mask,x0=x0,n_ode_steps=4,terminal_time=1.,parameterization='displacement')
    torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_zero_pilot_curvature_reduces_to_force_translation():
    basis=centered_basis(3);empty=torch.empty(0,3,3,dtype=torch.float64)
    h,k=fit_curvature(empty,empty,basis)
    force=torch.tensor([[1.,2.,3.],[-1.,0.,-1.],[0.,-2.,-2.]],dtype=torch.float64)
    a,b,j=affine_parameters(force,.03,.025,h,basis)
    torch.testing.assert_close(a,torch.eye(6,dtype=torch.float64))
    torch.testing.assert_close(b,(basis.T@force.flatten())*.03**2/.025)
    assert j==0 and k==0


def test_archived_midpoint_is_second_order_for_time_dependent_field():
    class TimeLinear(nn.Module):
        def forward(self,g,t,node_batch_idx,**kw):return {'x':t[node_batch_idx,None]*g.ndata['x_t']}
    draws=WeightedEndpointStore([group()]).draw(1,np.random.default_rng(1))
    g=TensorGraph(draws,dtype=torch.float64);m=SimpleNamespace(vector_field=TimeLinear())
    x=g.ndata['x_1_true']-g.ndata['x_1_true'].mean(0);errors=[]
    for n in [8,16,32]:
        y=sample_clamped_flow(m,g,g.nbi,g.edges()[0]<g.edges()[1],x0=x,n_ode_steps=n,
            terminal_time=1.,parameterization='velocity')
        errors.append(float((y-math.exp(.5)*x).abs().max()))
    assert errors[0]/errors[1]>3.5 and errors[1]/errors[2]>3.5


def test_existing_smc_population_transfers_current_not_ancestral_weights():
    from cfm_mol.tempered_smc import IsotropicGaussianMixture, tempered_smc
    from cfm_mol.weighted_endpoints import endpoints_from_smc
    gen=torch.Generator().manual_seed(401)
    initial=IsotropicGaussianMixture(torch.zeros(1,2,dtype=torch.float64),2.)
    target_density=IsotropicGaussianMixture(torch.tensor([[-1.,0.],[1.,0.]],dtype=torch.float64),.3)
    x,_=initial.sample(64,gen)
    population=tempered_smc(x,initial,target_density,[0.,.1,.3,.6,1.],proposal_std=.3,
        generator=gen,kernel='mala',moves_per_stage=2,resample_threshold=.8)
    sign=population.positions[:,0].numpy()>0
    coords=np.repeat(group().minimum_positions[:1],64,axis=0).copy()
    coords[sign]*=1.2
    g=endpoints_from_smc(population,minimum_positions=coords,success=np.ones(64,dtype=bool),
        basin_id=tuple('B' if k else 'A' for k in sign),group_id='smc',graph_id='g',
        numbers=(8,1,1),charge=0,spin_multiplicity=1,temperature_K=300.,mixture_mass=1.,
        target=group().target,provenance={})
    a=g.aggregate_identical_minima()
    expected=float(population.log_weights.exp()[torch.from_numpy(sign)].sum())
    assert a.masses().get('B',0)==pytest.approx(expected,abs=1e-12)
    assert population.summary()['resampling_events']>0
    assert g.provenance['weight_source'].startswith('current')


def test_duplicate_graph_populations_cannot_be_equal_weighted_implicitly():
    a=group(mixture_mass=.5);b=group(group_id='anchor2',mixture_mass=.5)
    with pytest.raises(ValueError,match='Duplicate graph'):WeightedEndpointStore([a,b])


def test_invalid_electron_multiplicity_is_rejected():
    with pytest.raises(ValueError,match='Electron count'):group(spin_multiplicity=2)
