"""Run only in the isolated JAX baseline environment."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cfm_mol.jax_energy_oracle import make_even_log_target

jax.config.update('jax_enable_x64',True)


class AnalyticRawOracle:
    def __init__(self):self.evaluated=0
    def evaluate_chunked(self,x,*,max_request):
        self.evaluated+=len(x)
        k=np.arange(1,x.shape[1]+1)[None,:,None]
        energy=.5*(k*x*x).sum((1,2))+.3*x[:,0,0]**3
        force=-k*x
        force[:,0,0]-=.9*x[:,0,0]**2
        return energy,force


def reference(x):
    x=x-x.mean(-2,keepdims=True)
    k=jnp.arange(1,x.shape[-2]+1)[...,None]
    return -(.5*((k+.1)*x*x).sum((-1,-2))-2.)/.7


def test_jitted_batch_vjp_cotangent_axes_and_parity():
    oracle=AnalyticRawOracle();fn=make_even_log_target(oracle,kT=.7,restraint=.1,energy_zero_eV=2.)
    x=jax.random.normal(jax.random.PRNGKey(13),(4,5,3),dtype=jnp.float64)+3.
    weights=jnp.array([1.,-2.,.4,3.])
    value,gradient=jax.jit(jax.value_and_grad(lambda q:jnp.sum(weights*fn(q))))(x)
    expected,expected_gradient=jax.value_and_grad(lambda q:jnp.sum(weights*reference(q)))(x)
    np.testing.assert_allclose(value,expected,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(gradient,expected_gradient,rtol=1e-12,atol=1e-12)
    assert oracle.evaluated==8
    original=jax.jit(fn)(x);mirrored=jax.jit(fn)(-x)
    np.testing.assert_allclose(original,mirrored,rtol=1e-12,atol=1e-12)
    assert oracle.evaluated==24
    np.testing.assert_allclose(np.asarray(gradient).sum(1),0,atol=1e-12)


def test_single_and_vmap_agree_with_full_batch():
    oracle=AnalyticRawOracle();fn=make_even_log_target(oracle,kT=.7,restraint=.1,energy_zero_eV=2.)
    x=jax.random.normal(jax.random.PRNGKey(14),(2,4,3),dtype=jnp.float64)
    values,gradients=jax.jit(jax.vmap(jax.value_and_grad(fn)))(x)
    expected,expected_gradients=jax.vmap(jax.value_and_grad(reference))(x)
    np.testing.assert_allclose(values,expected,atol=1e-12)
    np.testing.assert_allclose(gradients,expected_gradients,atol=1e-12)
    assert oracle.evaluated==4
    with pytest.raises(ValueError,match='float64'):
        fn(x.astype(jnp.float32))


def test_raw_training_estimator_averages_to_projected_energy_and_gradient():
    from cfm_mol.jax_energy_oracle import make_raw_training_log_energy
    oracle=AnalyticRawOracle()
    raw=make_raw_training_log_energy(oracle,kT=.7,restraint=.1,energy_zero_eV=2.)
    even=make_even_log_target(oracle,kT=.7,restraint=.1,energy_zero_eV=2.)
    x=jax.random.normal(jax.random.PRNGKey(95),(3,4,3),dtype=jnp.float64)
    signs=jnp.concatenate([x,-x],axis=0)
    v,g=jax.jit(jax.value_and_grad(lambda scale:jnp.mean(raw(scale*signs))))(1.1)
    ev,eg=jax.jit(jax.value_and_grad(lambda scale:jnp.mean(even(scale*x))))(1.1)
    np.testing.assert_allclose([v,g],[ev,eg],atol=1e-12,rtol=1e-12)
    assert oracle.evaluated==12
