"""First-order JAX log-target bridge to the real inversion-averaged potential.

No Torch or fairchem import is needed in the baseline environment. The callback
is deterministic in coordinates. Its observed worker counts are instrumentation,
not a promised number of callback executions: JAX may elide/repeat pure calls.
Block on returned results before reading the actual query counters.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np


def make_even_log_target(oracle, *, kT, restraint=.1, energy_zero_eV=0., max_request=32):
    """Log unnormalized pi_plus on COM-free coordinates, extended by centering.

    Accepts [N,3] or [B,N,3]. Uses BOTH raw orientations on every real callback.
    Gradient is P_H(F_plus-restraint*x)/kT. The VJP broadcasts a batch cotangent
    over the atom and Cartesian axes; multiplying a bare [B] vector is wrong.
    Only first-order reverse differentiation is supported, not force Hessians.
    """
    if not math.isfinite(kT) or kT<=0 or not math.isfinite(restraint) or restraint<0 or not math.isfinite(energy_zero_eV):
        raise ValueError('Invalid target temperature, confinement or energy offset')
    if not isinstance(max_request,int) or max_request<1:raise ValueError('Positive RPC bound required')
    def callback(positions):
        x=np.asarray(positions,dtype=np.float64)
        single=x.ndim==2
        if single:x=x[None]
        if x.ndim!=3 or x.shape[-1]!=3 or not np.isfinite(x).all():
            raise ValueError('Finite Cartesian positions required')
        x=x-x.mean(axis=1,keepdims=True)
        energies,forces=oracle.evaluate_chunked(np.concatenate([x,-x]),max_request=max_request)
        energy,inverted_energy=np.split(energies,2)
        force,inverted_force=np.split(forces,2)
        physical_energy=.5*(energy+inverted_energy)
        physical_force=.5*(force-inverted_force)
        logp=-(physical_energy+.5*restraint*(x*x).sum(axis=(1,2))-energy_zero_eV)/kT
        gradient=(physical_force-restraint*x)/kT
        gradient-=gradient.mean(axis=1,keepdims=True)
        dtype=np.asarray(positions).dtype
        if single:return np.asarray(logp[0],dtype=dtype),np.asarray(gradient[0],dtype=dtype)
        return np.asarray(logp,dtype=dtype),np.asarray(gradient,dtype=dtype)

    def forward(x):
        if x.ndim not in (2,3) or x.shape[-1]!=3 or x.dtype!=jnp.float64:
            raise ValueError('Bridge requires float64 [N,3] or [B,N,3] inputs')
        shapes=(jax.ShapeDtypeStruct(x.shape[:-2],x.dtype),jax.ShapeDtypeStruct(x.shape,x.dtype))
        logp,gradient=jax.pure_callback(callback,shapes,x)
        return logp,gradient

    @jax.custom_vjp
    def log_target(x):
        return forward(x)[0]

    def backward(gradient,cotangent):
        return (cotangent[...,None,None]*gradient,)

    log_target.defvjp(forward,backward)
    return log_target
