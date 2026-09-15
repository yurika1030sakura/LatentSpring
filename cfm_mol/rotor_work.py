"""Generalized escorted-work identity on a periodic molecular coordinate.

The rotor and map are a controlled mechanism test, not a full molecular generator.
Initial von Mises density and the escort Jacobian are both retained in the work.
"""
import numpy as np


def log_reference(angle, concentration=1.):
    return concentration*np.cos(angle)-np.log(2*np.pi*np.i0(concentration))


def escort(angle, amplitude):
    if not np.isfinite(amplitude) or abs(amplitude)>=1:
        raise ValueError('A periodic diffeomorphism requires |amplitude| < 1')
    angle=np.asarray(angle)
    return angle+amplitude*np.sin(angle),np.log1p(amplitude*np.cos(angle))


def complete_work(initial, final, log_jacobian, energy_difference, kT, concentration=1.):
    if not np.isfinite(kT) or kT<=0:
        raise ValueError('kT must be finite and positive')
    return energy_difference+kT*(log_reference(initial,concentration)-log_reference(final,concentration)-log_jacobian)


def rotate_methyl(positions, carbon, anchor, hydrogens, angle):
    x=np.asarray(positions,dtype=float)
    axis=x[carbon]-x[anchor]
    axis=axis/np.linalg.norm(axis)
    v=x[hydrogens]-x[carbon]
    rotated=v*np.cos(angle)+np.cross(axis,v)*np.sin(angle)+np.outer(v@axis,axis)*(1-np.cos(angle))
    y=x.copy();y[hydrogens]=x[carbon]+rotated
    return y-y.mean(0)
