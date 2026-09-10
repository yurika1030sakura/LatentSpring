"""Endpoint KL gradient helpers for an explicit terminal-noise generator.

The score-difference gradient and denoising regression are established methods
(including VSD/DMD), not claimed novelty. A learned proposal score must describe
the actual generator distribution. An arbitrary FM velocity readout is not a
substitute. The surrogate's scalar value is not a KL estimate.
"""
import math

import torch


def denoising_score_loss(proposal_score, final_noise, noise_std):
    """DSM for y=F(z)+sigma*epsilon; F and z do not depend on epsilon.

    The conditional optimum is the score of this noisy endpoint distribution.
    Extra corruption or a different sigma changes the distribution being scored.
    This loss alone does not certify score accuracy or target calibration.
    """
    if proposal_score.ndim != 2 or proposal_score.shape != final_noise.shape:
        raise ValueError('Matched [batch,dimension] score and noise required')
    if not math.isfinite(noise_std) or noise_std <= 0:
        raise ValueError('Positive finite terminal noise required')
    if not torch.isfinite(proposal_score).all() or not torch.isfinite(final_noise).all():
        raise ValueError('Non-finite score or noise')
    return (noise_std*proposal_score+final_noise.detach()).square().mean()


def endpoint_kl_surrogate(endpoint, proposal_score, target_score):
    """Backpropagate E[(s_q(y)-s_pi(y)) d_theta y], with both scores held fixed.

    With the true proposal score and regularity, the expectation is the marginal
    reverse-KL gradient. Learned/stale scores introduce gradient error. Target
    score for the specified molecular Boltzmann law is physical force/kT,
    including the restraint, in the same orthonormal COM coordinates.
    """
    if endpoint.ndim != 2 or any(v.shape != endpoint.shape for v in [proposal_score, target_score]):
        raise ValueError('Matched [batch,dimension] tensors required')
    if not all(torch.isfinite(v).all() for v in [endpoint, proposal_score, target_score]):
        raise ValueError('Non-finite endpoint or score')
    return (endpoint*(proposal_score-target_score).detach()).sum(-1).mean()


def normalized_dsm_loss(score, noise, noise_std, *, mean_score=None):
    """DSM without its parameter-constant term, optionally using a known CV.

    score is evaluated at mean+sigma*noise. mean_score is evaluated at the
    conditional mean, independently of that noise. Its zero-mean term must
    retain parameter derivatives to reduce gradient variance. This established
    control variate (e.g. nonlinear DSM) changes neither the expected gradient
    nor the finite-sigma marginal being fitted. Antithetic examples can instead
    be supplied as paired rows. Scaling changes optimizer numerics, so include
    a matched scaled-IID control when comparing to raw DSM.
    """
    if score.ndim != 2 or score.shape != noise.shape:
        raise ValueError('Matched score and Gaussian-noise matrices required')
    if not math.isfinite(noise_std) or noise_std <= 0:
        raise ValueError('Positive finite Gaussian scale required')
    if mean_score is not None and mean_score.shape != score.shape:
        raise ValueError('Mean score must match noisy score')
    terms = [score, noise] if mean_score is None else [score, noise, mean_score]
    if not all(torch.isfinite(v).all() for v in terms):
        raise ValueError('Non-finite DSM control-variate inputs')
    difference = score if mean_score is None else score-mean_score
    return (.5*score.square()+difference*noise.detach()/noise_std).mean()
