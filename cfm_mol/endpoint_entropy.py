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
