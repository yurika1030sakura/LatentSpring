"""Normalized Gaussian sampling and scoring from a full relative precision."""
import math

import torch


def precision_cholesky(precision):
    if precision.ndim!=3 or precision.shape[-1]!=precision.shape[-2] or not torch.isfinite(precision).all():raise ValueError('Finite batched square precision required')
    if not torch.allclose(precision,precision.transpose(-1,-2),rtol=1e-9,atol=1e-10):raise ValueError('Symmetric precision required')
    return torch.linalg.cholesky(precision)


def gaussian_precision_sample(mean,cholesky,std,noise):
    if not math.isfinite(std) or std<=0 or mean.shape!=noise.shape or cholesky.shape!=(len(mean),mean.shape[1],mean.shape[1]):raise ValueError('Invalid Gaussian sample inputs')
    displacement=torch.linalg.solve_triangular(cholesky.transpose(-1,-2),noise[...,None],upper=True).squeeze(-1)
    return mean+std*displacement


def gaussian_precision_log_density(value,mean,cholesky,std):
    if value.shape!=mean.shape or value.ndim!=2 or cholesky.shape!=(len(value),value.shape[1],value.shape[1]):raise ValueError('Invalid Gaussian density inputs')
    if not math.isfinite(std) or std<=0:raise ValueError('Positive Gaussian scale required')
    residual=(value-mean)/std
    white=(cholesky.transpose(-1,-2)@residual[...,None]).squeeze(-1)
    return -.5*white.square().sum(-1)+cholesky.diagonal(dim1=-2,dim2=-1).log().sum(-1)-value.shape[1]*math.log(std*math.sqrt(2*math.pi))
