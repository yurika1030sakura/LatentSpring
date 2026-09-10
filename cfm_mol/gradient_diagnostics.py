"""Descriptive paired gradient-noise diagnostics at frozen model parameters."""
import math

import torch


def summarize_paired_gradients(collections,batch_size):
    if batch_size<2:raise ValueError('Batch size must be at least two')
    output={};means={}
    for name,values in collections.items():
        matrix=torch.stack(values).double()
        if matrix.ndim!=2 or len(matrix)<2 or not torch.isfinite(matrix).all():
            raise ValueError('At least two finite, equal-dimensional gradients required')
        mean=matrix.mean(0);means[name]=mean
        norm=float(mean.norm());variance=float((matrix-mean).square().sum()/(len(matrix)-1))
        output[name]={'batches':len(matrix),'parameters':matrix.shape[1],
            'mean_gradient_norm':norm,'mean_single_batch_norm':float(matrix.norm(dim=1).mean()),
            'trace_gradient_covariance':variance,'mean_gradient_rms_standard_error':math.sqrt(variance/len(matrix)),
            'estimated_noise_to_mean_squared_ratio':variance/(norm*norm) if norm else None}
    # Only forward derivatives share a population gradient, up to this finite-B
    # correction: E[2/B sum_i (W_i-Wbar) grad log Q_i] = 2(1-1/B) grad E_Q[W].
    # The backward objectives differ; their gradient directions are not equivalent.
    factor=2*(1-1/batch_size)
    left=means['pathwise_forward'];right=means['fixed_score_forward']/factor
    if left.shape!=right.shape:raise ValueError('Forward gradient coordinates differ')
    denominator=float(left.norm()*right.norm())
    return {'estimators':output,'forward_population_scale':factor,
        'forward_mean_gradient_cosine':float(left@right)/denominator if denominator else None,
        'forward_scaled_mean_difference_norm':float((left-right).norm()),
        'limitations':['Frozen parameters; no optimizer or gradient clipping is applied.',
            'Finite-batch mean-norm estimates are themselves noisy.',
            'Forward means share an expectation under normalized forward kernels; backward objectives differ.',
            'These diagnostics do not establish molecular sampling performance.']}
