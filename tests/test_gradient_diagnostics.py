import torch

from cfm_mol.gradient_diagnostics import summarize_paired_gradients


def test_gaussian_fixed_score_forward_mean_matches_pathwise_but_can_have_high_noise():
    # Q=N(a,sigma^2), target N(1,1), fixed trivial backward factor.
    # There is no trajectory/normalizer ambiguity in this counterexample.
    batch=16;a=.3;sigma=.1
    epsilon=torch.randn(20000,batch,dtype=torch.float64,generator=torch.Generator().manual_seed(9068))
    y=a+sigma*epsilon
    work=.5*(y-1).square()-.5*epsilon.square()
    pathwise=(y-1).mean(-1)
    fixed_score=2*((work-work.mean(-1,keepdim=True))*epsilon/sigma).mean(-1)
    corrected=fixed_score/(2*(1-1/batch))
    assert abs(float(pathwise.mean())-(a-1))<.001
    assert abs(float(corrected.mean())-(a-1))<4*float(corrected.std()/len(corrected)**.5)
    assert corrected.var()>1000*pathwise.var()
    values={'pathwise_forward':[v.reshape(1) for v in pathwise[:64]],
        'fixed_score_forward':[v.reshape(1) for v in fixed_score[:64]],
        'pathwise_backward':[torch.zeros(1) for _ in range(64)],
        'fixed_score_backward':[torch.zeros(1) for _ in range(64)]}
    report=summarize_paired_gradients(values,batch)
    assert report['forward_population_scale']==1.875
    assert report['estimators']['pathwise_backward']['estimated_noise_to_mean_squared_ratio'] is None
    assert report['estimators']['fixed_score_forward']['trace_gradient_covariance']>1.
