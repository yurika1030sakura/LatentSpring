import math
import pytest
import torch
from cfm_mol.gate_teacher import oracle_gate_targets,log_retention_lower_bound
from cfm_mol.source_force_screen import screened_log_acceptance


def test_oracle_targets_preserve_original_acceptance_and_minimize_each_gate():
    r=torch.tensor([-1e6,-4.,-.7,0.,.4,5.,1e6],dtype=torch.float64);bound=math.log(16)
    f,rev=oracle_gate_targets(r,bound);_,total=screened_log_acceptance(r,f,rev)
    expected=torch.minimum(torch.zeros_like(r),r)
    torch.testing.assert_close(total,expected,atol=1e-12,rtol=0)
    # Lowering a non-floor gate while retaining its partner loses original flow.
    for i in range(len(r)):
        if f[i]>-bound:
            smaller=f.clone();smaller[i]-=min(.01,float(f[i]+bound)/2)
            _,changed=screened_log_acceptance(r,smaller,rev);assert changed[i]<expected[i]
        if rev[i]>-bound:
            smaller=rev.clone();smaller[i]-=min(.01,float(rev[i]+bound)/2)
            _,changed=screened_log_acceptance(r,f,smaller);assert changed[i]<expected[i]


def test_underprediction_bound_controls_retained_probability_for_general_gates():
    rng=torch.Generator().manual_seed(25941);bound=math.log(16)
    r=30*torch.randn(200,dtype=torch.float64,generator=rng)
    f=-bound*torch.rand(200,dtype=r.dtype,generator=rng);rev=-bound*torch.rand(200,dtype=r.dtype,generator=rng)
    _,total=screened_log_acceptance(r,f,rev);original=torch.minimum(torch.zeros_like(r),r)
    lower=log_retention_lower_bound(r,f,rev,bound)
    assert (total-original>=lower-1e-12).all() and (total<=original+1e-12).all()
    assert (lower>=-bound-1e-12).all()
    with pytest.raises(ValueError):oracle_gate_targets(r.requires_grad_())
