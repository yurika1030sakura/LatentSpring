import pytest
import torch
from scripts.research.recover_conditional_arc_chain import ReplayThenLiveOracle


def test_cached_requests_are_reused_once_and_coordinate_mismatch_never_calls_live():
    class Live:
        evaluated=0;requested_evaluations=0;handshake={}
        def evaluate_chunked(self,x,*,max_request):
            self.evaluated+=len(x);self.requested_evaluations+=len(x)
            return x.square().sum((1,2)),-2*x
    x=torch.tensor([[[1.,0.,0.],[-1.,0.,0.]]],dtype=torch.float64)
    query=dict(positions=x,raw_energy_eV=torch.tensor([2.]),inverted_energy_eV=torch.tensor([2.]),
        raw_force_eV_A=-2*x,inverted_force_eV_A=2*x,raw_queries_before=10,raw_queries_after=12)
    live=Live();o=ReplayThenLiveOracle(live,[query],10)
    with pytest.raises(AssertionError):o.evaluate_chunked(torch.cat([2*x,-2*x]))
    assert live.evaluated==0 and o.evaluated==10
    e,f=o.evaluate_chunked(torch.cat([x,-x]));assert e.tolist()==[2.,2.]
    assert live.evaluated==0 and o.evaluated==12 and o.cached_raw_calls==2
    o.evaluate_chunked(torch.cat([2*x,-2*x]))
    assert live.evaluated==2 and o.evaluated==14 and o.requested_evaluations==14
    assert o.cached_raw_calls==2
