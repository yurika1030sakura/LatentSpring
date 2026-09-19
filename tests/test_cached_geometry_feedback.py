from types import SimpleNamespace
import pytest
import torch
from cfm_mol.cached_geometry_feedback import sample_cached


@pytest.mark.parametrize('calls',[2,64,128])
def test_exact_budget_determinism_centering_and_real_previous_geometry(monkeypatch,calls):
    from cfm_mol import matched_egnn as base
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(1));self.norm_values=[1.]
            block=torch.nn.Module();block._geometry_context=None
            egnn=torch.nn.Module();egnn.n_layers=1;egnn.add_module('e_block_0',block)
            self.dynamics=SimpleNamespace(egnn=egnn)
    class Source:
        def sample(self,numbers,rng):
            x=torch.tensor(rng.normal(size=(len(numbers),3)));return x-x.mean(0)
    model=Model();records=[]
    def vector(model,x,t,numbers,spec):
        geometry=model.dynamics.egnn.e_block_0._geometry_context
        records.append((x.clone(),t.clone(),geometry.clone()))
        return base.center(-.2*x+.05*geometry)
    monkeypatch.setattr(base,'vector',vector)
    spec=dict(kind='harmonic_fm',two_pass=True);context=lambda x,z:x
    y,x0=sample_cached(model,[1,6,8],spec,Source(),context,12,2,calls)
    assert len(records)==calls and all((r[1]<1).all() for r in records)
    torch.testing.assert_close(records[0][2],x0,atol=1e-7,rtol=0)
    torch.testing.assert_close(records[1][2],.85*x0,atol=2e-7,rtol=0)
    assert model.dynamics.egnn.e_block_0._geometry_context is None
    torch.testing.assert_close(y.mean(1),torch.zeros(2,3),atol=1e-7,rtol=0)
    repeat,_=sample_cached(model,[1,6,8],spec,Source(),context,12,2,calls)
    torch.testing.assert_close(y,repeat,atol=0,rtol=0)
