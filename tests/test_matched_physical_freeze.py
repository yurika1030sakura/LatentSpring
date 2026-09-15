import json
from pathlib import Path
import torch
from cfm_mol import matched_egnn as base
from scripts.research.run_matched_physical import make_model


def test_physical_optimizer_preserves_fixed_noise_schedule():
    spec=json.loads(Path('research/evidence/matched_generators_gaga_s0_v1.json').read_text())
    parent=base.initialize(spec,'cpu');state=parent.state_dict()
    model=make_model(dict(spec=spec),state,device='cpu').train()
    frozen={name:p.detach().clone() for name,p in model.named_parameters() if not p.requires_grad}
    assert 'gamma.gamma' in frozen
    before={name:p.detach().clone() for name,p in model.named_parameters() if p.requires_grad}
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,amsgrad=True,weight_decay=1e-12)
    x=base.center(torch.randn(2,5,3));z=torch.tensor([[6,6,8,1,1]]*2)
    loss=base.loss(model,x,z,'gaga',spec,base.HarmonicSource(),46419)
    optimizer.zero_grad(set_to_none=True);loss.backward()
    assert model.gamma.gamma.grad is None
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
    for name,value in frozen.items():torch.testing.assert_close(model.state_dict()[name],value,atol=0,rtol=0)
    assert any(not torch.equal(model.state_dict()[name],value) for name,value in before.items())
