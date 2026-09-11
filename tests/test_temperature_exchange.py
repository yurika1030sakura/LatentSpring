import torch
from cfm_mol.temperature_exchange import exchange_scaled_states
from cfm_mol.tempered_smc import DensityValue


def test_temperature_exchange_preserves_physical_gaussian_and_updates_cached_scores():
    generator=torch.Generator().manual_seed(9830)
    temperatures=torch.tensor([.03,.09,.27,.81],dtype=torch.float64)
    mean=torch.tensor([1.,-2.],dtype=torch.float64)
    w=torch.randn(16384,4,2,dtype=torch.float64,generator=generator)+mean/temperatures.sqrt()[:,None]
    def target(w):
        delta=w-mean/temperatures.sqrt()[:,None]
        return DensityValue(-.5*delta.square().sum(-1).reshape(-1),-delta.reshape(-1,2))
    value=target(w);labels=torch.arange(w.shape[0]*4).reshape(-1,4)
    for step in range(8):
        w,value,labels,stats=exchange_scaled_states(w,value,temperatures,labels,parity=step%2,generator=generator)
        independent=target(w)
        torch.testing.assert_close(value.log_value,independent.log_value,atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(value.score,independent.score,atol=1e-12,rtol=1e-12)
        assert 0<float(stats['accepted'].double().mean())<1
    residual=(w*temperatures.sqrt()[None,:,None]-mean)/temperatures.sqrt()[None,:,None]
    assert float(residual.mean(0).abs().max())<.035
    assert float((residual.var(0)-1).abs().max())<.045
    torch.testing.assert_close(labels.sort(1).values,torch.arange(len(labels)*4).reshape(-1,4))


def test_scaled_physical_swap_has_unit_jacobian_and_correct_log_ratio():
    t=torch.tensor([.025,1.],dtype=torch.float64)
    a=torch.tensor([.1,-.2,1.,-.5],dtype=torch.float64,requires_grad=True)
    scale=(t[1]/t[0]).sqrt()
    def swap(x):return torch.cat([x[2:]*scale,x[:2]/scale])
    jac=torch.autograd.functional.jacobian(swap,a)
    torch.testing.assert_close(torch.linalg.det(jac),torch.ones((),dtype=torch.float64))
    x=a.detach().reshape(1,2,2)
    logp=-.5*x.square().sum(-1)
    u=-t*logp
    ratio=(1/t[0]-1/t[1])*(u[:,0]-u[:,1])
    direct=-.5*swap(a).square().sum()+.5*a.square().sum()
    torch.testing.assert_close(ratio[0],direct)
