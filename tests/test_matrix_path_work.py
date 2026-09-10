import torch

from cfm_mol.path_work import gaussian_training_path


def test_identity_precision_recovers_scalar_kernel_and_checkpointed_gradients_include_covariance():
    x0=torch.randn(4,2,dtype=torch.float64,generator=torch.Generator().manual_seed(9076))
    a=torch.tensor(.3,dtype=torch.float64,requires_grad=True)
    settings=dict(terminal_std=.7,max_drift_norm=4.,mean_parameterization='native',noise_annealing_power=.5)
    def draw(matrix=False,saved=False):
        identity=lambda x,t:torch.eye(2,dtype=x.dtype)[None].expand(len(x),-1,-1)
        options=dict(forward_precision=identity,backward_precision=identity) if matrix else {}
        return gaussian_training_path(x0,lambda x,t:a*x,lambda x,t:-a*x,[0.,.3,.7,1.],.2,
            torch.Generator().manual_seed(9077),checkpoint_steps=saved,**settings,**options)
    plain=draw();identity=draw(True)
    torch.testing.assert_close(plain.terminal,identity.terminal,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(plain.log_forward_minus_backward,identity.log_forward_minus_backward,atol=1e-12,rtol=1e-12)
    rho=torch.tensor(.2,dtype=torch.float64,requires_grad=True)
    def objective(saved=False):
        def precision(x,t):return torch.eye(2,dtype=x.dtype)[None]+torch.exp(rho)*(x[...,None]*x[:,None,:])
        path=gaussian_training_path(x0,lambda x,t:a*x,lambda x,t:-a*x,[0.,.3,.7,1.],.2,
            torch.Generator().manual_seed(9077),checkpoint_steps=saved,forward_precision=precision,backward_precision=precision,**settings)
        return path.work(.5*path.terminal.square().sum(-1)).mean()
    value=objective();grad=torch.autograd.grad(value,(a,rho))
    saved_value=objective(True);saved_grad=torch.autograd.grad(saved_value,(a,rho))
    torch.testing.assert_close(value,saved_value,atol=1e-12,rtol=1e-12)
    for left,right in zip(grad,saved_grad):torch.testing.assert_close(left,right,atol=1e-10,rtol=1e-10)
    h=1e-5
    for parameter,expected in zip((a,rho),grad):
        old=parameter.detach().clone()
        with torch.no_grad():parameter.copy_(old+h)
        plus=objective().detach()
        with torch.no_grad():parameter.copy_(old-h)
        minus=objective().detach()
        with torch.no_grad():parameter.copy_(old)
        torch.testing.assert_close(expected,(plus-minus)/(2*h),atol=1e-6,rtol=1e-6)
