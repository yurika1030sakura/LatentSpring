"""Gaussian-preserving latent shape couplings for component-mass calibration.

The general coupled-ratio estimation principle has prior art. The candidate
here is a nonlinear O(3)-equivariant coupling of two Jacobi vectors. It preserves
each reference generator's marginal, so learning can change estimator covariance
without changing either reference density or requiring a new likelihood model.
"""
import math
import numpy as np
import torch
from torch import nn
from scipy.special import gammainc,gammaincinv,iv


def invariants(z):
    if z.shape[-2:]!=(2,3):raise ValueError('Two three-dimensional Jacobi vectors required')
    s=z.square().sum((-2,-1)).clamp_min(1e-30)
    difference=(z[...,0,:].square().sum(-1)-z[...,1,:].square().sum(-1))/s
    cross=2*(z[...,0,:]*z[...,1,:]).sum(-1)/s
    return torch.stack([torch.log(s/6),difference,cross],-1)


def rotate(z,angle,reflection=False):
    c,s=torch.cos(angle)[...,None],torch.sin(angle)[...,None]
    u,v=z[...,0,:],z[...,1,:]
    if reflection:v=-v
    return torch.stack([c*u-s*v,s*u+c*v],-2)


class ShapeTwist(nn.Module):
    """Angle depends only on total squared radius, invariant along each orbit.

    Consequently the map preserves radius and Lebesgue volume, hence standard
    Gaussian measure. Its inverse simply negates the angle at the same radius.
    This transforms internal shape and commutes with a common spatial rotation.
    """
    def __init__(self,hidden=16,initial_angle=0.,reflection=False):
        super().__init__();self.reflection=reflection
        self.configuration=dict(hidden=hidden,initial_angle=initial_angle,reflection=reflection)
        self.angle_net=nn.Sequential(nn.Linear(1,hidden),nn.Tanh(),nn.Linear(hidden,1))
        nn.init.zeros_(self.angle_net[-1].weight)
        nn.init.constant_(self.angle_net[-1].bias,math.atanh(float(initial_angle)/(math.pi/2)))

    def angle(self,z):
        return math.pi/2*torch.tanh(self.angle_net(invariants(z)[...,:1])[...,0])

    def forward(self,z,noise=None):return rotate(z,self.angle(z),self.reflection)

    def inverse(self,z):
        # Forward is rotation followed (in input order) by an optional reflection.
        x=rotate(z,-self.angle(z))
        if self.reflection:x=torch.stack([x[...,0,:],-x[...,1,:]],-2)
        return x


class GaussianCoupling(nn.Module):
    """General O(3)-equivariant Gaussian joint coupling in two-vector space.

    Cross-covariance is S kron I3; independent noise supplies I-S S^T. This is
    the relevant full linear Gaussian family, not only a fixed rotation.
    """
    def __init__(self,initial_angle=0.,reflection=False):
        super().__init__();self.configuration=dict(initial_angle=initial_angle,reflection=reflection)
        self.left=nn.Parameter(torch.tensor(float(initial_angle)))
        self.right=nn.Parameter(torch.tensor(0.))
        self.raw_rho=nn.Parameter(torch.tensor([math.atanh(.95),math.atanh(-.95 if reflection else .95)]))

    @staticmethod
    def rotation(angle):
        c,s=angle.cos(),angle.sin()
        return torch.stack([torch.stack([c,-s]),torch.stack([s,c])])

    def matrices(self):
        left,right=self.rotation(self.left),self.rotation(self.right)
        rho=.9999*torch.tanh(self.raw_rho)
        cross=left@torch.diag(rho)@right.T
        noise=left@torch.diag(torch.sqrt(1-rho.square()))@left.T
        return cross,noise

    def forward(self,z,noise):
        cross,scale=self.matrices()
        return torch.einsum('ij,...jk->...ik',cross,z)+torch.einsum('ij,...jk->...ik',scale,noise)


class ResidualSurrogate(nn.Module):
    """Learn log importance weights from labelled reference samples only."""
    def __init__(self,hidden=16):
        super().__init__();self.configuration=dict(hidden=hidden)
        self.coefficients=nn.Sequential(nn.Linear(1,hidden),nn.Tanh(),nn.Linear(hidden,3))

    def forward(self,z):
        f=invariants(z);coeff=self.coefficients(f[...,:1])
        return coeff[...,0]+coeff[...,1]*f[...,1]+coeff[...,2]*f[...,2]


def analytic_angle(z,scenario):
    if scenario=='constant':return z.new_full(z.shape[:-2],.4)
    if scenario=='nonlinear':return 1.1*torch.tanh(1.5*invariants(z)[...,0])
    raise ValueError('Unknown constructed target')


def log_weight(z,component,scenario,kappas=(1.,2.5),relative_scale=.4):
    if component==0:return kappas[0]*invariants(z)[...,1]
    if component!=1:raise ValueError('Two components required')
    canonical=rotate(z,-analytic_angle(z,scenario))
    return math.log(relative_scale)+kappas[1]*invariants(canonical)[...,1]


def exact_targets(kappas=(1.,2.5),relative_scale=.4):
    # (|u|^2-|v|^2)/(|u|^2+|v|^2) has semicircle density for u,v~N3.
    mgf=lambda k:2*float(iv(1,k))/k
    z=np.array([mgf(kappas[0]),relative_scale*mgf(kappas[1])])
    mass=z/z.sum();shape_kl=np.log([mgf(k) for k in kappas])
    reverse_kl_mass=np.array([1.,relative_scale])/(1+relative_scale)
    variance=(mgf(2*kappas[0])/mgf(kappas[0])**2+mgf(2*kappas[1])/mgf(kappas[1])**2
              -2*mgf(sum(kappas))/(mgf(kappas[0])*mgf(kappas[1])))
    return dict(normalizers=z.tolist(),component_mass=mass.tolist(),log_normalizer_ratio=float(np.log(z[1]/z[0])),
        fixed_shape_reverse_KL_optimal_mass=reverse_kl_mass.tolist(),conditional_reverse_KL=shape_kl.tolist(),
        optimal_fixed_marginal_log_ratio_asymptotic_variance=float(variance))


def com_basis(dtype=torch.float64):
    return torch.tensor([[1/math.sqrt(2),1/math.sqrt(6)],[-1/math.sqrt(2),1/math.sqrt(6)],[0.,-2/math.sqrt(6)]],dtype=dtype)


def reference_flow(z,component):
    """Exact radial terminal transport to a six-dimensional COM shell.

    The path r_t=(1-t)r+t[a+CDF_chi6(r)] is monotone and defines a conditional
    flow-matching reference. This is an analytic reference, not trained FlowMol.
    Shells (1,2) and (3,4) are disjoint mathematical support sets, not chemistry.
    The SciPy CDF utilities are numerical sampling/evaluation references, not
    differentiable PyTorch training layers.
    """
    radius=z.square().sum((-2,-1)).sqrt()
    cdf=torch.as_tensor(gammainc(3.,radius.detach().cpu().numpy()**2/2),dtype=z.dtype,device=z.device)
    if ((cdf<=0)|(cdf>=1)).any():raise FloatingPointError('Numerical shell-map saturation')
    rho=(1.+2*component)+cdf
    jacobi=z*(rho/radius)[...,None,None]
    x=torch.einsum('ni,...ij->...nj',com_basis(z.dtype).to(z.device),jacobi)
    # Uniform radial density on a unit-width shell; area of S5 is pi^3.
    logq=-3*math.log(math.pi)-5*rho.log()
    return x,logq


def reference_inverse(x,component):
    jacobi=torch.einsum('in,...nj->...ij',com_basis(x.dtype).T.to(x.device),x)
    rho=jacobi.square().sum((-2,-1)).sqrt();cdf=rho-(1.+2*component)
    if ((cdf<=0)|(cdf>=1)).any():raise ValueError('Outside the specified shell')
    radius=torch.as_tensor(np.sqrt(2*gammaincinv(3.,cdf.detach().cpu().numpy())),dtype=x.dtype,device=x.device)
    return jacobi*(radius/rho)[...,None,None]


def toy_energy(x,component,scenario):
    z=reference_inverse(x,component);rho=x.square().sum((-2,-1)).sqrt()
    logq=-3*math.log(math.pi)-5*rho.log()
    return -logq-log_weight(z,component,scenario)


def joint_reverse_kl(mass,truth):
    mass=np.asarray(mass,dtype=float);target=np.asarray(truth['component_mass'])
    return float(np.sum(mass*(np.log(mass/target)+np.asarray(truth['conditional_reverse_KL']))))
