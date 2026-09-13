import math
import numpy as np
import torch
from scipy.integrate import quad
from cfm_mol.latent_mass_coupling import ShapeTwist,GaussianCoupling,reference_flow,reference_inverse,toy_energy,log_weight,exact_targets,com_basis


def test_shape_twist_preserves_gaussian_measure_inverse_and_spatial_symmetry():
    torch.manual_seed(29001);model=ShapeTwist(hidden=8).double()
    with torch.no_grad():model.angle_net[-1].weight.normal_(0,.3)
    x=torch.randn(4,2,3,dtype=torch.float64)
    y=model(x)
    torch.testing.assert_close(x.square().sum((-2,-1)),y.square().sum((-2,-1)),atol=1e-12,rtol=0)
    torch.testing.assert_close(model.inverse(y),x,atol=1e-12,rtol=0)
    q=torch.linalg.qr(torch.randn(3,3,dtype=x.dtype))[0];q[:,0]*=-1
    torch.testing.assert_close(model(x@q),model(x)@q,atol=1e-12,rtol=0)
    for z in x:
        jac=torch.autograd.functional.jacobian(lambda flat:model(flat.reshape(2,3)).flatten(),z.flatten())
        torch.testing.assert_close(torch.linalg.slogdet(jac)[1],z.new_tensor(0.),atol=1e-10,rtol=0)
    model.reflection=True
    torch.testing.assert_close(model.inverse(model(x)),x,atol=1e-12,rtol=0)


def test_gaussian_joint_marginal_and_equivariance():
    model=GaussianCoupling(.3,True).double();cross,noise=model.matrices()
    torch.testing.assert_close(cross@cross.T+noise@noise.T,torch.eye(2,dtype=torch.float64),atol=1e-12,rtol=0)
    z=torch.randn(5,2,3,dtype=torch.float64);eps=torch.randn_like(z);q=torch.linalg.qr(torch.randn(3,3,dtype=z.dtype))[0]
    torch.testing.assert_close(model(z@q,eps@q),model(z,eps)@q,atol=1e-12,rtol=0)


def test_flow_support_inverse_density_and_exact_physical_work():
    torch.manual_seed(29003);z=torch.randn(12,2,3,dtype=torch.float64)
    for component in [0,1]:
        x,logq=reference_flow(z,component)
        torch.testing.assert_close(x.mean(-2),torch.zeros_like(x.mean(-2)),atol=1e-14,rtol=0)
        torch.testing.assert_close(reference_inverse(x,component),z,atol=1e-10,rtol=0)
        for scenario in ['constant','nonlinear']:
            torch.testing.assert_close(-toy_energy(x,component,scenario)-logq,log_weight(z,component,scenario),atol=1e-10,rtol=0)
        a=1.+2*component
        assert abs(quad(lambda rho: math.exp(-3*math.log(math.pi)-5*math.log(rho))*math.pi**3*rho**5,a,a+1)[0]-1)<1e-12
        point=z[0];eps=1e-5;columns=[]
        for j in range(6):
            d=torch.zeros_like(point).flatten();d[j]=eps;d=d.reshape(2,3)
            plus=reference_flow(point+d,component)[0];minus=reference_flow(point-d,component)[0]
            columns.append((com_basis().T@(plus-minus)/(2*eps)).flatten())
        logdet=torch.linalg.slogdet(torch.stack(columns,1))[1]
        logprior=-3*math.log(2*math.pi)-point.square().sum()/2
        torch.testing.assert_close(logprior-logdet,logq[0],atol=1e-7,rtol=0)


def test_partition_functions_and_misspecification_bias():
    truth=exact_targets();mgfs=[quad(lambda t:2/math.pi*np.sqrt(1-t*t)*np.exp(k*t),-1,1,epsabs=1e-11)[0] for k in [1.,2.5]]
    np.testing.assert_allclose(truth['normalizers'],[mgfs[0],.4*mgfs[1]],atol=1e-10,rtol=0)
    assert abs(truth['component_mass'][1]-truth['fixed_shape_reverse_KL_optimal_mass'][1])>.1
