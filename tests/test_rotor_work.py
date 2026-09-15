import numpy as np
from scipy.special import logsumexp
from cfm_mol.rotor_work import log_reference,escort,complete_work,rotate_methyl


def test_exact_quadrature_identity_and_missing_jacobian_control():
    x=np.linspace(-np.pi,np.pi,65536,endpoint=False)+np.pi/65536
    kT=.0258519998
    energy=lambda angle:.04*(1-np.cos(3*angle))+.02*np.sin(angle)
    logq=log_reference(x)
    reference=np.exp(logq-energy(x)/kT).mean()*2*np.pi
    for amplitude in [-.5,0.,.5]:
        y,logj=escort(x,amplitude)
        w=complete_work(x,y,logj,energy(y),kT)
        estimate=np.exp(logq-w/kT).mean()*2*np.pi
        np.testing.assert_allclose(estimate,reference,atol=1e-12,rtol=1e-12)
        no_j=np.exp(logq-(w+kT*logj)/kT).mean()*2*np.pi
        if amplitude:
            assert abs(no_j-reference)>1e-3
        else:
            np.testing.assert_allclose(no_j,reference,atol=1e-12)


def test_rotation_preserves_internal_fragment_geometry_and_map_derivative():
    x=np.array([[0.,0.,0.],[0.,0.,1.],[1.,0.,1.],[-.5,.86,1.],[-.5,-.86,1.]])
    y=rotate_methyl(x,1,0,[2,3,4],.7)
    np.testing.assert_allclose(np.linalg.norm(y[1:]-y[1],axis=1),np.linalg.norm(x[1:]-x[1],axis=1))
    angles=np.array([-.8,.1,2.])
    _,logj=escort(angles,.5)
    h=1e-6
    derivative=(escort(angles+h,.5)[0]-escort(angles-h,.5)[0])/(2*h)
    np.testing.assert_allclose(derivative,np.exp(logj),rtol=1e-8)
