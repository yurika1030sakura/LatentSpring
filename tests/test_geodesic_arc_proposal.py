import math
import numpy as np

import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from cfm_mol.chemical_moves import covalent_radii,infer_chemical_graph
from cfm_mol.geodesic_arc_proposal import (TAU,intersect_arcs,root_distance_constraints,
    inverse_exponential_fraction,direction_log_prob,draw_direction,circle_law,arc_teacher_kl)


def test_exact_arc_intersections_match_pointwise_halfspaces_and_empty_domain():
    rng=torch.Generator().manual_seed(25301)
    base=torch.tensor([0.,0.,1.],dtype=torch.float64);tangent=torch.tensor([1.,0.,0.],dtype=torch.float64)
    normals=torch.randn(9,3,dtype=torch.float64,generator=rng)
    limits=normals@base+.3*torch.rand(9,dtype=torch.float64,generator=rng)
    arcs=intersect_arcs(normals,limits,base,tangent)
    theta=(torch.arange(10001,dtype=torch.float64)+.5)*TAU/10001
    directions=theta.cos()[:,None]*base+theta.sin()[:,None]*tangent
    direct=(directions@normals.T<=limits).all(1)
    intervals=torch.zeros_like(direct)
    for lo,hi in arcs:intervals|=(theta>=lo)&(theta<=hi)
    assert torch.equal(direct,intervals) and intervals.any() and (~intervals).any()
    assert intersect_arcs(torch.zeros(1,3,dtype=torch.float64),torch.tensor([-1.],dtype=torch.float64),base,tangent)==[]


def test_exponential_segment_inverse_has_correct_cdf_at_extreme_slopes():
    for delta in [-1000.,-50.,-1.,-1e-8,0.,1e-8,1.,50.,1000.]:
        for u in [.1,.5,.9]:
            t=float(inverse_exponential_fraction(torch.tensor(u,dtype=torch.float64),torch.tensor(delta,dtype=torch.float64)))
            assert 0<t<1
            if delta>0:cdf=math.exp(delta*(t-1))*math.expm1(-delta*t)/math.expm1(-delta)
            elif delta<0:cdf=math.expm1(delta*t)/math.expm1(delta)
            else:cdf=t
            assert abs(cdf-u)<1e-10


def test_sphere_density_has_two_preimages_and_spherical_jacobian():
    base=torch.tensor([0.,0.,1.],dtype=torch.float64)
    normals=torch.empty(0,3,dtype=torch.float64);limits=torch.empty(0,dtype=torch.float64)
    eta=torch.zeros(3,dtype=torch.float64)
    total=0.
    for theta in (torch.arange(48,dtype=torch.float64)+.5)*math.pi/48:
        y=torch.stack([theta.sin(),theta*0,theta.cos()])
        logq,_=direction_log_prob(y,base,normals,limits,eta)
        expected=-math.log(2*math.pi**2)-float(theta.sin().log())
        assert abs(float(logq)-expected)<1e-10
        total+=float(logq.exp()*theta.sin())*TAU*math.pi/48
    assert abs(total-1)<1e-10
    # Direct surface-area Jacobian of the two-angle map, independent of q.
    coordinates=torch.tensor([.37,.91],dtype=torch.float64,requires_grad=True)
    def mapping(z):
        phi,theta=z
        return torch.stack([theta.sin()*phi.cos(),theta.sin()*phi.sin(),theta.cos()])
    jac=torch.autograd.functional.jacobian(mapping,coordinates)
    area=torch.cross(jac[:,0],jac[:,1],dim=0).norm()
    torch.testing.assert_close(area,coordinates[1].sin())


def test_density_parameter_gradient_matches_finite_difference():
    base=torch.tensor([0.,0.,1.],dtype=torch.float64)
    y=torch.tensor([.3,.4,math.sqrt(.75)],dtype=torch.float64)
    normals=torch.tensor([[0.,0.,-1.]],dtype=torch.float64);limits=torch.tensor([-.2],dtype=torch.float64)
    eta=torch.tensor([3.,-1.,-20.],dtype=torch.float64,requires_grad=True)
    value,_=direction_log_prob(y,base,normals,limits,eta);gradient=torch.autograd.grad(value,eta)[0]
    for i in range(3):
        perturb=torch.zeros(3,dtype=torch.float64);perturb[i]=1e-5
        a,_=direction_log_prob(y,base,normals,limits,eta.detach()+perturb)
        b,_=direction_log_prob(y,base,normals,limits,eta.detach()-perturb)
        assert abs(float((a-b)/(2e-5)-gradient[i]))<1e-8


def test_arc_kl_matches_independent_quadrature_and_has_stationary_matching_gradient():
    base=torch.tensor([0.,0.,1.],dtype=torch.float64);tangent=torch.tensor([1.,0.,0.],dtype=torch.float64)
    normals=torch.tensor([[0.,0.,-1.]],dtype=torch.float64);limits=torch.tensor([-.95],dtype=torch.float64)
    teacher_eta=torch.tensor([5.,1.,-200.],dtype=torch.float64)
    student_eta=torch.tensor([3.,2.,100.],dtype=torch.float64)
    teacher=circle_law(base,tangent,normals,limits,teacher_eta)
    student=circle_law(base,tangent,normals,limits,student_eta)
    exact=float(arc_teacher_kl(teacher,student))
    nodes,weights=np.polynomial.legendre.leggauss(64);fraction=(nodes+1)/2
    integral=0.;mass=0.
    for i in range(len(teacher['edges'])):
        logt=float(teacher['heights'][i,0])+float(teacher['delta'][i])*fraction-float(teacher['log_normalizer'])
        logs=float(student['heights'][i,0])+float(student['delta'][i])*fraction-float(student['log_normalizer'])
        w=weights*float(teacher['width'][i])/2
        integral+=float(np.sum(w*np.exp(logt)*(logt-logs)))
        mass+=float(np.sum(w*np.exp(logt)))
    assert abs(mass-1)<1e-10 and abs(integral-exact)<1e-9 and exact>0
    matching=teacher_eta.clone().requires_grad_()
    loss=arc_teacher_kl(teacher,circle_law(base,tangent,normals,limits,matching))
    gradient=torch.autograd.grad(loss,matching)[0]
    assert abs(float(loss))<1e-10 and float(gradient.abs().max())<1e-10


def test_known_cap_target_is_preserved_with_mh_but_not_without_correction():
    rng=torch.Generator().manual_seed(25311);count=512;lower=.2;beta=3.
    uniforms=torch.rand(count,dtype=torch.float64,generator=rng)
    z=lower+(1-lower)*inverse_exponential_fraction(uniforms,uniforms.new_full((count,),beta*(1-lower)))
    phi=TAU*torch.rand(count,dtype=torch.float64,generator=rng)
    radial=(1-z.square()).sqrt();states=torch.stack([radial*phi.cos(),radial*phi.sin(),z],1)
    normals=torch.tensor([[0.,0.,-1.]],dtype=torch.float64);limits=torch.tensor([-lower],dtype=torch.float64)
    eta=torch.tensor([0.,0.,-5.],dtype=torch.float64)
    final=[];uncorrected=[]
    for x in states:
        y,qf,_=draw_direction(x,normals,limits,eta,generator=rng)
        assert y is not None and float(y[2])>=lower-1e-10
        qr,_=direction_log_prob(x,y,normals,limits,eta)
        ratio=beta*(y[2]-x[2])+qr-qf
        accepted=float(torch.rand((),dtype=torch.float64,generator=rng).log())<min(0.,float(ratio))
        final.append(float(y[2] if accepted else x[2]));uncorrected.append(float(y[2]))
    length=1-lower;expected=lower+length*(1/(-math.expm1(-beta*length))-1/(beta*length))
    assert abs(sum(final)/count-expected)<.03
    assert sum(uncorrected)/count<expected-.08


def test_molecular_distance_arcs_preserve_graph_and_transform_equivariantly():
    molecule=Chem.AddHs(Chem.MolFromSmiles('FCCF'))
    assert AllChem.EmbedMolecule(molecule,randomSeed=25321)==0
    x=torch.tensor(molecule.GetConformer().GetPositions(),dtype=torch.float64);x-=x.mean(0)
    numbers=[a.GetAtomicNum() for a in molecule.GetAtoms()];radii=covalent_radii(numbers)
    graph=infer_chemical_graph(x,numbers,0);root=(0,1);radius=(x[0]-x[1]).norm();base=(x[0]-x[1])/radius
    normals,limits=root_distance_constraints(x,root,radius,radii)
    eta=-400*base;rng=torch.Generator().manual_seed(25323)
    for i in range(64):
        u,logq,_=draw_direction(base,normals,limits,eta,generator=rng)
        assert u is not None and torch.isfinite(logq)
        y=x.clone();y[0]=x[1]+radius*u;y-=y.mean(0)
        actual=infer_chemical_graph(y,numbers,0)
        assert torch.equal(actual['bond_orders'],graph['bond_orders'])
        if i==0:
            orthogonal=torch.linalg.qr(torch.randn(3,3,dtype=torch.float64,generator=rng))[0]
            for reflected in [False,True]:
                matrix=orthogonal.clone()
                if reflected:matrix[:,0]*=-1
                newq,_=direction_log_prob(u@matrix,base@matrix,normals@matrix,limits,eta@matrix)
                assert abs(float(newq-logq))<1e-7
            permutation=torch.randperm(len(x),generator=rng);inverse=torch.argsort(permutation)
            nn,cc=root_distance_constraints(x[permutation],(int(inverse[0]),int(inverse[1])),radius,radii[permutation])
            newq,_=direction_log_prob(u,base,nn,cc,eta)
            assert abs(float(newq-logq))<1e-8


def test_near_pole_density_reorthogonalizes_without_weakening_frame_check():
    from scripts.research.audit_arc_teacher_support import independent_direction_logp
    base=torch.tensor([-.33645891068101025,-.8750305875573113,.348018494138536],dtype=torch.float64)
    observed=torch.tensor([-.33645891845714837,-.8750305351865997,.3480186182974966],dtype=torch.float64)
    normal=torch.empty(0,3,dtype=base.dtype);limit=torch.empty(0,dtype=base.dtype)
    perpendicular=observed-(base@observed)*base
    assert abs(float(base@(perpendicular/perpendicular.norm())))>1e-9
    eta=torch.tensor([50.,-40.,70.],dtype=base.dtype)
    value,trace=direction_log_prob(observed,base,normal,limit,eta)
    assert torch.isfinite(value) and abs(float(base@trace['tangent']))<1e-12
    assert abs(float(trace['tangent'].norm())-1)<1e-12
    independent=independent_direction_logp(observed.numpy(),base.numpy(),normal.numpy(),limit.numpy(),eta.numpy(),math.pi/64)
    assert abs(float(value)-independent)<1e-7
