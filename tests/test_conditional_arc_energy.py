import math
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from cfm_mol.chemical_moves import infer_chemical_graph
from cfm_mol.conditional_arc_energy import ConditionalArcEnergy
from cfm_mol.geodesic_arc_proposal import circle_law,direction_log_prob,draw_direction


def fixture():
    mol=Chem.AddHs(Chem.MolFromSmiles('FCCF'));assert AllChem.EmbedMolecule(mol,randomSeed=25501)==0
    x=torch.tensor(mol.GetConformer().GetPositions(),dtype=torch.float64);x-=x.mean(0)
    numbers=torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()])
    bonds=infer_chemical_graph(x,numbers.tolist(),0)['bond_orders']
    return x,bonds,numbers,torch.tensor([[0,1]]),torch.tensor([0.,1.,.026],dtype=torch.float64)


def test_scalar_energy_masks_direction_and_obeys_o3_permutation_translation():
    x,bonds,z,roots,e=fixture();torch.manual_seed(25503)
    model=ConditionalArcEnergy().double()
    with torch.no_grad():model.query_head[-1].weight.normal_(0,.2)
    u=torch.randn(1,7,3,dtype=x.dtype);u/=u.norm(dim=-1,keepdim=True)
    c=model.encode(x[None],bonds[None],z,e,roots);value=model.energy(u,c)
    y=x.clone();radius=(x[0]-x[1]).norm();y[0]=x[1]+radius*u[0,3];y-=y.mean(0)
    other=model.encode(y[None],bonds[None],z,e,roots)
    for key in c:torch.testing.assert_close(c[key],other[key],atol=1e-9,rtol=1e-9)
    permutation=torch.randperm(len(x));inverse=torch.argsort(permutation)
    q=torch.linalg.qr(torch.randn(3,3,dtype=x.dtype))[0];q[:,0]*=-torch.linalg.det(q)
    transformed=model((x[permutation]@q+torch.tensor([1.,2.,3.]))[None],bonds[permutation][:,permutation][None],
        z[permutation],e,inverse[roots],u@q)
    torch.testing.assert_close(value,transformed,atol=1e-9,rtol=1e-9)


def test_energy_direction_derivative_and_force_parameter_gradient_match_finite_difference():
    x,bonds,z,roots,e=fixture();torch.manual_seed(25505)
    model=ConditionalArcEnergy().double()
    with torch.no_grad():model.query_head[-1].weight.normal_(0,.3)
    u=torch.tensor([[[.3,.4,math.sqrt(.75)]]],dtype=x.dtype,requires_grad=True)
    context=model.encode(x[None],bonds[None],z,e,roots)
    value=model.energy(u,context);gradient=torch.autograd.grad(value.sum(),u,create_graph=True)[0]
    axis=torch.tensor([[[.7,-.2,.1]]],dtype=x.dtype);axis-=((axis*u).sum(-1,keepdim=True))*u;axis/=axis.norm()
    h=1e-5
    plus=u.detach()*math.cos(h)+axis.detach()*math.sin(h)
    minus=u.detach()*math.cos(h)-axis.detach()*math.sin(h)
    fd=(model.energy(plus,context)-model.energy(minus,context))/(2*h)
    torch.testing.assert_close(fd,(gradient*axis).sum(-1),atol=1e-7,rtol=1e-7)
    parameter=model.query_head[-1].weight
    loss=(gradient*axis).sum().square();g=torch.autograd.grad(loss,parameter)[0]
    def force_loss():
        uu=u.detach().requires_grad_();v=model(x[None],bonds[None],z,e,roots,uu)
        gg=torch.autograd.grad(v.sum(),uu)[0];return float((gg*axis.detach()).sum().square())
    with torch.no_grad():parameter[0,0]+=h
    a=force_loss()
    with torch.no_grad():parameter[0,0]-=2*h
    b=force_loss()
    with torch.no_grad():parameter[0,0]+=h
    assert abs((a-b)/(2*h)-float(g[0,0]))<1e-6


def test_nonlinear_circle_score_has_exact_interpolated_mass_and_gradient():
    base=torch.tensor([0.,0.,1.],dtype=torch.float64);tangent=torch.tensor([1.,0.,0.],dtype=torch.float64)
    normals=torch.empty(0,3,dtype=base.dtype);limits=torch.empty(0,dtype=base.dtype);eta=base*0
    parameter=torch.tensor(3.,dtype=base.dtype,requires_grad=True)
    score=lambda u:parameter*u[...,0].square()+.7*u[...,2]
    law=circle_law(base,tangent,normals,limits,eta,score=score)
    nodes,weights=np.polynomial.legendre.leggauss(48);t=torch.tensor((nodes+1)/2,dtype=base.dtype)
    masses=[]
    for (lo,hi),heights in zip(law['edges'],law['heights']):
        values=torch.exp(heights[0]+(heights[1]-heights[0])*t)
        masses.append((hi-lo)*(values*torch.tensor(weights,dtype=base.dtype)).sum()/2)
    independent=torch.stack(masses).sum().log()
    torch.testing.assert_close(independent,law['log_normalizer'],atol=1e-12,rtol=0)
    y=torch.tensor([.6,0.,.8],dtype=base.dtype)
    value,_=direction_log_prob(y,base,normals,limits,eta,score=score)
    g=torch.autograd.grad(value,parameter)[0];h=1e-5
    def at(p):return direction_log_prob(y,base,normals,limits,eta,score=lambda u:p*u[...,0].square()+.7*u[...,2])[0]
    assert abs(float((at(3+h)-at(3-h))/(2*h)-g))<1e-8
    # Sampling uses the same nonlinear law as density evaluation.
    rng=torch.Generator().manual_seed(25507)
    for _ in range(16):
        drawn,q,_=draw_direction(base,normals,limits,eta,generator=rng,score=score)
        observed,_=direction_log_prob(drawn,base,normals,limits,eta,score=score)
        torch.testing.assert_close(q,observed)


def test_complete_scalar_joint_density_reversal_and_public_query_ledger():
    from cfm_mol.chemical_moves import covalent_radii
    from cfm_mol.chemical_sampler import ChemicalTarget
    from cfm_mol.joint_arc_geometry import marginal_joint_arc_proposal
    from cfm_mol.joint_chemical_geometry import distinct_anchor_actions,joint_chemical_transition
    from scripts.research.audit_joint_arc_support import independent_arc_q,independent_energy_log_score
    x,bonds,z,roots,e=fixture();torch.manual_seed(25541)
    model=ConditionalArcEnergy().double()
    with torch.no_grad():model.query_head[-1].weight.normal_(0,.15)
    radii=covalent_radii(z);rng=torch.Generator().manual_seed(25543);checked=0
    action=distinct_anchor_actions(z,bonds)[0];i,j,k,l=action
    for order in [0,1]:
        for _ in range(8):
            y,q,trace=marginal_joint_arc_proposal(x,bonds,z,e,radii,action,kind='arc_energy',order=order,model=model,generator=rng)
            if y is None:continue
            new=infer_chemical_graph(y,z.tolist(),0)['bond_orders']
            _,qr,reverse=marginal_joint_arc_proposal(y,new,z,e,radii,(i,j,l,k),kind='arc_energy',order=order,model=model,observed=x)
            assert abs(independent_arc_q(trace)-float(q))<1e-7
            if torch.isfinite(qr):assert abs(independent_arc_q(reverse)-float(qr))<1e-7
            for component in trace['components']:
                if component['failed']:continue
                for step in component['steps']:
                    u=step['direction'];root=torch.tensor([step['root']])
                    actual=-model(step['context'][None],component['desired_bonds'][None],z,e,root,u[None])/e[2]
                    assert abs(float(actual)-float(independent_energy_log_score(u.numpy(),step['energy_score'])))<1e-8
            checked+=1
    assert checked>=8
    class Oracle:
        evaluated=0
        def evaluate_chunked(self,positions,max_request):
            self.evaluated+=len(positions);return .01*positions.square().sum((1,2)),-.02*positions
    target=ChemicalTarget(Oracle(),dict(numbers=z.tolist(),charge=0,spin_multiplicity=1),.026,.1)
    states=target.evaluate([target.coordinate_state(x),target.coordinate_state(-x)],phase='initial')
    total=0
    for _ in range(8):
        before=target.oracle.evaluated
        states,rows=joint_chemical_transition(target,states,kind='arc_energy',generator=rng,phase='scalar',model=model,site_concentration=64.)
        assert target.oracle.evaluated-before==2*sum(r['valid'] for r in rows)
        total+=sum(r['valid'] for r in rows)
        for row in rows:
            if not row['valid']:continue
            old=target.states[row['old_state_id']];new=target.states[row['new_state_id']]
            ratio=-float(new['potential_eV']-old['potential_eV'])/target.kT+independent_arc_q(row['reverse'])-independent_arc_q(row['forward'])+row['action_log_ratio']
            assert abs(ratio-row['log_acceptance_ratio'])<1e-7
            assert row['accepted']==(row['log_uniform']<min(0.,ratio))
    assert total>=8
