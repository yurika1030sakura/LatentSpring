import math
import pytest
import itertools
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from cfm_mol.chemical_moves import covalent_radii,infer_chemical_graph
from cfm_mol.chemical_path_guide import exchanged_bond_graph
from cfm_mol.chemical_sampler import ChemicalTarget
from cfm_mol.joint_arc_geometry import joint_arc_proposal,marginal_joint_arc_proposal
from cfm_mol.joint_chemical_geometry import joint_chemical_transition,radial_cartesian_log_density
from cfm_mol.local_site_guide import LocalSiteGuide
from cfm_mol.spherical_proposal import vmf_sample,vmf_log_prob
from scripts.research.audit_arc_teacher_support import independent_direction_logp


def fixture():
    mol=Chem.AddHs(Chem.MolFromSmiles('FCCF'))
    assert AllChem.EmbedMolecule(mol,randomSeed=25401)==0
    x=torch.tensor(mol.GetConformer().GetPositions(),dtype=torch.float64);x-=x.mean(0)
    numbers=torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()],dtype=torch.long)
    graph=infer_chemical_graph(x,numbers.tolist(),0);bonds=graph['bond_orders']
    leaf=next(i for i,z in enumerate(numbers) if int(z)==1 and bonds[i,2]>0)
    return x,bonds,numbers,(0,leaf,1,2)


def test_joint_arc_density_graph_reversal_and_full_symmetries():
    x,bonds,numbers,action=fixture();radii=covalent_radii(numbers)
    electronic=torch.tensor([0.,1.,.026],dtype=torch.float64)
    model=LocalSiteGuide(bound=2048.,site_concentration=64.).double()
    with torch.no_grad():model.center_weight.weight.normal_(0,.1)
    rng=torch.Generator().manual_seed(25403);checked=0;positive_reverse=0
    for kind in ['arc_uniform','arc_site','arc_site_confinement','arc_model']:
        for order in [0,1]:
            for _ in range(8):
                kw=dict(kind=kind,order=order,model=model,site_concentration=64.,radial_width=.05)
                y,q,trace=joint_arc_proposal(x,bonds,numbers,electronic,radii,action,generator=rng,**kw)
                if y is None:
                    assert trace['failed'] and not torch.isfinite(q);continue
                new_bonds=exchanged_bond_graph(bonds,action)
                assert torch.equal(infer_chemical_graph(y,numbers.tolist(),0)['bond_orders'],new_bonds)
                recovered,qeval,_=joint_arc_proposal(x,bonds,numbers,electronic,radii,action,observed=y,**kw)
                torch.testing.assert_close(recovered,y);torch.testing.assert_close(qeval,q)
                independent=float(radial_cartesian_log_density(trace['log_radii'],trace['radial_means'],.05).sum())
                for step in trace['steps']:
                    independent+=independent_direction_logp(step['direction'].numpy(),step['base_direction'].numpy(),
                        step['normals'].numpy(),step['limits'].numpy(),step['eta'].numpy(),trace['max_segment_width'])
                assert abs(independent-float(q))<1e-7
                i,j,k,l=action
                back,qr,_=joint_arc_proposal(y,new_bonds,numbers,electronic,radii,(i,j,l,k),observed=x,**kw)
                if torch.isfinite(qr):
                    torch.testing.assert_close(back,x);positive_reverse+=1
                permutation=torch.randperm(len(x),generator=rng);inverse=torch.argsort(permutation)
                aa=tuple(int(inverse[a]) for a in action);swapped=aa[0]>aa[1]
                if swapped:aa=(aa[1],aa[0],aa[3],aa[2])
                orthogonal=torch.linalg.qr(torch.randn(3,3,dtype=x.dtype,generator=rng))[0]
                orthogonal[:,0]*=-torch.linalg.det(orthogonal)
                _,qt,_=joint_arc_proposal(x[permutation]@orthogonal,bonds[permutation][:,permutation],numbers[permutation],
                    electronic,radii[permutation],aa,observed=y[permutation]@orthogonal,
                    **dict(kw,order=1-order if swapped else order))
                torch.testing.assert_close(qt,q,atol=1e-7,rtol=1e-9);checked+=1
    assert checked>=20 and positive_reverse>=10


def test_order_marginal_density_dominates_retained_order_probability_flow():
    x,bonds,numbers,action=fixture();radii=covalent_radii(numbers)
    electronic=torch.tensor([0.,1.,.026],dtype=x.dtype);rng=torch.Generator().manual_seed(25405)
    checked=0
    for order in [0,1]:
        for _ in range(12):
            kw=dict(kind='arc_uniform',order=order,site_concentration=64.)
            y,q,trace=marginal_joint_arc_proposal(x,bonds,numbers,electronic,radii,action,generator=rng,**kw)
            if y is None:continue
            _,qe,te=marginal_joint_arc_proposal(x,bonds,numbers,electronic,radii,action,observed=y,**kw)
            torch.testing.assert_close(q,qe)
            i,j,k,l=action;new_bonds=exchanged_bond_graph(bonds,action)
            _,qr,reverse=marginal_joint_arc_proposal(y,new_bonds,numbers,electronic,radii,(i,j,l,k),observed=x,**kw)
            f=trace['order_log_densities'];r=reverse.get('order_log_densities',x.new_full((2,),-torch.inf))
            target_ratio=x.new_tensor(-.7)
            # Scaled probability flow avoids overflow; this tests the actual
            # component densities, including asymmetric zero-support orders.
            scale=torch.maximum(f.max(),(r+target_ratio).max())
            retained=.5*torch.minimum((f-scale).exp(),(r+target_ratio-scale).exp()).sum()
            marginal=torch.minimum((q-scale).exp(),(qr+target_ratio-scale).exp())
            assert float(marginal-retained)>-1e-10
            checked+=1
    assert checked>=10


def test_two_root_geodesic_cartesian_chart_has_all_volume_factors():
    x,_,_,(i,j,k,l)=fixture();n=len(x)
    bases=[(x[j]-x[l]),(x[i]-x[k])];bases=[b/b.norm() for b in bases]
    tangents=[]
    for b in bases:
        a=torch.tensor([.7,-.3,1.],dtype=x.dtype);a-=a@b*b;a/=a.norm()
        tangents.append((a,torch.cross(b,a,dim=0)))
    def transform(z):
        y=x.clone()
        for root,anchor,base,(a,b),offset in zip([i,j],[l,k],bases,tangents,[0,3]):
            ell,phi,theta=z[offset:offset+3]
            tangent=phi.cos()*a+phi.sin()*b
            y[root]=x[anchor]+ell.exp()*(theta.cos()*base+theta.sin()*tangent)
        return (y-y.mean(0)).flatten()
    z=torch.tensor([.1,.7,.9,-.1,1.1,.6],dtype=x.dtype)
    jac=torch.autograd.functional.jacobian(transform,z)
    actual=torch.linalg.det(jac.T@jac).sqrt()
    expected=((n-2)/n)**1.5*(3*(z[0]+z[3])).exp()*z[2].sin().abs()*z[5].sin().abs()
    torch.testing.assert_close(actual,expected,atol=1e-10,rtol=1e-10)


def test_public_joint_transition_preserves_graph_and_charges_only_scored_endpoints():
    x,bonds,numbers,_=fixture()
    class Oracle:
        evaluated=0
        def evaluate_chunked(self,x,max_request):
            self.evaluated+=len(x);return .1*x.square().sum((1,2)),-.2*x
    oracle=Oracle();target=ChemicalTarget(oracle,dict(numbers=numbers.tolist(),charge=0,spin_multiplicity=1),.026,.1)
    states=target.evaluate([target.coordinate_state(x),target.coordinate_state(-x)],phase='initial')
    rng=torch.Generator().manual_seed(25407);scored=0
    for step in range(32):
        before=oracle.evaluated
        new,rows=joint_chemical_transition(target,states,kind='arc_uniform',generator=rng,phase=f'step_{step}',site_concentration=64.)
        assert oracle.evaluated-before==2*sum(r['valid'] for r in rows)
        for old,end,row in zip(states,new,rows):
            if row['valid']:
                scored+=1
                candidate=target.states[row['new_state_id']]
                assert torch.equal(candidate['graph']['bond_orders'],exchanged_bond_graph(old['graph']['bond_orders'],row['action']))
                expected=-float(candidate['potential_eV']-old['potential_eV'])/target.kT+float(row['log_reverse_coordinate']-row['log_forward_coordinate'])+row['action_log_ratio']
                assert abs(expected-row['log_acceptance_ratio'])<1e-8
            else:assert end['state_id']==old['state_id'] and not row['accepted']
        states=new
    assert scored>10


@pytest.mark.parametrize('kind',['arc_uniform','arc_energy'])
def test_complete_joint_kernel_preserves_exact_geometry_conditioned_target(kind):
    # Exact joint rejection sampling chooses an assignment BEFORE each trial;
    # it never normalizes each graph separately. Both graph populations and all
    # six terminal Cartesian vectors therefore start from the stated target.
    x,bonds,numbers,_=fixture();radii=covalent_radii(numbers)
    anchors=torch.tensor([1,2]);leaves=torch.tensor([i for i in range(len(x)) if i not in [1,2]])
    assignments=[]
    for chosen in itertools.combinations(range(6),3):
        assignment=torch.ones(6,dtype=torch.long);assignment[list(chosen)]=0;assignments.append(assignment)
    assignments=torch.stack(assignments);rmean=(radii[leaves]+radii[1]).log();sigma=.12;concentration=.5
    rng=torch.Generator().manual_seed(25419);collected=[];total=0
    away=(x[1]-x[2]);away/=away.norm();directions=torch.stack([away,-away])
    off=~torch.eye(len(x),dtype=torch.bool);sizes=radii[:,None]+radii[None,:]
    for _ in range(32):
        batch=8192;labels=torch.randint(len(assignments),(batch,),generator=rng);slots=assignments[labels]
        ell=rmean+sigma*torch.randn(batch,6,dtype=x.dtype,generator=rng)
        u,_=vmf_sample((concentration*directions[slots]).reshape(-1,3),generator=rng);u=u.reshape(batch,6,3)
        poses=x.expand(batch,-1,-1).clone()
        poses[:,leaves]=x[anchors[slots]]+ell.exp()[...,None]*u;poses-=poses.mean(1,keepdim=True)
        distance=(poses[:,:,None]-poses[:,None,:]).square().sum(3)
        expected=torch.zeros(batch,len(x),len(x),dtype=torch.bool);expected[:,1,2]=True;expected[:,2,1]=True
        ids=torch.arange(batch)[:,None].expand(-1,6)
        expected[ids,leaves[None].expand(batch,-1),anchors[slots]]=True
        expected[ids,anchors[slots],leaves[None].expand(batch,-1)]=True
        actual=(distance<=(1.25*sizes).square())&off
        okay=(actual==expected).all((1,2))&~((distance<(.6*sizes).square())&off).any((1,2))
        if okay.any():collected.append(poses[okay]);total+=int(okay.sum())
        if total>=512:break
    assert total>=512
    initial=torch.cat(collected)[:512]
    def logtarget(poses):
        vectors=poses[:,leaves,None,:]-poses[:,anchors][:,None,:,:]
        distance=vectors.square().sum(3).sqrt();slot=distance.argmin(2)
        selected=vectors.gather(2,slot[:,:,None,None].expand(-1,-1,1,3))[:,:,0]
        radius=selected.norm(dim=2);unit=selected/radius[:,:,None]
        axis=poses[:,1]-poses[:,2];axis=axis/axis.norm(dim=1,keepdim=True)
        mu=torch.stack([axis,-axis],1).gather(1,slot[:,:,None].expand(-1,-1,3))
        return (radial_cartesian_log_density(radius.log(),rmean,sigma)+concentration*(mu*unit).sum(2)).sum(1)
    class Oracle:
        evaluated=0
        def evaluate_chunked(self,positions,max_request):
            self.evaluated+=len(positions)
            with torch.enable_grad():
                z=positions.clone().requires_grad_();energy=-.026*logtarget(z)
                force=-torch.autograd.grad(energy.sum(),z)[0]
            return energy.detach(),force.detach()
    target=ChemicalTarget(Oracle(),dict(numbers=numbers.tolist(),charge=0,spin_multiplicity=1),.026,0.)
    states=target.evaluate([target.coordinate_state(z) for z in initial],phase='exact_initial')
    model=None
    if kind=='arc_energy':
        from cfm_mol.conditional_arc_energy import ConditionalArcEnergy
        # Use the known target's broad scale so this stationarity check has
        # accepted movement; the separate real-kernel density test uses64.
        torch.manual_seed(25547);model=ConditionalArcEnergy(restraint=0.,site_concentration=concentration).double()
        with torch.no_grad():model.query_head[-1].weight.normal_(0,.1)
    final,rows=joint_chemical_transition(target,states,kind=kind,generator=rng,phase='joint',site_concentration=concentration if model else 64.,model=model)
    assert sum(r['accepted'] for r in rows)>=20
    final=torch.stack([s['positions'] for s in final])
    def observables(poses):
        v=poses[:,leaves,None,:]-poses[:,anchors][:,None,:,:]
        distance=v.norm(dim=3);slot=distance.argmin(2)
        vv=v.gather(2,slot[:,:,None,None].expand(-1,-1,1,3))[:,:,0]
        rr=vv.norm(dim=2);axis=poses[:,1]-poses[:,2];axis/=axis.norm(dim=1,keepdim=True)
        mu=torch.stack([axis,-axis],1).gather(1,slot[:,:,None].expand(-1,-1,3))
        fluorines=(numbers[leaves]==9)[None]
        count=((slot==0)&fluorines).sum(1)
        return torch.stack([rr.log().mean(1),rr.square().mean(1),((vv/rr[:,:,None])*mu).sum(2).mean(1),(count!=1).double()],1)
    change=observables(final)-observables(initial)
    stderr=change.std(0)/math.sqrt(len(change))
    assert bool((change.mean(0).abs()<5*stderr+1e-3).all())
