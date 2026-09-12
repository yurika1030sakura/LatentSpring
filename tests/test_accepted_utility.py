import math
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from cfm_mol.accepted_utility import accepted_importance_utility
from cfm_mol.bounded_arc_guide import BoundedArcGuide
from cfm_mol.chemical_moves import covalent_radii,infer_chemical_graph
from cfm_mol.joint_arc_geometry import marginal_joint_arc_proposal,observed_joint_arc_density
from cfm_mol.joint_chemical_geometry import distinct_anchor_actions


def test_exact_finite_state_utility_and_gradient_include_the_reverse_density():
    logits=torch.tensor([[.4,-.7,.2],[.1,.8,-.3],[-.2,.5,.3]],dtype=torch.float64,requires_grad=True)
    behavior=torch.tensor([[.4,.3,.3],[.2,.5,.3],[.3,.4,.3]],dtype=logits.dtype)
    target=torch.tensor([.2,.3,.5],dtype=logits.dtype);source=torch.tensor([.7,.2,.1],dtype=logits.dtype)
    energy=-target.log();reward=energy[:,None]-energy[None,:]
    logq=logits.log_softmax(1);tr=target.log()[None,:]-target.log()[:,None]
    utility,weight=accepted_importance_utility(logq,logq.T,behavior.log(),tr,torch.zeros_like(tr),reward)
    estimated=(source[:,None]*behavior*utility).sum()
    acceptance=torch.minimum(torch.ones_like(logq),(tr+logq.T-logq).exp())
    direct=(source[:,None]*logq.exp()*acceptance*reward).sum()
    torch.testing.assert_close(estimated,direct,atol=1e-14,rtol=0)
    g=torch.autograd.grad(estimated,logits,retain_graph=True)[0]
    gd=torch.autograd.grad(direct,logits,retain_graph=True)[0]
    torch.testing.assert_close(g,gd,atol=1e-14,rtol=0)
    wrong,_=accepted_importance_utility(logq,logq.T.detach(),behavior.log(),tr,torch.zeros_like(tr),reward)
    bad=torch.autograd.grad((source[:,None]*behavior*wrong).sum(),logits)[0]
    assert (g-bad).abs().max()>.001
    h=1e-5
    def exact(z):
        q=z.softmax(1);alpha=torch.minimum(torch.ones_like(q),target[None,:]*q.T/(target[:,None]*q))
        return (source[:,None]*q*alpha*reward).sum()
    perturb=torch.zeros_like(logits);perturb[1,0]=h
    fd=(exact(logits.detach()+perturb)-exact(logits.detach()-perturb))/(2*h)
    assert abs(float(fd-g[1,0]))<1e-9


def test_action_count_ratio_zero_support_and_signed_rewards_are_preserved():
    qf=torch.tensor([.2,.4,0.],dtype=torch.float64);qr=torch.tensor([.1,.3,0.],dtype=qf.dtype)
    qb=torch.tensor([.3,.2,.2],dtype=qf.dtype);ratio=torch.tensor([.7,-.2,0.],dtype=qf.dtype)
    actions=torch.full_like(qf,math.log(3/2));reward=torch.tensor([-2.,1.,0.],dtype=qf.dtype)
    result,w=accepted_importance_utility(qf.log(),qr.log(),qb.log(),ratio,actions,reward)
    expected=reward*torch.minimum(qf/(qb),ratio.exp()*(qr/2)/(qb/3))
    torch.testing.assert_close(result,expected)
    assert result[0]<0 and w[2]==0
    with pytest.raises(ValueError,match='overflow'):
        accepted_importance_utility(qf.new_tensor([1000.]),qf.new_tensor([1000.]),qf.new_tensor([0.]),qf.new_tensor([0.]),qf.new_tensor([0.]),qf.new_tensor([1.]))


def molecular_pair():
    mol=Chem.AddHs(Chem.MolFromSmiles('FCCF'));assert AllChem.EmbedMolecule(mol,randomSeed=25601)==0
    x=torch.tensor(mol.GetConformer().GetPositions(),dtype=torch.float64);x-=x.mean(0)
    z=torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()]);radii=covalent_radii(z)
    bonds=infer_chemical_graph(x,z.tolist(),0)['bond_orders'];e=x.new_tensor([0.,1.,.026])
    action=distinct_anchor_actions(z,bonds)[0];i,j,k,l=action;inverse=(i,j,l,k)
    rng=torch.Generator().manual_seed(25603)
    for _ in range(24):
        y,q,t=marginal_joint_arc_proposal(x,bonds,z,e,radii,action,kind='arc_site',order=0,generator=rng)
        if y is None:continue
        new=infer_chemical_graph(y,z.tolist(),0)['bond_orders']
        qr,_=observed_joint_arc_density(y,x,new,z,e,radii,inverse,kind='arc_site')
        if torch.isfinite(qr):return x,y,bonds,new,z,e,radii,action,inverse,q,qr
    raise AssertionError('No bidirectionally supported test pair')


def test_observed_joint_gradient_and_exact_physical_initialization():
    x,y,bonds,new,z,e,radii,action,inverse,base_q,base_qr=molecular_pair()
    torch.manual_seed(25605);model=BoundedArcGuide(log_score_bound=.5).double()
    q,_=observed_joint_arc_density(x,y,bonds,z,e,radii,action,kind='arc_bounded',model=model)
    qr,_=observed_joint_arc_density(y,x,new,z,e,radii,inverse,kind='arc_bounded',model=model)
    torch.testing.assert_close(q,base_q,atol=1e-10,rtol=0);torch.testing.assert_close(qr,base_qr,atol=1e-10,rtol=0)
    assert q.requires_grad and qr.requires_grad
    with torch.no_grad():model.query_head[-1].weight.normal_(0,.2)
    q,_=observed_joint_arc_density(x,y,bonds,z,e,radii,action,kind='arc_bounded',model=model)
    qr,_=observed_joint_arc_density(y,x,new,z,e,radii,inverse,kind='arc_bounded',model=model)
    loss=q+.37*qr;weight=model.query_head[-1].weight;gradient=torch.autograd.grad(loss,weight)[0]
    flat=int(gradient.abs().argmax());i,j=flat//weight.shape[1],flat%weight.shape[1];h=1e-5
    def value():
        a,_=observed_joint_arc_density(x,y,bonds,z,e,radii,action,kind='arc_bounded',model=model)
        b,_=observed_joint_arc_density(y,x,new,z,e,radii,inverse,kind='arc_bounded',model=model)
        return float(a+.37*b)
    with torch.no_grad():weight[i,j]+=h
    a=value()
    with torch.no_grad():weight[i,j]-=2*h
    b=value()
    with torch.no_grad():weight[i,j]+=h
    assert abs((a-b)/(2*h)-float(gradient[i,j]))<1e-7
    _,production,_=marginal_joint_arc_proposal(x,bonds,z,e,radii,action,kind='arc_bounded',order=1,model=model,observed=y)
    assert not production.requires_grad
    from scripts.research.audit_joint_arc_support import independent_arc_q
    _,_,trace=marginal_joint_arc_proposal(x,bonds,z,e,radii,action,kind='arc_bounded',order=1,model=model,observed=y)
    assert abs(independent_arc_q(trace)-float(production))<1e-7
    torch.testing.assert_close(production,q.detach(),atol=1e-10,rtol=0)
    with pytest.raises(ValueError,match='semantics'):
        observed_joint_arc_density(x,y,bonds,z,e,radii,action,kind='arc_energy',model=model)


def test_bounded_score_controls_the_complete_marginal_joint_density_ratio():
    x,y,bonds,new,z,e,radii,action,inverse,q0,qr0=molecular_pair()
    torch.manual_seed(25607);bound=.5;model=BoundedArcGuide(log_score_bound=bound).double()
    with torch.no_grad():model.query_head[-1].weight.normal_(0,8.)
    for source,end,graph,move,reference in [(x,y,bonds,action,q0),(y,x,new,inverse,qr0)]:
        q,_=observed_joint_arc_density(source,end,graph,z,e,radii,move,kind='arc_bounded',model=model)
        assert abs(float(q-reference))<=4*bound+1e-10


def test_stratified_index_sampling_preserves_all_attempt_empirical_objective():
    from scripts.research.train_accepted_utility import population_weights
    records=[dict(index=1,parent=10,replica=0),dict(index=1,parent=10,replica=0),
        dict(index=1,parent=20,replica=0),dict(index=2,parent=30,replica=0)]
    weights=population_weights(records)
    torch.testing.assert_close(weights,torch.tensor([.125,.125,.25,.5],dtype=torch.float64))
    value=torch.tensor([0.,.4,-.1,.2],dtype=torch.float64,requires_grad=True)
    selection=.5*weights+.5*(weights*value.detach().abs())/(weights*value.detach().abs()).sum()
    exhaustive=(selection*(weights/selection)*value).sum();direct=(weights*value).sum()
    torch.testing.assert_close(exhaustive,direct,atol=1e-14,rtol=0)
    torch.testing.assert_close(torch.autograd.grad(exhaustive,value)[0],weights,atol=1e-14,rtol=0)
