import math
import pytest
import torch

from cfm_mol.bounded_action_geometry import (
    BoundedConditionalActionPolicy, ActionGeometryGuide, full_observed_densities, edge_values)
from cfm_mol.accepted_utility import accepted_importance_utility
from cfm_mol.joint_chemical_geometry import distinct_anchor_actions
from test_chemical_policy import example
from test_accepted_utility import molecular_pair


def test_conditional_policy_normalization_empty_symmetries_and_bounds():
    model = BoundedConditionalActionPolicy(logit_bound=.5).double()
    x, b, z, e, a = example()
    torch.testing.assert_close(model(x,b,z,e,a).exp(), x.new_full((2,3),1/3))
    assert not hasattr(model, 'family_head')
    assert model(x,b,z,e,a[:,:0]).shape == (2,0)
    mask = torch.tensor([[True,False,True],[False,False,False]])
    torch.testing.assert_close(model(x,b,z,e,a,mask).exp(), x.new_tensor([[.5,0,.5],[0,0,0]]))
    with torch.no_grad():
        model.action_head[-1].weight.normal_(0,3)
    expected = model(x,b,z,e,a)
    assert float((expected+math.log(3)).abs().max()) <= 1+1e-12
    q = torch.linalg.qr(torch.randn(3,3,dtype=x.dtype))[0]; q[:,0]*=-1
    torch.testing.assert_close(model(x@q+7,b,z,e,a),expected,atol=1e-10,rtol=0)
    perm=torch.tensor([3,0,5,2,1,4]);inv=torch.argsort(perm)
    torch.testing.assert_close(model(x[:,perm],b[:,perm][:,:,perm],z[perm],e,inv[a]),expected)
    torch.testing.assert_close(model(x,b,z,e,a[:,:,[1,0,3,2]]),expected)
    torch.testing.assert_close(model(x,b,z,e,a[:,[2,0,1]]),expected[:,[2,0,1]])
    torch.testing.assert_close(model(x[:1],b[:1],z,e,a[:1]),expected[:1])


def test_full_density_identity_with_unequal_action_counts_and_reverse_gradients():
    # Three-state path: state 0 has one action, state 1 has two, state 2 one.
    pairs=[(0,1,0,0),(1,0,0,0),(1,2,1,0),(2,1,0,1)]
    logits=torch.tensor([[.2,-.7],[.4,-.3],[-.5,.1]],dtype=torch.float64,requires_grad=True)
    mask=torch.tensor([[True,False],[True,True],[True,False]])
    logp=logits.masked_fill(~mask,-torch.inf).log_softmax(1)
    pi=logits.new_tensor([.2,.3,.5]);source=logits.new_tensor([.6,.3,.1])
    qf=torch.stack([logp[x,a] for x,y,a,ar in pairs])
    qr=torch.stack([logp[y,ar] for x,y,a,ar in pairs])
    qb=logits.new_tensor([1,.5,.5,1]);tr=torch.stack([pi[y].log()-pi[x].log() for x,y,a,ar in pairs])
    reward=torch.stack([pi[y].log()-pi[x].log() for x,y,a,ar in pairs])
    mass=torch.stack([source[x] for x,y,a,ar in pairs])*qb
    utility,_=accepted_importance_utility(qf,qr,qb.log(),tr,torch.zeros_like(tr),reward)
    result=(mass*utility).sum()
    direct=(mass/qb*qf.exp()*torch.minimum(torch.ones_like(tr),(tr+qr-qf).exp())*reward).sum()
    torch.testing.assert_close(result,direct,atol=1e-14,rtol=0)
    actual=torch.autograd.grad(result,logits,retain_graph=True)[0]
    torch.testing.assert_close(actual,torch.autograd.grad(direct,logits,retain_graph=True)[0])
    wrong,_=accepted_importance_utility(qf,qr,qb.log(),tr,logits.new_tensor([math.log(.5),math.log(2),math.log(2),math.log(.5)]),reward)
    assert abs(float((mass*wrong).sum()-result))>1e-3


def pair_record():
    x,y,b,new,z,e,radii,a,ar,q,qr=molecular_pair()
    nf=len(distinct_anchor_actions(z,b));nr=len(distinct_anchor_actions(z,new))
    return dict(valid=True,x=x,y=y,bonds=b,new_bonds=new,numbers=z,electronic=e,radii=radii,
        action=a,inverse_action=ar,log_behavior_forward=q,log_behavior_reverse=qr,
        action_log_ratio=x.new_tensor(math.log(nf/nr)),target_log_ratio=x.new_tensor(.37),
        reward_eV=x.new_tensor(.1),connectivity_changed=True)


@pytest.mark.parametrize('variant',['action','geometry','joint'])
def test_actual_pair_initialization_bounds_and_parameter_gradients(variant):
    row=pair_record();torch.manual_seed(25703);model=ActionGeometryGuide(variant).double()
    assert model.log_ratio_bound==2
    qf,qr,bf,br=full_observed_densities(model,row,{})
    torch.testing.assert_close(qf,bf,atol=1e-10,rtol=0);torch.testing.assert_close(qr,br,atol=1e-10,rtol=0)
    baseline,_=accepted_importance_utility(row['log_behavior_forward'],row['log_behavior_reverse'],
        row['log_behavior_forward'],row['target_log_ratio'],row['action_log_ratio'],row['reward_eV'])
    torch.testing.assert_close(edge_values(model,row,{})[0],baseline,atol=1e-11,rtol=0)
    heads=[]
    if variant!='geometry':heads.append(model.action.action_head[-1].weight)
    if variant!='action':heads.append(model.geometry.query_head[-1].weight)
    with torch.no_grad():
        for weight in heads:weight.normal_(0,.3)
    def objective():
        u,c,_,_,penalty,_=edge_values(model,row,{})
        return -u+.006*c+.005*penalty
    loss=objective();grads=torch.autograd.grad(loss,heads)
    for weight,g in zip(heads,grads):
        idx=int(g.abs().argmax());ij=(idx//weight.shape[1],idx%weight.shape[1]);h=1e-5
        assert abs(float(g[ij]))>1e-10
        with torch.no_grad():weight[ij]+=h
        up=float(objective())
        with torch.no_grad():weight[ij]-=2*h
        down=float(objective())
        with torch.no_grad():weight[ij]+=h
        assert abs((up-down)/(2*h)-float(g[ij]))<1e-7
    qf,qr,bf,br=full_observed_densities(model,row,{})
    assert max(abs(float(qf-bf)),abs(float(qr-br)))<=2+1e-10
    assert all(float(v)==0 for v in edge_values(model,{'valid':False},{}))


@pytest.mark.parametrize('variant',['action','joint'])
def test_actual_transition_uses_selected_forward_and_inverse_probabilities(variant):
    from cfm_mol.chemical_sampler import ChemicalTarget
    from cfm_mol.joint_chemical_geometry import joint_chemical_transition
    from scripts.research.audit_masked_angular import ReplayOracle,equal
    row=pair_record();model=ActionGeometryGuide(variant).double()
    with torch.no_grad():
        model.action.action_head[-1].weight.normal_(0,.2)
        if variant=='joint':model.geometry.query_head[-1].weight.normal_(0,.2)
    class Oracle:
        evaluated=0
        def evaluate_chunked(self,p,max_request):
            self.evaluated+=len(p);return .01*p.square().sum((1,2)),-.02*p
    def run(oracle):
        target=ChemicalTarget(oracle,dict(numbers=row['numbers'].tolist(),charge=0,spin_multiplicity=1),.026,.1)
        old=target.evaluate([target.coordinate_state(row['x'])],phase='initial')
        states,rows=joint_chemical_transition(target,old*12,kind='arc_site' if variant=='action' else 'arc_bounded',
            model=None if variant=='action' else model.geometry,generator=torch.Generator().manual_seed(25711),
            phase='test',site_concentration=64.,action_policy=model.action)
        return target,states,rows
    target,states,rows=run(Oracle())
    valid=[r for r in rows if r['valid']];assert valid
    for r in valid:
        ratio=float(r['log_reverse_action']-r['log_forward_action'])
        assert abs(r['action_log_ratio']-ratio)<1e-12
        assert abs(r['log_acceptance_ratio']-(r['target_log_ratio']+float(r['coordinate_log_ratio'])+ratio))<1e-12
        from scripts.research.audit_action_geometry import independent_action_logp
        from scripts.research.audit_joint_arc_support import independent_arc_q
        old=target.states[r['old_state_id']];new=target.states[r['new_state_id']]
        af,pf=independent_action_logp(model.action,old['positions'],old['graph']['bond_orders'],row['numbers'],row['electronic'])
        ar,pr=independent_action_logp(model.action,new['positions'],new['graph']['bond_orders'],row['numbers'],row['electronic'])
        action_ratio=pr[ar.index(tuple(r['inverse_action']))]-pf[af.index(tuple(r['action']))]
        independent=r['target_log_ratio']+independent_arc_q(r['reverse'])-independent_arc_q(r['forward'])+action_ratio
        assert abs(independent-r['log_acceptance_ratio'])<1e-7
    oracle=ReplayOracle(target.query_trace);other,states2,rows2=run(oracle)
    equal(rows,rows2);equal(states,states2)
    assert oracle.index==len(oracle.queries)
