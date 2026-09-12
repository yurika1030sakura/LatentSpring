import math
import pytest
import torch

from cfm_mol.delayed_acceptance import (
    delayed_log_acceptance, paired_screen_features, DelayedAcceptanceScreen, screen_edge_values)
from test_bounded_action_geometry import pair_record


def test_two_stage_balance_cost_objective_and_gradients():
    dtype=torch.float64
    pi=torch.tensor([.2,.3,.5],dtype=dtype)
    q=torch.tensor([[.1,.6,.3],[.4,.2,.4],[.2,.5,.3]],dtype=dtype)
    source=torch.tensor([.6,.3,.1],dtype=dtype)
    theta=torch.tensor([[0.,.3,-.7],[.1,0.,.5],[-.2,.8,0.]],dtype=dtype,requires_grad=True)
    ratio=pi.log()[None,:]-pi.log()[:,None]+q.T.log()-q.log()
    def objective(t):
        factor=2*torch.tanh((t-t.T)/2)
        first,second,total=delayed_log_acceptance(ratio,factor)
        flow=pi[:,None]*q*total.exp()
        torch.testing.assert_close(flow,flow.T,atol=1e-14,rtol=0)
        assert (total<=torch.minimum(torch.zeros_like(ratio),ratio)+1e-14).all()
        reward=pi.log()[None,:]-pi.log()[:,None]
        utility=(source[:,None]*q*total.exp()*reward).sum()
        cost=(source[:,None]*q*2*first.exp()).sum()
        return -utility+.05*cost
    loss=objective(theta);g=torch.autograd.grad(loss,theta)[0];h=1e-5
    perturb=torch.zeros_like(theta);perturb[0,2]=h
    fd=(objective(theta.detach()+perturb)-objective(theta.detach()-perturb))/(2*h)
    torch.testing.assert_close(fd,g[0,2],atol=1e-9,rtol=1e-6)
    _,_,plain=delayed_log_acceptance(ratio,torch.zeros_like(ratio))
    torch.testing.assert_close(plain,torch.minimum(torch.zeros_like(ratio),ratio))
    with pytest.raises(ValueError):delayed_log_acceptance(ratio,torch.full_like(ratio,math.inf))


@pytest.mark.parametrize('variant',['zero','physical','linear','neural'])
def test_real_screen_antisymmetry_rigid_atom_action_symmetry_and_bound(variant):
    r=pair_record();torch.manual_seed(25801)
    model=DelayedAcceptanceScreen(variant).double()
    if variant in ('linear','neural'):
        with torch.no_grad():
            model.coefficients.normal_(0,.2)
            if model.encoder is not None:model.encoder.action_head[-1].weight.normal_(0,.2)
    x,y,b,new,z,e,a,ar=[r[k] for k in ['x','y','bonds','new_bonds','numbers','electronic','action','inverse_action']]
    proposal=r['log_behavior_reverse']-r['log_behavior_forward']+r['action_log_ratio']
    s=model.log_factor(x,y,b,new,z,e,a,proposal)
    reverse=model.log_factor(y,x,new,b,z,e,ar,-proposal)
    torch.testing.assert_close(s,-reverse,atol=1e-12,rtol=0)
    assert abs(float(s))<=math.log(16)+1e-12
    q=torch.linalg.qr(torch.randn(3,3,dtype=x.dtype))[0];q[:,0]*=-1
    torch.testing.assert_close(model.log_factor(x@q+11,y@q+11,b,new,z,e,a,proposal),s,atol=1e-10,rtol=0)
    perm=torch.randperm(len(z));inv=torch.argsort(perm);ap=tuple(int(inv[v]) for v in a)
    torch.testing.assert_close(model.log_factor(x[perm],y[perm],b[perm][:,perm],new[perm][:,perm],z[perm],e,ap,proposal),s,atol=1e-10,rtol=0)
    swapped=(a[1],a[0],a[3],a[2])
    torch.testing.assert_close(model.log_factor(x,y,b,new,z,e,swapped,proposal),s,atol=1e-12,rtol=0)
    features=paired_screen_features(x,y,b,new,z,e,proposal)
    torch.testing.assert_close(features,-paired_screen_features(y,x,new,b,z,e,-proposal),atol=0,rtol=0)


def test_real_cost_adjusted_screen_loss_gradient_and_no_oracle_input():
    row=pair_record();torch.manual_seed(25802);model=DelayedAcceptanceScreen('neural').double()
    u,c,g,s,a=screen_edge_values(model,row)
    assert float(s)==0 and float(c)==2
    with torch.no_grad():
        features=paired_screen_features(row['x'],row['y'],row['bonds'],row['new_bonds'],row['numbers'],row['electronic'],row['log_behavior_reverse']-row['log_behavior_forward']+row['action_log_ratio'])
        model.coefficients.copy_(-.4*features/features.square().sum())
        model.encoder.action_head[-1].weight.normal_(0,.002)
    def objective():
        u,c,_,s,_=screen_edge_values(model,row)
        return -u+.006*c+.0001*s.square()
    loss=objective();heads=[model.coefficients,model.encoder.action_head[-1].weight]
    grads=torch.autograd.grad(loss,heads)
    for w,g in zip(heads,grads):
        flat=int(g.abs().argmax());h=1e-5
        assert abs(float(g.flatten()[flat]))>1e-12
        with torch.no_grad():w.flatten()[flat]+=h
        high=float(objective())
        with torch.no_grad():w.flatten()[flat]-=2*h
        low=float(objective())
        with torch.no_grad():w.flatten()[flat]+=h
        assert abs((high-low)/(2*h)-float(g.flatten()[flat]))<1e-7
    assert all(float(v)==0 for v in screen_edge_values(model,{'valid':False}))
    # Changing recorded oracle outcomes must not change the pre-query factor.
    corrupt=dict(row,target_log_ratio=row['target_log_ratio']+100,reward_eV=row['reward_eV']-30)
    torch.testing.assert_close(screen_edge_values(model,row)[3],screen_edge_values(model,corrupt)[3],atol=0,rtol=0)


def test_zero_screen_preserves_actual_sampler_random_streams_and_queries():
    from cfm_mol.chemical_sampler import ChemicalTarget
    from cfm_mol.joint_chemical_geometry import joint_chemical_transition
    from scripts.research.audit_masked_angular import equal
    row=pair_record()
    class Oracle:
        evaluated=0
        def evaluate_chunked(self,p,max_request):
            self.evaluated+=len(p);return .01*p.square().sum((1,2)),-.02*p
    def run(screen):
        target=ChemicalTarget(Oracle(),dict(numbers=row['numbers'].tolist(),charge=0,spin_multiplicity=1),.026,.1)
        old=target.evaluate([target.coordinate_state(row['x'])],phase='initial')
        states,rows=joint_chemical_transition(target,old*8,kind='arc_site',generator=torch.Generator().manual_seed(25803),
            phase='joint',site_concentration=64.,screen=screen)
        return target,states,rows
    baseline,states,rows=run(None);screened,other,new=run(DelayedAcceptanceScreen('zero').double())
    equal(states,other);equal(baseline.query_trace,screened.query_trace)
    for a,b in zip(rows,new):
        assert a['valid']==b['scored'] and a['accepted']==b['accepted'] and a['log_uniform']==b['log_uniform']
        for k in a:equal(a[k],b[k])


def test_screen_rejects_before_oracle_and_full_corrected_transition_replays():
    from cfm_mol.chemical_sampler import ChemicalTarget
    from cfm_mol.joint_chemical_geometry import joint_chemical_transition
    from scripts.research.audit_masked_angular import ReplayOracle,equal
    row=pair_record();model=DelayedAcceptanceScreen('physical').double()
    class Oracle:
        evaluated=0
        def evaluate_chunked(self,p,max_request):
            self.evaluated+=len(p);return .01*p.square().sum((1,2)),-.02*p
    def run(oracle):
        target=ChemicalTarget(oracle,dict(numbers=row['numbers'].tolist(),charge=0,spin_multiplicity=1),.026,.1)
        old=target.evaluate([target.coordinate_state(row['x'])],phase='initial')
        before=oracle.evaluated
        states,rows=joint_chemical_transition(target,old*48,kind='arc_site',generator=torch.Generator().manual_seed(25807),
            phase='screen',site_concentration=64.,screen=model)
        assert oracle.evaluated-before==2*sum(r['scored'] for r in rows)
        return target,states,rows
    target,states,rows=run(Oracle())
    assert any(r['valid'] and not r['scored'] for r in rows)
    assert any(r['scored'] for r in rows)
    for r in rows:
        if not r['scored']:
            assert r['new_state_id']==-1 and 'target_log_ratio' not in r
            continue
        first,second,total=delayed_log_acceptance(torch.tensor(r['log_acceptance_ratio'],dtype=torch.float64),
                                                 torch.tensor(r['log_screen_factor'],dtype=torch.float64))
        assert abs(float(total)-r['total_log_acceptance'])<1e-12
        assert r['accepted']==(r['log_uniform']<float(second))
    oracle=ReplayOracle(target.query_trace);other,states2,rows2=run(oracle)
    equal(states,states2);equal(rows,rows2);assert oracle.index==len(oracle.queries)
