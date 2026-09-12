import math
import pytest
import torch
from cfm_mol.source_force_screen import SourceForceScreen,screened_log_acceptance,force_edge_values
from cfm_mol.delayed_acceptance import DelayedAcceptanceScreen,screen_edge_values
from test_bounded_action_geometry import pair_record


def force_pair():
    r=pair_record();r['source_force_eV_A']=-.02*r['x'];r['candidate_force_eV_A']=-.02*r['y']
    return r


def test_nonreciprocal_gates_balance_and_actual_gradient():
    dtype=torch.float64;pi=torch.tensor([.2,.3,.5],dtype=dtype)
    q=torch.tensor([[.1,.6,.3],[.4,.2,.4],[.2,.5,.3]],dtype=dtype)
    theta=torch.tensor([[.1,-.3,.7],[-.2,.6,-.4],[.5,-.1,.2]],dtype=dtype,requires_grad=True)
    ratio=pi.log()[None,:]-pi.log()[:,None]+q.T.log()-q.log();source=pi.new_tensor([.7,.2,.1])
    reward=pi.log()[None,:]-pi.log()[:,None]
    def loss(t):
        logg=torch.nn.functional.logsigmoid(t)
        _,total=screened_log_acceptance(ratio,logg,logg.T)
        flow=pi[:,None]*q*total.exp();torch.testing.assert_close(flow,flow.T,atol=1e-14,rtol=0)
        direct=torch.minimum(logg.exp(),ratio.exp()*logg.T.exp())
        torch.testing.assert_close(total.exp(),direct,atol=1e-14,rtol=0)
        return (source[:,None]*q*(-reward*total.exp()+.04*2*logg.exp())).sum()
    value=loss(theta);gradient=torch.autograd.grad(value,theta)[0];h=1e-5
    delta=torch.zeros_like(theta);delta[0,1]=h
    torch.testing.assert_close((loss(theta.detach()+delta)-loss(theta.detach()-delta))/(2*h),gradient[0,1],atol=1e-9,rtol=1e-6)
    # Constant thinning is a valid special case, with ordinary MH correction.
    gate=torch.full_like(ratio,math.log(.4));second,total=screened_log_acceptance(ratio,gate,gate)
    torch.testing.assert_close(second,torch.minimum(torch.zeros_like(ratio),ratio))
    torch.testing.assert_close(total.exp(),.4*second.exp())


def test_zero_and_physical_controls_reproduce_previous_expected_values():
    r=force_pair()
    for name in ['zero','physical']:
        before=screen_edge_values(DelayedAcceptanceScreen(name).double(),r)
        after=force_edge_values(SourceForceScreen(name).double(),r)
        for i in [0,1,2]:torch.testing.assert_close(after[i],before[i],atol=1e-12,rtol=0)
        torch.testing.assert_close(after[-1],before[-1],atol=1e-12,rtol=0)


def test_force_network_symmetries_nonreciprocity_and_nonzero_gradients():
    r=force_pair();torch.manual_seed(25901);model=SourceForceScreen('neural').double()
    with torch.no_grad():
        model.coefficients.fill_(.001);model.encoder.head.weight.normal_(0,.01);model.encoder.head.bias.fill_(-.2)
    x,y,b,new,z,e,a=[r[k] for k in ['x','y','bonds','new_bonds','numbers','electronic','action']]
    force=r['source_force_eV_A'];proposal=r['log_behavior_reverse']-r['log_behavior_forward']+r['action_log_ratio']
    def gate(x,y,b,new,z,e,a,f):return model.log_pass_probability(x,y,b,new,z,e,a,proposal,f)
    expected=gate(x,y,b,new,z,e,a,force)
    q=torch.linalg.qr(torch.randn(3,3,dtype=x.dtype))[0];q[:,0]*=-1
    torch.testing.assert_close(gate(x@q+9,y@q+9,b,new,z,e,a,force@q),expected,atol=1e-10,rtol=0)
    perm=torch.randperm(len(z));inv=torch.argsort(perm)
    torch.testing.assert_close(gate(x[perm],y[perm],b[perm][:,perm],new[perm][:,perm],z[perm],e,tuple(int(inv[i]) for i in a),force[perm]),expected,atol=1e-10,rtol=0)
    torch.testing.assert_close(gate(x,y,b,new,z,e,(a[1],a[0],a[3],a[2]),force),expected,atol=1e-12,rtol=0)
    def loss():
        u,c,_,f,rev,_=force_edge_values(model,r)
        return -u+.006*c+.0001*(f.square()+rev.square())
    heads=[model.coefficients,model.encoder.head.weight];grads=torch.autograd.grad(loss(),heads)
    for w,g in zip(heads,grads):
        index=int(g.abs().argmax());assert abs(float(g.flatten()[index]))>1e-12;h=1e-5
        with torch.no_grad():w.flatten()[index]+=h
        plus=float(loss())
        with torch.no_grad():w.flatten()[index]-=2*h
        minus=float(loss())
        with torch.no_grad():w.flatten()[index]+=h
        assert abs((plus-minus)/(2*h)-float(g.flatten()[index]))<1e-7
    # Candidate force changes the reverse decision, never the forward one.
    changed=dict(r,candidate_force_eV_A=r['candidate_force_eV_A']+torch.randn_like(force))
    first=force_edge_values(model,r);other=force_edge_values(model,changed)
    torch.testing.assert_close(first[3],other[3],atol=0,rtol=0)
    assert abs(float(first[4]-other[4]))>1e-6


def test_production_gate_information_order_zero_rng_and_paid_reverse_failure():
    from cfm_mol.chemical_sampler import ChemicalTarget
    from cfm_mol.joint_chemical_geometry import joint_chemical_transition
    from scripts.research.audit_masked_angular import equal,ReplayOracle
    r=force_pair()
    class Oracle:
        evaluated=0
        def evaluate_chunked(self,p,max_request):
            self.evaluated+=len(p);return .01*p.square().sum((1,2)),-.02*p
    def target(oracle):return ChemicalTarget(oracle,dict(numbers=r['numbers'].tolist(),charge=0,spin_multiplicity=1),.026,.1)
    def run(oracle,screen):
        t=target(oracle);initial=t.evaluate([t.coordinate_state(r['x'])],phase='initial')
        end,rows=joint_chemical_transition(t,initial*24,kind='arc_site',generator=torch.Generator().manual_seed(25903),phase='test',site_concentration=64.,screen=screen)
        return t,end,rows
    plain,end,rows=run(Oracle(),None);zero,end2,rows2=run(Oracle(),SourceForceScreen('zero').double())
    equal(end,end2);equal(plain.query_trace,zero.query_trace)
    assert [r['log_uniform'] for r in rows]==[r['log_uniform'] for r in rows2]
    oracle=Oracle();seen=[]
    class CappedWork(SourceForceScreen):
        def log_pass_probability(self,x,*args):
            # The quadratic toy favors every forward work estimate. A valid
            # ceiling forces this fixture to exercise pre-query rejection.
            return torch.minimum(super().log_pass_probability(x,*args),x.new_tensor(math.log(.4)))
    class Spy(CappedWork):
        def log_pass_probability(self,x,y,b,new,z,e,act,q,source_force):
            torch.testing.assert_close(source_force,-.02*x,atol=1e-12,rtol=0)
            seen.append((oracle.evaluated,bool(torch.allclose(x,r['x']))))
            return super().log_pass_probability(x,y,b,new,z,e,act,q,source_force)
    screen=Spy('work').double();t,end,rows=run(oracle,screen)
    assert any(row['valid'] and not row['scored'] for row in rows)
    assert oracle.evaluated==2+2*sum(row['scored'] for row in rows)
    assert all(count==2 if is_source else count>2 for count,is_source in seen)
    replay=ReplayOracle(t.query_trace);t2,end2,rows2=run(replay,CappedWork('work').double());equal(end,end2);equal(rows,rows2)
    oracle=Oracle();t=target(oracle);initial=t.evaluate([t.coordinate_state(r['x'])],phase='initial')
    class Broken(SourceForceScreen):
        def log_pass_probability(self,x,*args):return x.new_tensor(0. if oracle.evaluated==2 else float('nan'))
    with pytest.raises(ValueError,match='after paid'):
        joint_chemical_transition(t,initial*8,kind='arc_site',generator=torch.Generator().manual_seed(25907),phase='bad',site_concentration=64.,screen=Broken('zero'))
    assert oracle.evaluated>2 and len(t.query_trace)>1
