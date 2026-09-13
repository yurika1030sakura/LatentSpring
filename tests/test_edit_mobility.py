import torch
import copy
from cfm_mol.edit_mobility import GraphEditMobility,MobilizedEditField,differentiable_oracle_potential
from cfm_mol.edit_conditioned_bridge import edit_bridge,center
from cfm_mol.nonequilibrium import centered_orthonormal_basis
from test_bounded_action_geometry import pair_record


def model():
    torch.manual_seed(28301)
    m=MobilizedEditField(dict(hidden=8,radial=6,layers=1,force_scale=40.,edit_conditioned=True,roots_only=False),dict(variant='collective',hidden=8,rank=2)).double()
    with torch.no_grad():
        m.mobility.coefficients.weight.normal_(0,.3);m.mobility.scale.weight.normal_(0,.2);m.force_field.head.weight.normal_(0,.03)
    return m


def test_mobility_initial_control_symmetry_PSD_and_learnable_collective_modes():
    r=pair_record();args=(r['bonds'],r['new_bonds'],r['numbers'],r['electronic'],r['action'],.25)
    variants=[GraphEditMobility(v).double() for v in ['fixed','scalar','collective']]
    for m in variants[1:]:torch.testing.assert_close(m(*args),variants[0](*args),atol=1e-12,rtol=0)
    m=model().mobility;M=m(*args)
    torch.testing.assert_close(M,M.T,atol=1e-12,rtol=0);assert torch.linalg.eigvalsh(M).min()>-1e-12
    torch.testing.assert_close(M.sum(0),torch.zeros(len(M),dtype=M.dtype),atol=1e-12,rtol=0)
    torch.testing.assert_close(M,m(r['new_bonds'],r['bonds'],r['numbers'],r['electronic'],r['inverse_action'],.75),atol=1e-12,rtol=0)
    perm=torch.randperm(len(M));inv=torch.argsort(perm);a=tuple(int(inv[i]) for i in r['action'])
    torch.testing.assert_close(m(r['bonds'][perm][:,perm],r['new_bonds'][perm][:,perm],r['numbers'][perm],r['electronic'],a,.25),M[perm][:,perm],atol=1e-12,rtol=0)
    M.square().sum().backward();assert m.coefficients.weight.grad.abs().max()>1e-8


def test_mobilized_map_inverse_and_full_augmented_volume():
    r=pair_record();m=model();x=r['x'];p=center(torch.randn_like(x));basis=centered_orthonormal_basis(len(x));d=3*(len(x)-1)
    def call(x,p,b,c,a):return edit_bridge(x,p,b,c,r['numbers'],r['electronic'],a,m,steps_per_side=1,drift_step=.02)
    y,q,j=call(x,p,r['bonds'],r['new_bonds'],r['action']);xx,pp,jj=call(y,q,r['new_bonds'],r['bonds'],r['inverse_action'])
    torch.testing.assert_close(xx,x,atol=1e-10,rtol=0);torch.testing.assert_close(pp,p,atol=1e-10,rtol=0);torch.testing.assert_close(jj,-j)
    free=torch.cat([(basis.T@x).flatten(),(basis.T@p).flatten()])
    def transform(free):
        y,q,_=call(basis@free[:d].reshape(-1,3),basis@free[d:].reshape(-1,3),r['bonds'],r['new_bonds'],r['action'])
        return torch.cat([(basis.T@y).flatten(),(basis.T@q).flatten()])
    jac=torch.autograd.functional.jacobian(transform,free);torch.testing.assert_close(torch.linalg.slogdet(jac)[1],j,atol=1e-10,rtol=0)


def test_actual_oracle_gradient_attachment_matches_independent_quadratic_gradient():
    x=torch.randn(3,4,3,dtype=torch.float64,requires_grad=True);force=-2*x.detach();energy=x.detach().square().sum((1,2))
    value=differentiable_oracle_potential(x,energy,force,.1)
    torch.testing.assert_close(value,1.05*x.square().sum((1,2)),atol=1e-12,rtol=0)
    gradient=torch.autograd.grad(value.sum(),x)[0];torch.testing.assert_close(gradient,2.1*x,atol=1e-12,rtol=0)


def test_work_training_replays_conditions_gradients_and_every_paid_query():
    from scripts.research.train_edit_mobility_work import train,ConditionalReplayOracle
    from scripts.research.audit_masked_angular import equal
    r=pair_record();data=[]
    for index in [1,2]:
        row=dict(r,index=index,fit_only=True,source_potential_eV=float(.06*r['x'].square().sum()),destination_potential_eV=float(.06*r['y'].square().sum()))
        row['numbers']=r['numbers'].clone()
        if index==2:row['numbers'][row['numbers']==9]=17
        data.append(row)
    class Oracle:
        evaluated=0
        def evaluate_chunked(self,x,max_request):
            assert len(self.condition['numbers'])==x.shape[1]
            self.evaluated+=len(x);return .01*x.square().sum((1,2)),-.02*x
    protocol=dict(condition_indices=[1,2],restraint_eV_A2=.1,source_energy_tolerance_eV=1e-10,steps=2,batch_size=2,
        learning_rate=.001,gradient_clip=10.,graph_penalty_weight=100.,bridge=dict(steps_per_side=1,kick_step=.2,drift_step=.02))
    first=model();second=copy.deepcopy(first);oracle=Oracle();saved=train(first,data,oracle,protocol,28311)
    assert oracle.evaluated==16 and saved['query_trace'][0]['condition']!=saved['query_trace'][1]['condition']
    replay=ConditionalReplayOracle(saved['query_trace']);actual=train(second,data,replay,protocol,28311);equal(actual,saved)
    assert replay.index==len(replay.queries) and replay.evaluated==16
