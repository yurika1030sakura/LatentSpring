import torch
from cfm_mol.terminal_rotation import rotate_terminal,quaternion_matrix,uniform_internal_transition
from cfm_mol.chemical_sampler import ChemicalTarget
from cfm_mol.nonequilibrium import centered_orthonormal_basis


def test_quaternion_auxiliary_and_com_rotation_are_an_involution_with_unit_volume():
    torch.manual_seed(119)
    basis=centered_orthonormal_basis(4);x=torch.randn(4,3,dtype=torch.float64);x-=x.mean(0)
    raw=torch.randn(4,dtype=x.dtype);y,reverse=rotate_terminal(x,(2,0),raw)
    recovered,recovered_raw=rotate_terminal(y,(2,0),reverse)
    torch.testing.assert_close(recovered,x,atol=1e-12,rtol=0);torch.testing.assert_close(recovered_raw,raw)
    torch.testing.assert_close((y[2]-y[0]).norm(),(x[2]-x[0]).norm())
    torch.testing.assert_close(quaternion_matrix(raw).det(),torch.ones((),dtype=x.dtype))
    initial=torch.cat([(basis.T@x).flatten(),raw])
    def extended(v):
        pos=basis@v[:9].reshape(3,3);mapped,aux=rotate_terminal(pos,(2,0),v[9:])
        return torch.cat([(basis.T@mapped).flatten(),aux])
    jac=torch.autograd.functional.jacobian(extended,initial)
    torch.testing.assert_close(torch.linalg.slogdet(jac)[1],torch.zeros((),dtype=x.dtype),atol=1e-11,rtol=0)
    torch.testing.assert_close(extended(extended(initial)),initial,atol=1e-12,rtol=0)
    perm=torch.tensor([3,0,2,1]);lookup=torch.argsort(perm)
    py,_=rotate_terminal(x[perm],(int(lookup[2]),int(lookup[0])),raw)
    torch.testing.assert_close(py,y[perm],atol=1e-12,rtol=0)


def test_gaussian_quaternion_rotations_cover_the_sphere_without_direction_bias():
    g=torch.Generator().manual_seed(120);raw=torch.randn(4096,4,dtype=torch.float64,generator=g)
    # One fixed direction under independent Haar rotations should have isotropic
    # first and second moments; inverse auxiliaries have the same Gaussian density.
    directions=torch.stack([quaternion_matrix(q)[:,0] for q in raw])
    assert float(directions.mean(0).abs().max())<.035
    torch.testing.assert_close(directions.T@directions/len(raw),torch.eye(3,dtype=raw.dtype)/3,atol=.025,rtol=0)
    reverse=raw*raw.new_tensor([1.,-1.,-1.,-1.])
    torch.testing.assert_close(raw.square().sum(1),reverse.square().sum(1))


def test_internal_move_decisions_use_physical_ratio_and_count_only_scored_candidates():
    class Oracle:
        evaluated=0
        def evaluate_chunked(self,x,max_request):
            self.evaluated+=len(x);return .5*x.square().sum((1,2)),-x
    target=ChemicalTarget(Oracle(),dict(numbers=[6,1,1,1,9],charge=0,spin_multiplicity=1),.5,.1)
    x=torch.tensor([[0.,0.,0.],[1.,1.,1.],[1.,-1.,-1.],[-1.,1.,-1.],[-1.,-1.,1.]],dtype=torch.float64)
    x[1:4]*=1.09/3**.5;x[4]*=1.35/3**.5;x-=x.mean(0)
    states=target.evaluate([target.coordinate_state(x)],phase='initial')
    rng=torch.Generator().manual_seed(121);valid=0
    for kind in ['rotation','exchange','force_rotation']*10:
        old=states[0]
        states,rows=uniform_internal_transition(target,states,kind=kind,generator=rng,phase='test')
        row=rows[0];valid+=row['valid']
        if row['valid']:
            new=target.states[row['new_state_id']]
            expected=-float(new['potential_eV']-old['potential_eV'])/target.kT+float(row['log_volume'])+row['action_log_ratio']
            if kind=='force_rotation':expected+=row['angular_log_reverse']-row['angular_log_forward']
            assert abs(expected-row['log_acceptance_ratio'])<1e-10
            assert row['accepted']==(row['log_uniform']<min(0.,expected))
        else:assert not row['accepted']
    assert target.oracle.evaluated==2*(1+valid)
