import torch
from cfm_mol.escorted_exchange import escorted_path
from cfm_mol.tempered_smc import DensityValue
from cfm_mol.chemical_path_guide import graph_guide_energy_force,exchanged_bond_graph
from cfm_mol.chemical_moves import covalent_radii


def test_graph_involution_and_guide_force_matches_independent_autograd():
    numbers=[6,16,1,9];radii=covalent_radii(numbers)
    bonds=torch.zeros(4,4,dtype=torch.float64)
    for i,j in [(0,1),(0,2),(1,3)]:bonds[i,j]=bonds[j,i]=1
    new=exchanged_bond_graph(bonds,(2,3,0,1))
    torch.testing.assert_close(exchanged_bond_graph(new,(2,3,1,0)),bonds)
    x=torch.tensor([[[-.7,0.,0.],[.7,0.,0.],[-.7,.9,.1],[.7,1.2,.2]]],dtype=torch.float64)
    cfg=dict(bond_kappa=20.,nonbond_kappa=100.,nonbond_factor=1.35)
    energy,force=graph_guide_energy_force(x,bonds[None],radii,**cfg)
    y=x.clone().requires_grad_();manual=0.
    for i in range(4):
        for j in range(i+1,4):
            distance=((y[:,i]-y[:,j]).square().sum(-1)+1e-12).sqrt();r0=radii[i]+radii[j]
            manual=manual+(10*(distance-r0)**2 if bonds[i,j] else 50*(1.35*r0-distance).clamp_min(0)**2)
    gradient=torch.autograd.grad(manual.sum(),y)[0]
    torch.testing.assert_close(energy,manual);torch.testing.assert_close(force,-gradient)
    torch.testing.assert_close(force.sum(1),torch.zeros(1,3,dtype=x.dtype),atol=1e-12,rtol=0)
    rotation=torch.linalg.qr(torch.randn(3,3,dtype=x.dtype))[0];rotation[:,0]*=-1
    perm=torch.tensor([3,0,2,1])
    e,f=graph_guide_energy_force((x@rotation+5)[:,perm],bonds[perm][:,perm][None],radii[perm],**cfg)
    torch.testing.assert_close(e,energy);torch.testing.assert_close(f,(force@rotation)[:,perm])


def test_endpoint_guide_correction_recovers_the_original_target_ratio():
    kT=.02585
    u=torch.tensor([-4.,-3.],dtype=torch.float64);v=torch.tensor([-5.,-2.],dtype=torch.float64)
    gx=torch.tensor([2.,.3],dtype=u.dtype);gy=torch.tensor([.1,1.5],dtype=u.dtype)
    path_ratio=torch.tensor([.6,-.9],dtype=u.dtype);logj=torch.tensor([.2,.1],dtype=u.dtype)
    guided=-(v+gy-u-gx)/kT+path_ratio+logj
    corrected=guided+(gy-gx)/kT
    torch.testing.assert_close(corrected,-(v-u)/kT+path_ratio+logj)


def test_paired_guides_preserve_known_target_and_omitting_endpoint_correction_biases_it():
    def run(correct):
        rng=torch.Generator().manual_seed(115)
        x=torch.randn(8192,1,dtype=torch.float64,generator=rng)
        for _ in range(12):
            old_sign=x.sign();current_sign=old_sign.clone()
            def smooth(v,**kwargs):
                displacement=v-1.5*current_sign
                return DensityValue(-.5*v.square().sum(1)-1.5*displacement.square().sum(1),-v-3*displacement)
            def mapping(v):
                nonlocal current_sign
                current_sign=-current_sign
                return -v,torch.zeros(len(v),dtype=v.dtype)
            initial=smooth(x)
            y,_,path=escorted_path(x,smooth,mapping,steps_per_side=2,std=.3,max_score_norm=4.,
                generator=rng,initial_value=initial)
            ratio=path['smooth_log_acceptance_ratio']
            if correct:
                gx=1.5*(x-1.5*old_sign).square().sum(1)
                gy=1.5*(y+1.5*old_sign).square().sum(1)
                ratio=ratio+gy-gx
            valid=(y.sign()==-old_sign).all(1)
            take=valid&(torch.rand(len(x),dtype=x.dtype,generator=rng).log()<ratio.clamp_max(0))
            x=torch.where(take[:,None],y,x)
        return x
    corrected=run(True);uncorrected=run(False)
    assert abs(float(corrected.mean()))<.03
    assert abs(float(corrected.square().mean())-1)<.05
    assert float(uncorrected.square().mean())>1.2
