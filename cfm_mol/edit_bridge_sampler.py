"""Physical-endpoint molecular proposals for the frozen edit-transport pilot."""
import math
import torch
from torch import nn
from cfm_mol.edit_conditioned_bridge import edit_bridge,center
from cfm_mol.chemical_path_guide import exchanged_bond_graph
from cfm_mol.joint_chemical_geometry import distinct_anchor_actions
from cfm_mol.joint_arc_geometry import marginal_joint_arc_proposal
from cfm_mol.conditional_molecular_proposal import invariant_jump_squared


class ZeroBridgeField(nn.Module):
    roots_only=False
    def forward(self,x,*args):return torch.zeros_like(x)


class RootZeroBridgeField(ZeroBridgeField):
    roots_only=True


class AnalyticBridgeField(nn.Module):
    """Fixed dimensionless graph springs; a control field, not the target energy."""
    roots_only=False
    def __init__(self,bond_stiffness=20.,nonbond_stiffness=100.,nonbond_factor=1.35,force_cap=40.):
        super().__init__();self.bond_stiffness=bond_stiffness;self.nonbond_stiffness=nonbond_stiffness;self.nonbond_factor=nonbond_factor;self.force_cap=force_cap
    def forward(self,x,bonds,new_bonds,numbers,electronic,action,t):
        from cfm_mol.chemical_moves import covalent_radii
        r=covalent_radii(numbers).to(x);reference=r[:,None]+r[None,:];delta=x[:,None]-x[None,:]
        distance=(delta.square().sum(-1)+1e-12).sqrt();bonded=(1-t)*(bonds>0).to(x)+t*(new_bonds>0).to(x)
        derivative=self.bond_stiffness*(distance-reference)*bonded-self.nonbond_stiffness*(self.nonbond_factor*reference-distance).clamp_min(0)*(1-bonded)
        derivative=derivative*(~torch.eye(len(x),dtype=torch.bool,device=x.device))
        force=center(-(derivative[...,None]*delta/distance[...,None]).sum(1))
        return force/(1+force.norm(dim=1).max()/self.force_cap)


@torch.no_grad()
def propose_edit(target,old,action,momentum,order,arc_seed,*,method,field,bridge_options,arc_options):
    x=old['positions'];b=old['graph']['bond_orders'];numbers=torch.tensor(target.numbers,dtype=torch.long)
    electronic=x.new_tensor([old['charge'],old['spin_multiplicity'],target.kT]);actions=distinct_anchor_actions(numbers,b)
    if action not in actions:raise ValueError('Declared edit is not eligible at the source')
    desired=exchanged_bond_graph(b,action);inverse=(action[0],action[1],action[3],action[2])
    record=dict(method=method,old_state_id=old['state_id'],action=action,inverse_action=inverse,
        forward_action_count=len(actions),valid=False,scored=False,accepted=False,new_state_id=-1,
        order=order,arc_seed=arc_seed,expected_utility_eV=0.,expected_constitutional_flow=0.,expected_invariant_jump_A2=0.,raw_cost=0)
    if method=='physical_arc':
        y,forward,trace=marginal_joint_arc_proposal(x,b,numbers,electronic,target.radii,action,kind='arc_site',order=order,
            generator=torch.Generator().manual_seed(arc_seed),site_concentration=64.,restraint=target.restraint,**arc_options)
        record['forward']=trace
        if y is None:record['failure_reason']=trace['failure'];return None,record
    else:
        y,new_p,volume=edit_bridge(x,momentum,b,desired,numbers,electronic,action,field,**bridge_options)
        record.update(input_momentum=momentum,output_momentum=new_p,log_volume=float(volume),
            momentum_log_ratio=float(.5*(momentum.square().sum()-new_p.square().sum())))
    record['proposal_positions']=y
    try:
        candidate=target.coordinate_state(y)
        if not torch.equal(candidate['graph']['bond_orders'],desired):raise ValueError('Endpoint graph differs from the proposed chemical edit')
        reverse_actions=distinct_anchor_actions(numbers,candidate['graph']['bond_orders'])
        if inverse not in reverse_actions:raise ValueError('Inverse edit is not eligible')
    except ValueError as exc:record['failure_reason']=str(exc);return None,record
    record.update(valid=True,reverse_action_count=len(reverse_actions),action_log_ratio=math.log(len(actions)/len(reverse_actions)))
    if method=='physical_arc':
        _,reverse,trace=marginal_joint_arc_proposal(y,desired,numbers,electronic,target.radii,inverse,kind='arc_site',order=0,
            observed=x,site_concentration=64.,restraint=target.restraint,**arc_options)
        record['reverse']=trace
        if not torch.isfinite(reverse):record['failure_reason']='Zero reverse coordinate density';return None,record
        record['coordinate_log_ratio']=float(reverse-forward)
    else:record['coordinate_log_ratio']=record['momentum_log_ratio']+record['log_volume']
    return candidate,record


@torch.no_grad()
def finish_edit(target,old,new,record,log_uniform):
    record['log_uniform']=log_uniform
    if new is None:return record
    difference=float(new['potential_eV']-old['potential_eV'])
    ratio=-difference/target.kT+record['coordinate_log_ratio']+record['action_log_ratio']
    if not math.isfinite(ratio):raise ValueError('Nonfinite augmented acceptance ratio')
    alpha=math.exp(min(0.,ratio));changed=new['graph']['connectivity_smiles']!=old['graph']['connectivity_smiles']
    jump=float(invariant_jump_squared(old['positions'][None],new['positions'][None],torch.tensor(target.numbers))[0])
    record.update(scored=True,raw_cost=2,new_state_id=new['state_id'],potential_change_eV=difference,
        log_acceptance_ratio=ratio,acceptance_probability=alpha,accepted=log_uniform<min(0.,ratio),
        connectivity_changed=changed,invariant_jump_A2=jump,expected_utility_eV=-alpha*difference,
        expected_constitutional_flow=alpha*changed,expected_invariant_jump_A2=alpha*jump)
    return record
