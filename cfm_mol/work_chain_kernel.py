"""Frozen single/cooperative work policies as complete-chain proposal kernels."""
import copy
import math
import torch
from cfm_mol.chemical_work_policy import catalogue as single_catalogue, policy as single_policy
from cfm_mol.cooperative_edit_policy import sample_root_blocks,panel_catalogue,block_policy,reverse_index,terminal_roots
from cfm_mol.edit_bridge_sampler import RootZeroBridgeField,propose_edit
from cfm_mol.edit_conditioned_bridge import center
from cfm_mol.joint_chemical_geometry import distinct_anchor_actions


def choose(log_probability,generator):
    u=torch.rand((),dtype=torch.float64,generator=generator)
    return min(int(torch.searchsorted(log_probability.exp().cumsum(0),u,right=True)),len(log_probability)-1)


def empty(old,reason):
    return None,dict(old_state_id=old['state_id'],new_state_id=-1,valid=False,scored=False,accepted=False,raw_cost=0,failure_reason=reason)


def record_probability(probability,rows):
    return dict(probability,log_volumes=torch.tensor([r['record']['log_volume'] for r in rows],dtype=torch.float64))


class WorkChainKernel:
    def __init__(self,kind,backbone=None,interaction=None,*,use_affinity=False,panel_size=32,policy_options=None,root_noise_options=None):
        if kind not in ('root_noise','single','panel'):raise ValueError('Unknown chain kernel')
        self.kind,self.backbone,self.interaction=kind,backbone,interaction
        self.use_affinity=use_affinity;self.panel_size=panel_size
        self.policy_options=policy_options or dict(scale_eV=.25,bound=1.,uniform_fraction=.1)
        self.root_noise_options=root_noise_options or dict(steps_per_side=2,kick_step=.2,drift_step=.01)

    @torch.no_grad()
    def single(self,target,old,generator,*,fallback=False):
        options=single_catalogue(target,old)
        if not options['valid']:return empty(old,'No valid single edit')
        forward=single_policy(target,old,options,self.backbone,self.policy_options['uniform_fraction'])
        j=choose(forward['log_probability'],generator);item=options['valid'][j]
        new=copy.deepcopy(item['candidate']);row=copy.deepcopy(item['record'])
        reverse=single_catalogue(target,new);inverse=tuple(row['inverse_action'])
        indices=[i for i,r in enumerate(reverse['valid']) if tuple(r['action'])==inverse]
        if len(indices)!=1:raise ValueError('Reverse single catalogue lacks unique inverse')
        k=indices[0];torch.testing.assert_close(reverse['valid'][k]['candidate']['positions'],old['positions'],atol=1e-9,rtol=0)
        backward=single_policy(target,new,reverse,self.backbone,self.policy_options['uniform_fraction'])
        assert terminal_roots(target.numbers,new['graph']['bond_orders'])==terminal_roots(target.numbers,old['graph']['bond_orders'])
        row.update(kernel_type='single',type_fallback=fallback,forward_selected=j,reverse_selected=k,
            forward=record_probability(forward,options['valid']),reverse=record_probability(backward,reverse['valid']),
            action_log_ratio=float(backward['log_probability'][k]-forward['log_probability'][j]),
            coordinate_log_ratio=row['log_volume'])
        return new,row

    @torch.no_grad()
    def propose(self,target,old,generator,seed):
        if self.kind=='root_noise':
            actions=distinct_anchor_actions(target.numbers,old['graph']['bond_orders'])
            if not actions:return empty(old,'No eligible root-noise edit')
            action=actions[int(torch.randint(len(actions),(1,),generator=generator))]
            momentum=center(torch.randn(old['positions'].shape,dtype=torch.float64,generator=generator))
            order=int(torch.randint(2,(1,),generator=generator))
            new,row=propose_edit(target,old,action,momentum,order,seed+100000000,method='root_noise',field=RootZeroBridgeField(),
                bridge_options=self.root_noise_options,arc_options={})
            row['kernel_type']='root_noise';return new,row
        if self.kind=='single':return self.single(target,old,generator)
        blocks,total=sample_root_blocks(target.numbers,old['graph']['bond_orders'],self.panel_size,generator)
        if total==0:
            # Type-only infeasibility is unchanged by single terminal swaps;
            # the deterministic fallback therefore needs no family correction.
            return self.single(target,old,generator,fallback=True)
        options=panel_catalogue(target,old,blocks)
        if not options['valid']:
            new,row=empty(old,'No valid matching in sampled panel');row.update(kernel_type='panel',panel=blocks,panel_pool_size=total);return new,row
        forward=block_policy(target,old,options,self.backbone,self.interaction,use_affinity=self.use_affinity,**self.policy_options)
        j=choose(forward['log_probability'],generator);item=options['valid'][j];new=copy.deepcopy(item['candidate'])
        reverse=panel_catalogue(target,new,blocks);k=reverse_index(item,reverse)
        backward=block_policy(target,new,reverse,self.backbone,self.interaction,use_affinity=self.use_affinity,**self.policy_options)
        assert terminal_roots(target.numbers,new['graph']['bond_orders'])==terminal_roots(target.numbers,old['graph']['bond_orders'])
        assert abs(float(forward['total_interaction_eV'][j]-backward['total_interaction_eV'][k]))<1e-8
        assert abs(float(forward['predicted_linear_work_eV'][j]+backward['predicted_linear_work_eV'][k]))<1e-8
        row=dict(item['record'],kernel_type='panel',method='panel',old_state_id=old['state_id'],new_state_id=-1,
            valid=True,scored=False,accepted=False,raw_cost=0,panel=blocks,panel_pool_size=total,
            forward_selected=j,reverse_selected=k,forward=record_probability(forward,options['valid']),
            reverse=record_probability(backward,reverse['valid']),coordinate_log_ratio=item['record']['log_volume'],
            action_log_ratio=float(backward['log_probability'][k]-forward['log_probability'][j]))
        return new,row


def independently_normalized(record,kT,uniform_fraction):
    """Independent scalar arithmetic for logged full valid-catalogue weights."""
    work=record.get('work',record.get('predicted_linear_work_eV'))
    affinity=record.get('log_affinity',torch.zeros_like(work))
    weights=[]
    for w,j,h in zip(work,record['log_volumes'],affinity):
        ratio=-float(w)/kT+float(j)
        weights.append(min(ratio,0.)-math.log1p(math.exp(-abs(ratio)))+float(h))
    maximum=max(weights);z=sum(math.exp(v-maximum) for v in weights)
    return [math.log((1-uniform_fraction)*math.exp(v-maximum)/z+uniform_fraction/len(weights)) for v in weights]
