"""Traceable, chemistry-supported local and terminal-exchange MH primitives.

The policy selects an augmented action. Its reverse family AND action probability
are evaluated at the proposal. States are labelled COM geometries, not graphs
alone. These kernels do not supply a finite-time endpoint density.
"""
import math
import torch
from cfm_mol.chemical_moves import (covalent_radii,infer_chemical_graph,
    terminal_exchange_actions,exchange_terminal_sites)
from cfm_mol.nonequilibrium import centered_orthonormal_basis
from cfm_mol.parity_refinement import evaluate_even_potential
from cfm_mol.tempered_smc import _clip_score


def pack_policy_states(states,numbers,kT):
    maximum=max(1,max(len(s['actions']) for s in states))
    actions=torch.full((len(states),maximum,4),-1,dtype=torch.long)
    for i,s in enumerate(states):
        if s['actions']:actions[i,:len(s['actions'])]=torch.tensor(s['actions'])
    x=torch.stack([s['positions'] for s in states])
    bonds=torch.stack([s['graph']['bond_orders'] for s in states])
    electronic=torch.tensor([[s['charge'],s['spin_multiplicity'],kT] for s in states],dtype=x.dtype)
    return x,bonds,torch.tensor(numbers,dtype=torch.long),electronic,actions


def policy_log_probabilities(states,numbers,kT,policy=None,uniform_local=.5):
    if policy is not None:return policy(*pack_policy_states(states,numbers,kT))
    if not 0<uniform_local<1:raise ValueError('Both fixed move families require positive probability')
    maximum=max(1,max(len(s['actions']) for s in states))
    result=torch.full((len(states),maximum+1),-torch.inf,dtype=torch.float64)
    for i,s in enumerate(states):
        m=len(s['actions']);result[i,0]=math.log(uniform_local) if m else 0.
        if m:result[i,1:m+1]=math.log((1-uniform_local)/m)
    return result


class ChemicalTarget:
    def __init__(self,oracle,condition,kT,restraint):
        self.oracle=oracle;self.condition=condition;self.numbers=condition['numbers']
        self.kT=kT;self.restraint=restraint
        self.basis=centered_orthonormal_basis(len(self.numbers))
        self.radii=covalent_radii(self.numbers)
        self.query_trace=[];self.states=[]

    def coordinate_state(self,x):
        x=x.detach().cpu().double()
        if float(x.mean(0).abs().max())>1e-8:raise ValueError('Noncentered state')
        graph=infer_chemical_graph(x,self.numbers,self.condition['charge'])
        return dict(positions=x.clone(),graph=graph,
            actions=terminal_exchange_actions(self.numbers,graph['bond_orders']),
            charge=self.condition['charge'],spin_multiplicity=self.condition['spin_multiplicity'])

    def evaluate(self,states,*,phase):
        if not states:return []
        positions=torch.stack([s['positions'] for s in states]);before=self.oracle.evaluated
        energy,force,parts=evaluate_even_potential(self.oracle,positions)
        if self.oracle.evaluated-before!=2*len(states):raise RuntimeError('Paired query count differs')
        # Retain BOTH raw forces. The even/odd decomposition reconstructs them.
        trace=dict(phase=phase,positions=positions,**parts,
            raw_force_eV_A=force+parts['odd_force_eV_A'],
            inverted_force_eV_A=parts['odd_force_eV_A']-force,
            raw_queries_before=before,raw_queries_after=self.oracle.evaluated)
        self.query_trace.append(trace)
        for i,state in enumerate(states):
            state.update(energy_eV=energy[i],force_eV_A=force[i],
                potential_eV=energy[i]+self.restraint/2*positions[i].square().sum(),
                score=(self.basis.T@((force[i]-self.restraint*positions[i])/self.kT)).flatten(),
                state_id=len(self.states),query_batch=len(self.query_trace)-1,query_row=i)
            self.states.append(state)
        return states

    def propose(self,state,action_index,generator,proposal_std):
        record=dict(old_state_id=state['state_id'],action_index=int(action_index),
            new_state_id=-1,valid=False,reverse_action_index=-1,proposal_std=float(proposal_std))
        if action_index==0:
            z=(self.basis.T@state['positions']).flatten()
            mean=z+.5*proposal_std**2*_clip_score(state['score'][None],100/self.kT)[0]
            noise=torch.randn(z.shape,dtype=z.dtype,generator=generator)
            y=self.basis@(mean+proposal_std*noise).reshape(len(self.numbers)-1,3)
            record.update(noise=noise,forward_mean=mean)
        else:
            action=state['actions'][action_index-1]
            y,volume,inverse=exchange_terminal_sites(state['positions'],self.radii,action)
            record.update(action=action,inverse_action=inverse,log_volume=volume)
        record['proposal_positions']=y
        try:
            candidate=self.coordinate_state(y)
            if action_index:
                inverse=record['inverse_action']
                if inverse not in candidate['actions']:raise ValueError('Reverse action not eligible')
                recovered,reverse_volume,_=exchange_terminal_sites(y,self.radii,inverse)
                torch.testing.assert_close(recovered,state['positions'],atol=1e-10,rtol=1e-10)
                torch.testing.assert_close(record['log_volume']+reverse_volume,
                    torch.zeros_like(reverse_volume),atol=1e-10,rtol=0)
                record['reverse_action_index']=1+candidate['actions'].index(inverse)
            else:record['reverse_action_index']=0
            record['valid']=True
            return candidate,record
        except ValueError as exc:
            record['rejection_reason']=str(exc)
            return None,record

    def finish_record(self,old,new,record,proposal_std):
        if new is None:
            record.update(base_log_ratio=torch.tensor(-torch.inf,dtype=torch.float64),utility=0.)
            return record
        base=-(new['potential_eV']-old['potential_eV'])/self.kT
        if record['action_index']:
            base=base+record['log_volume']
        else:
            z=(self.basis.T@old['positions']).flatten();y=(self.basis.T@new['positions']).flatten()
            reverse=y+.5*proposal_std**2*_clip_score(new['score'][None],100/self.kT)[0]
            gaussian=((y-record['forward_mean']).square().sum()-(z-reverse).square().sum())/(2*proposal_std**2)
            base=base+gaussian;record.update(reverse_mean=reverse,gaussian_log_ratio=gaussian)
        # Symmetric, fixed movement utility; neither a reference identity nor an
        # equilibrium error. No claim that it bounds the global spectral gap.
        topology=old['graph']['connectivity_smiles']!=new['graph']['connectivity_smiles']
        utility=float(topology)+min(1.,float(((new['potential_eV']-old['potential_eV'])/self.kT)**2))
        record.update(new_state_id=new['state_id'],base_log_ratio=base,utility=utility)
        return record

    @torch.no_grad()
    def transition(self,states,*,policy,generator,proposal_std,phase,local_only=False,uniform_local=.5):
        scales=torch.as_tensor(proposal_std,dtype=torch.float64).expand(len(states))
        if not torch.isfinite(scales).all() or (scales<=0).any():raise ValueError('Positive finite local scales required')
        logp=policy_log_probabilities(states,self.numbers,self.kT,policy,uniform_local)
        if local_only:indices=torch.zeros(len(states),dtype=torch.long)
        else:indices=torch.multinomial(logp.exp(),1,generator=generator)[:,0]
        candidates=[];records=[]
        for state,index,scale in zip(states,indices,scales):
            candidate,record=self.propose(state,int(index),generator,float(scale))
            candidates.append(candidate);records.append(record)
        self.evaluate([s for s in candidates if s is not None],phase=phase)
        valid_indices=[i for i,s in enumerate(candidates) if s is not None]
        reverse=policy_log_probabilities([candidates[i] for i in valid_indices],self.numbers,self.kT,policy,uniform_local) if valid_indices else None
        accepted=list(states);logu=torch.rand(len(states),dtype=torch.float64,generator=generator).log()
        reverse_row=0
        for i,(old,new,record) in enumerate(zip(states,candidates,records)):
            self.finish_record(old,new,record,float(scales[i]))
            pf=0. if local_only else float(logp[i,indices[i]])
            pr=0.
            if new is not None:
                pr=0. if local_only else float(reverse[reverse_row,record['reverse_action_index']])
                reverse_row+=1
            ratio=float(record['base_log_ratio'])+pr-pf
            take=new is not None and float(logu[i])<min(0.,ratio)
            record.update(phase=phase,log_forward_policy=pf,log_reverse_policy=pr,
                log_acceptance_ratio=ratio,log_uniform=float(logu[i]),accepted=take)
            if take:accepted[i]=new
        return accepted,records
