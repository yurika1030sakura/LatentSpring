"""Capped, support-conditioned angular proposals with invariant masked context."""
import math
import torch
from cfm_mol.masked_angular_guide import masked_angular_context,angular_log_score
from cfm_mol.terminal_rotation import terminal_rotation_actions
from cfm_mol.angular_envelope import envelope_draw


@torch.no_grad()
def capped_directions(eta,matrix,envelope,valid,*,max_trials,generator,proposal='uniform'):
    if (eta.ndim!=2 or eta.shape[1]!=3 or matrix.shape!=(len(eta),3,3)
            or envelope.shape!=(len(eta),) or not isinstance(max_trials,int) or max_trials<1):
        raise ValueError('Valid angular coefficients and positive trial cap required')
    if not all(torch.isfinite(v).all() for v in [eta,matrix,envelope]):raise ValueError('Finite coefficients required')
    success=torch.zeros(len(eta),dtype=torch.bool,device=eta.device);directions=torch.zeros_like(eta);trace=[]
    for trial in range(max_trials):
        active=(~success).nonzero().flatten()
        if len(active)==0:break
        auxiliary=None;noise=None
        if proposal=='uniform':
            noise=torch.randn((len(active),3),dtype=eta.dtype,device=eta.device,generator=generator)
            u=noise/noise.norm(dim=1,keepdim=True)
            score=angular_log_score(u,eta[active],matrix[active]);logp=score-envelope[active]
        elif proposal=='vmf_envelope':
            u,logp,auxiliary=envelope_draw(eta[active],matrix[active],generator=generator)
            score=angular_log_score(u,eta[active],matrix[active])
        else:raise ValueError('Unknown angular rejection base')
        if (logp>1e-8).any():raise ValueError('Angular rejection envelope violated')
        logu=torch.rand(len(active),dtype=eta.dtype,device=eta.device,generator=generator).log()
        passed=logu<logp.clamp_max(0);supported=torch.zeros_like(passed)
        if passed.any():
            checked=valid(u[passed],active[passed])
            if checked.shape!=(int(passed.sum()),) or checked.dtype!=torch.bool:raise ValueError('Invalid support callback')
            supported[passed]=checked
        take=passed&supported;success[active[take]]=True;directions[active[take]]=u[take]
        row=dict(trial=trial,indices=active,noise=noise,directions=u,log_uniform=logu,
            score=score,score_passed=passed,geometry_checked=passed,geometry_valid=supported,accepted=take)
        if proposal!='uniform':row.update(proposal=proposal,proposal_auxiliary=auxiliary,log_rejection_probability=logp)
        trace.append(row)
    return directions,success,trace


@torch.no_grad()
def masked_angular_transition(target,states,model,*,max_trials,generator,phase,proposal='uniform'):
    roots=[];choices=[];counts=[]
    for s in states:
        actions=terminal_rotation_actions(target.numbers,s['graph']['bond_orders'])
        if not actions:raise ValueError('Masked angular pilot requires eligible leaves')
        index=int(torch.randint(len(actions),(1,),generator=generator));roots.append(actions[index]);choices.append(index);counts.append(len(actions))
    x=torch.stack([s['positions'] for s in states]);bonds=torch.stack([s['graph']['bond_orders'] for s in states])
    roots=torch.tensor(roots,dtype=torch.long);numbers=torch.tensor(target.numbers,dtype=torch.long)
    electronic=torch.tensor([target.condition['charge'],target.condition['spin_multiplicity'],target.kT],dtype=x.dtype)
    batch=torch.arange(len(states));leaf,anchor=roots.unbind(1);relative=x[batch,leaf]-x[batch,anchor]
    radius=relative.norm(dim=1);old_direction=relative/radius[:,None]
    if model is None:eta=torch.zeros(len(x),3,dtype=x.dtype);matrix=torch.zeros(len(x),3,3,dtype=x.dtype);envelope=torch.zeros(len(x),dtype=x.dtype)
    else:eta,matrix,envelope=model(x,bonds,numbers,electronic,roots)
    candidates=[None]*len(states);checks=[]
    def valid(u,indices):
        values=[]
        for direction,index in zip(u,indices.tolist()):
            y=x[index].clone();y[leaf[index]]=y[anchor[index]]+radius[index]*direction;y-=y.mean(0)
            check=dict(chain=index,direction=direction.clone(),positions=y,valid=False)
            try:
                candidate=target.coordinate_state(y)
                if not torch.equal(candidate['graph']['bond_orders'],bonds[index]):raise ValueError('Perceived graph changed')
                for before,after in zip(masked_angular_context(x[index:index+1],roots[index:index+1]),
                        masked_angular_context(y[None],roots[index:index+1])):
                    torch.testing.assert_close(before,after,atol=1e-10,rtol=1e-10)
                if model is not None:
                    reverse=model(y[None],candidate['graph']['bond_orders'][None],numbers,electronic,roots[index:index+1])
                    for old,new in zip([eta[index:index+1],matrix[index:index+1],envelope[index:index+1]],reverse):
                        torch.testing.assert_close(old,new,atol=1e-8,rtol=1e-9)
                candidates[index]=candidate;check['valid']=True
            except ValueError as exc:check['rejection_reason']=str(exc)
            checks.append(check);values.append(check['valid'])
        return torch.tensor(values,dtype=torch.bool)
    direction,success,trials=capped_directions(eta,matrix,envelope,valid,max_trials=max_trials,generator=generator,proposal=proposal)
    target.evaluate([c for c in candidates if c is not None],phase=phase)
    old_scores=angular_log_score(old_direction,eta,matrix);new_scores=angular_log_score(direction,eta,matrix)
    logu=torch.rand(len(states),dtype=x.dtype,generator=generator).log();rows=[];updated=list(states)
    for i,(old,new) in enumerate(zip(states,candidates)):
        row=dict(kind='masked_angle',phase=phase,old_state_id=old['state_id'],new_state_id=-1,
            action=tuple(roots[i].tolist()),choice_index=choices[i],forward_count=counts[i],valid=bool(success[i]),
            exhausted=not bool(success[i]),accepted=False,log_uniform=float(logu[i]),
            old_angular_score=float(old_scores[i]),new_angular_score=float(new_scores[i]))
        if new is not None:
            reverse_count=len(terminal_rotation_actions(target.numbers,new['graph']['bond_orders']))
            correction=float(old_scores[i]-new_scores[i])+math.log(counts[i]/reverse_count)
            ratio=-float(new['potential_eV']-old['potential_eV'])/target.kT+correction
            take=float(logu[i])<min(0.,ratio)
            row.update(new_state_id=new['state_id'],reverse_count=reverse_count,proposal_log_ratio=correction,
                log_acceptance_ratio=ratio,accepted=take)
            if take:updated[i]=new
        rows.append(row)
    search=dict(phase=phase,old_state_ids=[s['state_id'] for s in states],roots=roots,eta=eta,matrix=matrix,
        envelope=envelope,radius=radius,old_directions=old_direction,directions=direction,success=success,
        trials=trials,geometry_checks=checks,decisions=rows)
    return updated,rows,search
