"""Classical fixed-length terminal rotations and uniform chemical exchange.

Gaussian quaternion auxiliaries have a measure-preserving conjugate reversal.
The Cartesian map has unit intrinsic COM volume. This is not AI novelty.
"""
import math
import torch


def terminal_rotation_actions(numbers,bonds):
    result=[]
    for i,z in enumerate(numbers):
        neighbours=(bonds[i]>0).nonzero().flatten()
        if int(z) in (1,9,17,35,53) and len(neighbours)==1 and float(bonds[i,neighbours[0]])==1:
            result.append((i,int(neighbours[0])))
    return result


def quaternion_matrix(raw):
    if raw.shape!=(4,) or not torch.isfinite(raw).all() or float(raw.norm())<1e-12:
        raise ValueError('Finite nonzero four-dimensional quaternion auxiliary required')
    w,x,y,z=raw/raw.norm()
    return torch.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
        2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
        2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]).reshape(3,3)


def rotate_terminal(x,action,raw_quaternion):
    i,k=action
    if i==k or not 0<=i<len(x) or not 0<=k<len(x):raise ValueError('Distinct terminal and anchor required')
    rotation=quaternion_matrix(raw_quaternion);y=x.clone()
    y[i]=x[k]+rotation@(x[i]-x[k]);y=y-y.mean(0,keepdim=True)
    reverse=raw_quaternion*raw_quaternion.new_tensor([1.,-1.,-1.,-1.])
    return y,reverse


@torch.no_grad()
def uniform_internal_transition(target,states,*,kind,generator,phase):
    if kind not in ('rotation','exchange'):raise ValueError('Unknown internal move')
    candidates=[];rows=[]
    for old in states:
        actions=(terminal_rotation_actions(target.numbers,old['graph']['bond_orders'])
            if kind=='rotation' else old['actions'])
        row=dict(kind=kind,phase=phase,old_state_id=old['state_id'],new_state_id=-1,
            forward_count=len(actions),valid=False,accepted=False)
        candidate=None
        if actions:
            index=int(torch.randint(len(actions),(1,),generator=generator));action=actions[index]
            row.update(choice_index=index,action=action)
            if kind=='exchange':
                candidate,existing=target.propose(old,index+1,generator,.1*target.kT**.5)
                row.update(existing)
            else:
                raw=torch.randn(4,dtype=torch.float64,generator=generator)
                y,reverse=rotate_terminal(old['positions'],action,raw)
                recovered,_=rotate_terminal(y,action,reverse)
                torch.testing.assert_close(recovered,old['positions'],atol=1e-10,rtol=1e-10)
                row.update(raw_quaternion=raw,reverse_quaternion=reverse,proposal_positions=y)
                try:
                    candidate=target.coordinate_state(y)
                    if not torch.equal(candidate['graph']['bond_orders'],old['graph']['bond_orders']):
                        raise ValueError('Rotation changed the perceived bond graph')
                    if action not in terminal_rotation_actions(target.numbers,candidate['graph']['bond_orders']):
                        raise ValueError('Reverse terminal rotation not eligible')
                    row['valid']=True
                except ValueError as exc:
                    candidate=None;row['rejection_reason']=str(exc)
        else:row['rejection_reason']='No eligible internal action'
        candidates.append(candidate);rows.append(row)
    target.evaluate([s for s in candidates if s is not None],phase=phase)
    logu=torch.rand(len(states),dtype=torch.float64,generator=generator).log();updated=list(states)
    for i,(old,new,row) in enumerate(zip(states,candidates,rows)):
        row['log_uniform']=float(logu[i])
        if new is None:continue
        if kind=='exchange':
            target.finish_record(old,new,row,.1*target.kT**.5)
            reverse_count=len(new['actions']);base=float(row['base_log_ratio'])
        else:
            reverse_count=len(terminal_rotation_actions(target.numbers,new['graph']['bond_orders']))
            base=-float(new['potential_eV']-old['potential_eV'])/target.kT
            row.update(new_state_id=new['state_id'],base_log_ratio=base,log_volume=0.)
        correction=math.log(row['forward_count']/reverse_count);ratio=base+correction
        take=float(logu[i])<min(0.,ratio)
        row.update(reverse_count=reverse_count,action_log_ratio=correction,log_acceptance_ratio=ratio,accepted=take)
        if take:updated[i]=new
    return updated,rows
