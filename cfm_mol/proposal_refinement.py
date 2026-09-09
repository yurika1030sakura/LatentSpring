"""Monotone pilot-center refinement; optimization is not equilibrium sampling."""
import torch
from cfm_mol.tempered_smc import DensityValue


@torch.no_grad()
def refine_centers(centers,target,steps=50,max_step=.05):
    if steps<0 or max_step<=0:raise ValueError('Invalid refinement settings')
    x=centers.detach().double().clone();current=target(x).validate(x,True)
    initial=current.log_value.clone();scale=x.new_full((len(x),1),max_step);history=[]
    for step in range(steps):
        direction=current.score/current.score.norm(dim=-1,keepdim=True).clamp_min(1.)
        candidate=x+scale*direction;value=target(candidate).validate(candidate,True)
        accept=value.log_value>=current.log_value
        x=torch.where(accept[:,None],candidate,x)
        current=DensityValue(torch.where(accept,value.log_value,current.log_value),
                             torch.where(accept[:,None],value.score,current.score))
        scale=torch.where(accept[:,None],(scale*1.05).clamp_max(max_step),scale*.5)
        history.append({'step':step+1,'accepted_centers':int(accept.sum()),
            'mean_log_target_gain':float((current.log_value-initial).mean()),'minimum_step':float(scale.min())})
    return x,{'steps':steps,'max_step':max_step,'target_evaluations':len(x)*(steps+1),'history':history,
              'minimum_log_target_gain':float((current.log_value-initial).min()),
              'scope':'all centers retained; monotone proposal optimization, not target draws'}
