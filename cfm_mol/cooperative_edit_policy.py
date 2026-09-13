"""Fixed-root-block cooperative proposals with symmetric learned edge affinity.

Draw four type-admissible terminal roots uniformly for a complete sampler. Their set and count
are unchanged by these degree-preserving terminal exchanges, so the outer block
probability cancels. This module implements the conditional kernel on that block.
The four-state interaction is an undirected affinity, NOT an additive correction
to the directional total work: reversing both edits preserves its sign.
"""
import itertools
import math
import torch
from cfm_mol.chemical_edit_interaction import independent_actions,four_positions,four_graphs,restraint_interaction
from cfm_mol.joint_chemical_geometry import distinct_anchor_actions


def terminal_roots(numbers,bonds):
    degree=(bonds>0).sum(1)
    return [i for i,z in enumerate(numbers) if int(z) in (1,9,17,35,53) and int(degree[i])==1]


def sample_root_blocks(numbers,bonds,maximum,generator):
    """Uniform distinct blocks that admit two unlike-element pairs.

    Type-count conditioning depends only on the unchanged terminal set, so its
    outer probability still cancels. Geometry/anchor-based filtering is absent.
    Count patterns avoid enumerating O(N^4) blocks or rare-event rejection.
    """
    leaves=terminal_roots(numbers,bonds);types=sorted(set(int(numbers[i]) for i in leaves))
    groups=[[i for i in leaves if int(numbers[i])==z] for z in types]
    patterns=[];weights=[]
    for pattern in itertools.product(range(3),repeat=len(groups)):
        if sum(pattern)!=4 or any(k>len(g) for k,g in zip(pattern,groups)):continue
        patterns.append(pattern);weights.append(math.prod(math.comb(len(g),k) for g,k in zip(groups,pattern)))
    total=sum(weights)
    if not total:return [],0
    weights=torch.tensor(weights,dtype=torch.float64);chosen=set()
    while len(chosen)<min(maximum,total):
        pattern=patterns[int(torch.multinomial(weights,1,generator=generator))]
        block=[]
        for group,k in zip(groups,pattern):
            block.extend(group[j] for j in torch.randperm(len(group),generator=generator)[:k].tolist())
        chosen.add(tuple(sorted(block)))
    return sorted(chosen),total


@torch.no_grad()
def block_catalogue(target,old,roots):
    roots=tuple(sorted(roots))
    if len(roots)!=4 or len(set(roots))!=4 or not set(roots)<=set(terminal_roots(target.numbers,old['graph']['bond_orders'])):
        raise ValueError('The block must contain four distinct terminal roots')
    actions=[a for a in distinct_anchor_actions(target.numbers,old['graph']['bond_orders']) if set(a[:2])<=set(roots)]
    valid=[];failed=[]
    for a,b in itertools.combinations(actions,2):
        if not independent_actions(a,b):continue
        corners,volume,inverse=four_positions(old['positions'],target.radii,a,b)
        graphs=four_graphs(old['graph']['bond_orders'],a,b)
        record=dict(actions=[a,b],inverse_actions=list(inverse),log_volume=float(volume),roots=roots)
        try:
            states=[target.coordinate_state(x) for x in corners[1:]]
            if any(not torch.equal(s['graph']['bond_orders'],graphs[i+1]) for i,s in enumerate(states)):
                raise ValueError('A corner graph differs from the prescribed edits')
            assert terminal_roots(target.numbers,states[-1]['graph']['bond_orders'])==terminal_roots(target.numbers,old['graph']['bond_orders'])
            valid.append(dict(record=record,positions=corners,graphs=graphs,candidate=states[-1]))
        except ValueError as exc:failed.append(dict(record=record,failure=str(exc)))
    assert len(valid)+len(failed)<=3
    return dict(roots=roots,valid=valid,failed=failed)


@torch.no_grad()
def block_policy(target,old,catalogue,backbone=None,interaction=None,*,use_affinity=False,
                 scale_eV=.25,bound=1.,uniform_fraction=.1):
    rows=catalogue['valid'];n=len(rows)
    if not n:return dict(log_probability=torch.empty(0,dtype=torch.float64))
    x=old['positions'][None].expand(n,-1,-1);numbers=torch.tensor(target.numbers,dtype=torch.long)
    electronic=x.new_tensor([old['charge'],old['spin_multiplicity'],target.kT]).expand(n,-1)
    corners=torch.stack([row['positions'] for row in rows]);graphs=torch.stack([row['graphs'] for row in rows])
    actions=torch.tensor([row['record']['actions'] for row in rows]);volume=x.new_tensor([row['record']['log_volume'] for row in rows])
    if backbone is None:
        work=x.new_zeros(n);log_weight=x.new_zeros(n)
    else:
        work=backbone(x,corners[:,3],graphs[:,0],graphs[:,3],numbers,electronic,actions[:,0,:2])
        log_weight=torch.nn.functional.logsigmoid(-work/target.kT+volume)
    electronic_interaction=x.new_zeros(n) if interaction is None else interaction(corners,graphs,numbers,electronic,actions)
    total=electronic_interaction+restraint_interaction(corners,target.restraint)
    if not scale_eV>0 or not bound>0 or not 0<uniform_fraction<=1:raise ValueError('Invalid cooperative policy parameters')
    affinity=bound*torch.tanh(-total/(bound*scale_eV)) if use_affinity else torch.zeros_like(total)
    logits=log_weight+affinity
    if not torch.isfinite(logits).all():raise ValueError('Nonfinite block policy')
    lp=logits-torch.logsumexp(logits,0)
    if uniform_fraction==1:lp=torch.full_like(lp,-math.log(n))
    else:lp=torch.logaddexp(lp+math.log1p(-uniform_fraction),torch.full_like(lp,math.log(uniform_fraction/n)))
    return dict(log_probability=lp,predicted_linear_work_eV=work,electronic_interaction_eV=electronic_interaction,
                total_interaction_eV=total,log_affinity=affinity)


def reverse_index(forward,reverse):
    inverse={tuple(a) for a in forward['record']['inverse_actions']}
    indices=[i for i,row in enumerate(reverse['valid']) if {tuple(a) for a in row['record']['actions']}==inverse]
    if len(indices)!=1:raise ValueError('The reverse block omits or duplicates the inverse matching')
    index=indices[0]
    torch.testing.assert_close(reverse['valid'][index]['candidate']['positions'],forward['positions'][0],atol=1e-9,rtol=0)
    return index
