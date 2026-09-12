"""Parent-aware angular force examples from the audited TRAINING artifact only."""
import torch
from cfm_mol.nonequilibrium import centered_orthonormal_basis
from cfm_mol.terminal_rotation import terminal_rotation_actions


def training_angular_examples(data):
    if data['stream']!='training':raise ValueError('Training stream required')
    states=data['states'];parents={}
    for sid,pid in zip(data['initial_state_ids'],data['source_parent_ids']):parents[sid]=pid
    for row in data['warm_transitions']:
        if row['old_state_id'] not in parents:raise ValueError('Missing warm trajectory parent')
        if row['valid']:parents[row['new_state_id']]=parents[row['old_state_id']]
    for row in data['table']:
        if parents[row['old_state_id']]!=row['parent_id']:raise ValueError('Action-table parent mismatch')
        if row['valid']:parents[row['new_state_id']]=row['parent_id']
    if set(parents)!=set(range(len(states))):raise ValueError('Unassigned scored training states')
    numbers=data['condition']['numbers'];basis=centered_orthonormal_basis(len(numbers))
    x=torch.stack([s['positions'] for s in states]);bonds=torch.stack([s['graph']['bond_orders'] for s in states])
    full_scores=torch.einsum('nk,bkd->bnd',basis,torch.stack([s['score'] for s in states]).reshape(len(states),len(numbers)-1,3))
    records=[];targets=[];directions=[]
    for sid,s in enumerate(states):
        for leaf,anchor in terminal_rotation_actions(numbers,s['graph']['bond_orders']):
            v=x[sid,leaf]-x[sid,anchor];r=v.norm();u=v/r;f=full_scores[sid,leaf]
            targets.append(r*(f-torch.dot(f,u)*u));directions.append(u)
            records.append([sid,leaf,anchor,parents[sid]])
    examples=torch.tensor(records,dtype=torch.long)
    groups=[(examples[:,3]==pid).nonzero().flatten() for pid in data['source_parent_ids']]
    if any(len(g)==0 for g in groups):raise ValueError('A training parent has no angular examples')
    electronic=torch.tensor([data['condition']['charge'],data['condition']['spin_multiplicity'],data['kT_eV']],dtype=x.dtype)
    return dict(x=x,bonds=bonds,numbers=torch.tensor(numbers,dtype=torch.long),electronic=electronic,
        examples=examples,directions=torch.stack(directions),targets=torch.stack(targets),parent_groups=groups,
        parent_ids=data['source_parent_ids'])
