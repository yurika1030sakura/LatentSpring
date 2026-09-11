#!/usr/bin/env python3
"""Offline exact discrete-action enumeration; no development coordinates loaded."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from cfm_mol.chemical_policy import ChemicalMovePolicy,accepted_action_mass
from cfm_mol.chemical_sampler import pack_policy_states


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--table',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--replica',type=int,choices=[0,1],required=True)
    args=p.parse_args();root=Path(__file__).resolve().parents[2]
    protocol_path=root/'research/evidence/chemical_policy_protocol_v1.json';protocol=json.loads(protocol_path.read_text())
    header=json.loads((args.table/'results.json').read_text());table_path=args.table/'training.pt'
    if not header['complete'] or header['protocol_sha256']!=sha(protocol_path) or sha(table_path)!=header['artifacts']['training']:
        raise ValueError('Training-table provenance failed')
    table=torch.load(table_path,map_location='cpu',weights_only=False)
    if table['stream']!='training' or table['protocol_sha256']!=sha(protocol_path):raise ValueError('Training stream required')
    rows=table['table'];warm=table['warm_state_ids'];all_states=table['states']
    identifiers=sorted(set(warm+[r['new_state_id'] for r in rows if r['valid']]))
    lookup={v:i for i,v in enumerate(identifiers)};states=[all_states[i] for i in identifiers]
    packed=pack_policy_states(states,table['condition']['numbers'],table['kT_eV'])
    old=torch.tensor([lookup[r['old_state_id']] for r in rows]);action=torch.tensor([r['action_index'] for r in rows])
    new=torch.tensor([lookup[r['new_state_id']] if r['valid'] else lookup[r['old_state_id']] for r in rows])
    reverse=torch.tensor([r['reverse_action_index'] if r['valid'] else 0 for r in rows])
    base=torch.stack([r['base_log_ratio'] for r in rows]);utility=torch.tensor([r['utility'] for r in rows],dtype=torch.float64)
    weights=torch.tensor([r['quadrature_weight'] for r in rows],dtype=torch.float64)
    # Ensure no missing or duplicated exchange action and normalized local quadrature.
    for i,state_id in enumerate(warm):
        group=[r for r in rows if r['source_index']==i]
        for a in range(len(all_states[state_id]['actions'])+1):
            matching=[r for r in group if r['action_index']==a]
            if abs(sum(r['quadrature_weight'] for r in matching)-1)>1e-12:raise ValueError('Incomplete action quadrature')
    torch.manual_seed(protocol['training_seeds'][args.replica])
    model=ChemicalMovePolicy(**protocol['policy']).double()
    optimizer=torch.optim.Adam(model.parameters(),lr=protocol['learning_rate'])
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'results.json').exists():raise FileExistsError(args.out/'results.json')
    initial={k:v.detach().clone() for k,v in model.state_dict().items()};history=[];start=time.perf_counter()
    for step in range(protocol['training_epochs']+1):
        optimizer.zero_grad();logp=model(*packed)
        mass=accepted_action_mass(logp[old,action],logp[new,reverse],base)
        value=(mass*utility*weights).sum()/len(warm)
        if not torch.isfinite(value):raise ValueError('Nonfinite policy objective')
        if step%20==0 or step==protocol['training_epochs']:
            row=dict(step=step,empirical_accepted_utility=float(value.detach()),
                warm_mean_local_probability=float(logp[[lookup[i] for i in warm],0].exp().mean().detach()))
            history.append(row);print(json.dumps(row),flush=True)
        if step==protocol['training_epochs']:break
        (-value).backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Nonfinite policy gradient')
        torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step()
    checkpoint=dict(configuration=model.configuration,state_dict=model.state_dict(),initial_state_dict=initial,
        protocol_sha256=sha(protocol_path),training_table_sha256=sha(table_path),replica=args.replica,
        training_seed=protocol['training_seeds'][args.replica],training_parent_ids=table['source_parent_ids'])
    torch.save(checkpoint,args.out/'policy.pt')
    report=dict(complete=True,replica=args.replica,training_table_sha256=sha(table_path),protocol_sha256=sha(protocol_path),
        checkpoint_sha256=sha(args.out/'policy.pt'),parameters=sum(p.numel() for p in model.parameters()),
        inherited_training_raw_queries=header['streams']['training']['raw_queries'],new_physical_queries=0,
        seconds=time.perf_counter()-start,history=history,development_coordinates_loaded=False,
        reference_coordinates_loaded=False,scientific_submission_ready=False)
    (args.out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
