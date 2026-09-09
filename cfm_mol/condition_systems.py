"""Audited molecule conditions without reference-coordinate initialization."""
import hashlib
import json
from pathlib import Path

import dgl
import torch
from ase.data import atomic_numbers as atomic_number_table
from torch.nn.functional import one_hot

from cfm_mol.electronic_conditioning import attach_electronic_state


def load_condition(path,index,*,evaluation_protocol=None,method_id=None):
    path=Path(path);raw=path.read_bytes();manifest=json.loads(raw)
    if not manifest['complete']:raise ValueError('Condition manifest is incomplete')
    role=manifest.get('role')
    if role=='reserved_evaluation':
        if evaluation_protocol is None:raise ValueError('Reserved outcomes require a frozen evaluation protocol')
        protocol=json.loads(Path(evaluation_protocol).read_text())
        if (protocol.get('frozen') is not True or protocol.get('condition_manifest_sha256')!=hashlib.sha256(raw).hexdigest()
                or method_id is None or method_id not in protocol.get('method_ids',[])):
            raise ValueError('Frozen evaluation protocol does not match this manifest and method')
    elif role!='new_development':raise ValueError('Unknown condition-manifest role')
    if not 0<=index<len(manifest['rows']):raise IndexError(index)
    row=dict(manifest['rows'][index])
    row['manifest_sha256']=hashlib.sha256(raw).hexdigest();row['manifest_index']=index
    row['manifest_role']=role
    return row


def graph_from_condition(condition,atom_map,*,device='cpu',dtype=torch.float32,n_charge_classes=6,n_bond_classes=4):
    supplied=torch.as_tensor(condition['atomic_numbers'])
    if supplied.ndim!=1 or not torch.isfinite(supplied).all() or not torch.equal(supplied,supplied.round()):
        raise ValueError('Atomic numbers must be finite integers')
    numbers=supplied.long()
    n=len(numbers)
    if not 2<=n<=200 or condition.get('n_atoms',n)!=n:raise ValueError('Condition must have 2--200 atoms')
    mapping={atomic_number_table[symbol]:i for i,symbol in enumerate(atom_map)}
    if any(int(z) not in mapping for z in numbers):raise ValueError('Condition contains unsupported elements')
    src,dst=torch.where(~torch.eye(n,dtype=torch.bool))
    graph=dgl.graph((src,dst),num_nodes=n).to(device)
    types=torch.tensor([mapping[int(z)] for z in numbers],device=device)
    atoms=one_hot(types,len(atom_map)).to(dtype)
    charges=torch.full((n,),2,dtype=torch.long,device=device)
    # Retained solely as the explicitly declared legacy-model interface.
    # The global electronic adapter neutralizes this marker during evaluation.
    charges[0]=max(-2,min(3,int(condition['charge'])))+2
    c=one_hot(charges,n_charge_classes).to(dtype)
    e=one_hot(torch.zeros(len(src),dtype=torch.long,device=device),n_bond_classes).to(dtype)
    graph.ndata['x_1_true']=torch.zeros(n,3,device=device,dtype=dtype)
    graph.ndata['x_t']=graph.ndata['x_1_true'].clone()
    graph.ndata['has_reference_geometry']=torch.zeros(n,1,dtype=torch.bool,device=device)
    for key,value in [('a',atoms),('c',c)]:
        graph.ndata[key+'_1_true']=value;graph.ndata[key+'_t']=value.clone()
    graph.edata['e_1_true']=e;graph.edata['e_t']=e.clone()
    attach_electronic_state(graph,condition['charge'],condition['spin_multiplicity'],
        condition.get('requested_kT_eV',1.),atomic_numbers=numbers.to(device))
    return graph
