#!/usr/bin/env python3
"""Find stoichiometries absent from every archived FM training geometry.

This development split is conservative: different elemental compositions
cannot be the same chemical identity. Shared compositions are NOT assumed to
be the same identity. Charge/spin provenance remains a separate requirement.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch


def key(types):
    counts = np.bincount(types, minlength=83)
    if counts.max() > 255:
        raise ValueError('Composition exceeds compact-key capacity')
    return counts.astype(np.uint8).tobytes()


def load(path):
    return torch.load(str(path),map_location='cpu',weights_only=False,mmap=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    start=time.monotonic()
    val=load(args.root/'val_data_processed.pt')
    train=load(args.root/'train_data_processed.pt')
    va=val['atom_types'].numpy(); vn=val['node_idx_array'].numpy()
    ta=train['atom_types'].numpy(); tn=train['node_idx_array'].numpy()
    val_keys=[key(va[s:e]) for s,e in vn]
    seen={k:0 for k in set(val_keys)}
    for i,(s,e) in enumerate(tn):
        k=key(ta[s:e])
        if k in seen:seen[k]+=1
        if (i+1)%250000==0:
            print(json.dumps({'training_scanned':i+1,'seconds':time.monotonic()-start}),flush=True)
    held=[i for i,k in enumerate(val_keys) if seen[k]==0]
    by_size={str(limit):[i for i in held if int(vn[i,1]-vn[i,0])<=limit] for limit in [12,24,40,80,200]}
    pert=load(args.root/'perturbation_val_n10000_s0.pt')
    pa=pert['atom_types'].numpy();pn=pert['node_idx_array'].numpy();K=int(pert['K'])
    perturbation_held=[]
    for parent in range(len(pn)//K):
        s,e=pn[parent*K]
        k=key(pa[s:e])
        if k not in seen:
            raise ValueError('Perturbation composition missing from claimed validation source')
        if seen[k]==0:
            perturbation_held.append({'parent_id':parent,'n_atoms':int(e-s),'composition_hex':k.hex()})
    report={'claim':'stoichiometry-disjoint development panel; not a new blind test set',
        'n_training':len(tn),'n_validation':len(vn),'n_validation_compositions':len(seen),
        'n_held_out_compositions':sum(count==0 for count in seen.values()),
        'validation_indices':held,'validation_indices_by_max_atoms':by_size,
        'perturbation_parents':perturbation_held,
        'source_files':{name:{'bytes':(args.root/name).stat().st_size,'mtime_ns':(args.root/name).stat().st_mtime_ns}
            for name in ['train_data_processed.pt','val_data_processed.pt','perturbation_val_n10000_s0.pt']},
        'seconds':time.monotonic()-start,
        'limitations':['The previous test file duplicates validation','No raw source IDs or spin metadata','Prior evaluation exposure remains; this is development data']}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'n_train':len(tn),'held_validation':len(held),'held_perturbation_parents':len(perturbation_held),'held_by_size':{k:len(v) for k,v in by_size.items()},'seconds':report['seconds']}),flush=True)


if __name__=='__main__':main()
