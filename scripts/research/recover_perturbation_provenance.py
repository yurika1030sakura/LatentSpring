#!/usr/bin/env python3
"""Verify archived parent indices by exact deterministic perturbation replay.

This recovers indices into processed tensors, not raw OMol source IDs, spin,
unclipped charge, or energy precision. No energy oracle is queried.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--perturbations',type=Path,required=True)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();start=time.monotonic()
    source=torch.load(str(args.source),map_location='cpu',mmap=True,weights_only=False)
    pert=torch.load(str(args.perturbations),map_location='cpu',mmap=True,weights_only=False)
    K=int(pert['K']);M=len(pert['node_idx_array'])//K
    if len(pert['node_idx_array'])!=M*K or not torch.equal(pert['group_id'],torch.arange(M).repeat_interleave(K)):
        raise ValueError('Noncontiguous parent layout; cannot use identity index map')
    if len(pert['sigmas'])!=K:raise ValueError('Sigma metadata and K differ')
    rng=np.random.default_rng(args.seed);exact=0;max_error=0.;mismatches=[]
    for parent in range(M):
        a,b=map(int,source['node_idx_array'][parent])
        x=source['positions'][a:b].numpy().astype(np.float64)
        if len(x)<2:raise ValueError('Source contains skipped singleton; identity map invalid')
        for k,sigma in enumerate(pert['sigmas']):
            replay=x+rng.normal(scale=sigma,size=x.shape)
            replay=(replay-replay.mean(0)).astype(np.float32)
            j=parent*K+k;c,d=map(int,pert['node_idx_array'][j])
            same_labels=torch.equal(source['atom_types'][a:b],pert['atom_types'][c:d]) and torch.equal(source['atom_charges'][a:b],pert['atom_charges'][c:d])
            actual=pert['positions'][c:d].numpy()
            same_shape=replay.shape==actual.shape
            error=float(np.max(np.abs(replay-actual))) if same_shape else None
            if error is not None:max_error=max(max_error,error)
            if same_labels and same_shape and np.array_equal(replay,actual):exact+=1
            else:mismatches.append({'parent':parent,'perturbation':k,'same_labels':same_labels,'max_abs_coordinate_error_A':error})
        if (parent+1)%5000==0:print(json.dumps({'parents':parent+1,'seconds':time.monotonic()-start}),flush=True)
    report={'source':str(args.source),'perturbations':str(args.perturbations),
        'source_stat':{'bytes':args.source.stat().st_size,'mtime_ns':args.source.stat().st_mtime_ns},
        'perturbation_sha256':hashlib.sha256(args.perturbations.read_bytes()).hexdigest(),
        'source_positions_sha256_verified_prefix':hashlib.sha256(source['positions'][:int(source['node_idx_array'][M-1,1])].numpy().tobytes()).hexdigest(),
        'seed':args.seed,'numpy_version':np.__version__,'K':K,'sigmas':pert['sigmas'],
        'parents':M,'geometries':M*K,'exact_geometry_and_label_matches':exact,
        'max_abs_coordinate_error_A':max_error,'mismatches':mismatches,
        'all_exact':exact==M*K,'complete':True,'seconds':time.monotonic()-start,
        'verified_map':'perturbation parent index equals processed source row index' if exact==M*K else None,
        'limitations':['Raw OMol identifiers and spin are not recovered','Clipped charge and float32 energy precision are not repaired',
            'Processed validation still duplicates the previous test set','This is deterministic data provenance, not a new test split']}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ['parents','geometries','exact_geometry_and_label_matches','all_exact','seconds']}))
    if not report['all_exact']:raise SystemExit(1)


if __name__=='__main__':main()
