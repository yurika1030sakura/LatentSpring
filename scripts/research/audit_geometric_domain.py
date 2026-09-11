#!/usr/bin/env python3
"""Audit admissible generated starts without relabelling geometry as chemical validity."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from rdkit import Chem
from cfm_mol.entropy_source import load_entropy_source
from cfm_mol.geometric_domain import connected_nonoverlapping
from cfm_mol.geometry_diagnostics import contact_summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();root=Path(__file__).resolve().parents[2]
    if args.out.exists():raise FileExistsError(args.out)
    rows=[]
    for index in range(8):
        source=load_entropy_source(args.source_root,index,
            protocol_path=root/'research/evidence/species_breadth_source_protocol_v2.json',
            manifest_path=root/'research/evidence/development_panel_v1.json')
        numbers=source['condition']['numbers']
        radii=torch.tensor([Chem.GetPeriodicTable().GetRcovalent(z) for z in numbers],dtype=torch.float64)
        row=dict(index=index,condition=source['condition'],source_results_sha256=source['source_results_sha256'],streams={})
        for name in ['training','development']:
            x=source[name]['positions']
            valid=torch.cat([connected_nonoverlapping(x[i:i+64],radii) for i in range(0,len(x),64)])
            for i in range(32):
                independent=contact_summary(x[i].numpy(),radii.tolist())
                if bool(valid[i])!=(independent['contact_components']==1 and independent['overlap_pairs']==0):
                    raise ValueError('Tensor domain disagrees with independent geometry assessor')
            row['streams'][name]=dict(proposals=len(x),admissible=int(valid.sum()),
                admissible_indices=valid.nonzero().flatten().tolist(),fraction=float(valid.double().mean()),
                source_sha256=source['training_sha256' if name=='training' else 'evaluation_sha256'])
        rows.append(row)
        print(json.dumps(dict(index=index,admissible={k:v['admissible'] for k,v in row['streams'].items()})),flush=True)
    result=dict(complete=True,scope=__doc__,rows=rows,contact_factor=1.25,overlap_factor=.6,
        validator_sha256=hashlib.sha256((root/'cfm_mol/geometric_domain.py').read_bytes()).hexdigest(),
        additional_oracle_queries=0,scientific_submission_ready=False,
        limitations=['Connected/no-overlap is a declared geometric support, not chemical valence or fixed molecular identity.',
            'Conditioning on this support changes the target and source; old unconstrained KL results cannot be reused.',
            'Keep rejected FM proposals and source-generation costs in all denominators.'])
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
