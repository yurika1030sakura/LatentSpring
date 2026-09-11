#!/usr/bin/env python3
"""Export exact frozen molecular parents and Torch RNG streams for a JAX control."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from cfm_mol.entropy_source import load_entropy_source


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True);p.add_argument('--condition-index',type=int,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[2]
    source=load_entropy_source(args.source_root,args.condition_index,
        protocol_path=root/'research/evidence/species_breadth_source_protocol_v2.json',
        manifest_path=root/'research/evidence/development_panel_v1.json')
    target=root/'research/evidence/parity_training_protocol_v1.json';protocol=json.loads(target.read_text())
    arrays={'training_positions':source['training']['positions'].numpy(),
        'development_positions':source['development']['positions'].numpy(),
        'development_parent_ids':np.asarray(source['development']['sample_ids'])}
    for replica in [0,1]:
        selection=torch.Generator().manual_seed(9141+replica)
        orientation=torch.Generator().manual_seed(9205+replica)
        arrays[f'indices_s{replica}']=torch.stack([torch.randint(4096,(16,),generator=selection) for _ in range(1000)]).numpy()
        arrays[f'signs_s{replica}']=torch.stack([2*torch.randint(2,(16,),generator=orientation)-1 for _ in range(1000)]).numpy()
    evaluation=torch.Generator().manual_seed(9207+args.condition_index)
    arrays['evaluation_signs']=(2*torch.randint(2,(512,),generator=evaluation)-1).numpy()
    file=args.out/'arrays.npz';np.savez(file,**arrays)
    report={'complete':True,'scope':__doc__,'condition':source['condition'],'condition_index':args.condition_index,
        'source_protocol_sha256':source['source_protocol_sha256'],'source_results_sha256':source['source_results_sha256'],
        'source_checkpoint_sha256':source['source_checkpoint_sha256'],
        'training_sha256':source['training_sha256'],'evaluation_sha256':source['evaluation_sha256'],
        'refinement_protocol_sha256':sha(target),'target':protocol,'arrays_sha256':sha(file),
        'energy_zero_eV':float(source['training']['energy_eV'].mean()),
        'streams':'Torch CPU generators: selection9141+replica, orientation9205+replica, evaluation9207+condition',
        'additional_oracle_evaluations':0,'source_generation_queries_additional':source['source']['oracle_evaluations']}
    (args.out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'condition':args.condition_index,'train':4096,'development':512,'new_oracle_queries':0}))


if __name__=='__main__':main()
