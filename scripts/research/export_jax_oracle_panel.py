#!/usr/bin/env python3
"""Export prescribed saved parents for cross-environment oracle qualification."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[2]
    protocol=root/'research/evidence/parity_training_protocol_v1.json'
    report={'complete':False,'scope':__doc__,'protocol':json.loads(protocol.read_text()),
        'protocol_sha256':sha(protocol),'conditions':[],'outcome_selection':'first two saved parents from prespecified conditions0 and7'}
    for index in [0,7]:
        parent=(args.runs_root/'parity_entropy_condition_00_v1' if index==0 else
                args.runs_root/'parity_entropy_conditions_1_7_v1'/f'condition_{index:02d}')
        source=parent/f'condition_{index:02d}_convex_s0'
        original=source/'base_samples.pt';trained=json.loads((source/'results.json').read_text())
        if not trained['complete'] or not trained['checkpoint_loader_replay_passed']:
            raise ValueError('Require a complete qualified saved source')
        if trained['artifacts']['base_samples.pt']!=sha(original) or trained['refinement_protocol_sha256']!=report['protocol_sha256']:
            raise ValueError('Original parent or target provenance differs')
        sample=torch.load(original,map_location='cpu',weights_only=False)
        output=args.out/f'condition_{index:02d}.npz'
        np.savez(output,positions=sample['positions'][:2].numpy(),energy_eV=sample['energy_eV'][:2].numpy())
        report['conditions'].append({'index':index,'condition':sample['condition'],'panel':output.name,
            'panel_sha256':sha(output),'source_samples':str(original.resolve()),'source_sha256':sha(original),
            'source_results_sha256':sha(source/'results.json'),'sample_ids':sample['sample_ids'][:2]})
    report['complete']=True
    (args.out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([{'index':r['index'],'charge':r['condition']['charge'],'spin':r['condition']['spin_multiplicity']} for r in report['conditions']]))


if __name__=='__main__':main()
