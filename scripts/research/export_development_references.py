#!/usr/bin/env python3
"""Recover raw development reference labels for assessment, not initialization."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    data=json.loads(args.manifest.read_text())
    if not data['complete'] or data['role']!='new_development':raise ValueError('Only development reference export is allowed')
    from fairchem.core.datasets import AseDBDataset
    raw=AseDBDataset({'src':str(args.raw)});rows=[]
    for index,condition in enumerate(data['rows']):
        atoms=raw.get_atoms(condition['raw_index']);info=atoms.info
        if atoms.numbers.tolist()!=condition['atomic_numbers'] or int(info['charge'])!=condition['charge'] or int(info['spin'])!=condition['spin_multiplicity'] or info['source']!=condition['source']:
            raise ValueError('Raw condition identity mismatch')
        energy=float(atoms.get_potential_energy());force=atoms.get_forces()
        if energy!=condition['energy_eV'] or not np.isfinite(force).all():raise ValueError('Raw energy/force mismatch')
        positions=atoms.positions.astype(np.float64);positions-=positions.mean(0)
        rows.append({'panel_index':index,'condition':condition,'positions':positions.tolist(),
            'force_eV_A':force.tolist(),'energy_eV':energy,'symbols':atoms.get_chemical_symbols()})
    report={'complete':True,'role':'new_development_references','source_manifest':str(args.manifest.resolve()),
        'source_manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        'raw_directory':str(args.raw.resolve()),'raw_records':len(raw),'rows':rows,
        'scope':'Independent assessment controls only; not supplied to the generation script.'}
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'complete':True,'references':len(rows)}),flush=True)


if __name__=='__main__':main()
