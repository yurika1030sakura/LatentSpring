# UNVALIDATED HISTORICAL DRAFT. Not used by any reported experiment.
# Original electronic metadata is now available; use verified metadata before reactivation.
#!/usr/bin/env python3
"""Build a versioned energy shard with explicit electronic-state provenance.

Run actual potential inference in envs/omol25. Energies are predictions of the
declared potential, not recovered original OMol DFT labels. The original
precomputation script is retained for historical reproduction.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch
import yaml


def sha256(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):result.update(chunk)
    return result.hexdigest()


def load_calculator(checkpoint,device):
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit
    return FAIRChemCalculator(load_predict_unit(str(checkpoint),device=device))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--src',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--spin-policy',choices=['minimum-parity','declared-singlet'],required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--start-index',type=int,default=0)
    p.add_argument('--n-molecules',type=int,required=True,help='Number of source rows to consider, before exclusions')
    p.add_argument('--max-atoms',type=int,default=200)
    p.add_argument('--sigmas',type=float,nargs='+',default=[.03,.06,.1,.2,.4])
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--checkpoint-every',type=int,default=250)
    args=p.parse_args(argv)
    partial=args.out.with_suffix(args.out.suffix+'.partial')
    if args.out.exists() or partial.exists():raise FileExistsError('Refusing to overwrite an existing or partial shard')
    if not args.checkpoint.is_file():raise FileNotFoundError(args.checkpoint)
    if args.start_index<0 or args.n_molecules<1 or args.max_atoms<2 or args.checkpoint_every<1:
        raise ValueError('Invalid source range, atom cap or checkpoint interval')
    if len(args.sigmas)<2 or any(not np.isfinite(v) or v<=0 for v in args.sigmas):
        raise ValueError('At least two finite positive perturbation scales are required')
    import ase
    from ase.data import atomic_numbers
    src=torch.load(str(args.src),map_location='cpu',weights_only=False,mmap=True)
    config=yaml.safe_load(args.config.read_text());atom_map=config['dataset']['atom_map']
    type_to_z=np.asarray([atomic_numbers[atom_map[i]] for i in range(len(atom_map))])
    if args.start_index>=len(src['node_idx_array']):raise ValueError('Start index is outside the source')
    end=min(args.start_index+args.n_molecules,len(src['node_idx_array']))
    versions={'torch':torch.__version__,'numpy':np.__version__,'ase':ase.__version__}
    for name in ['fairchem-core']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    protocol={'schema_version':2,'source_path':str(args.src.resolve()),'source_sha256':sha256(args.src),
        'checkpoint_path':str(args.checkpoint.resolve()),'checkpoint_sha256':sha256(args.checkpoint),
        'config_sha256':sha256(args.config),'generator_sha256':sha256(__file__),
        'source_range':[args.start_index,end],'seed':args.seed,'sigmas':args.sigmas,
        'max_atoms':args.max_atoms,'spin_policy':args.spin_policy,'package_versions':versions,
        'source_identity':'processed source row, not original raw OMol identity',
        'charge_policy':'atom zero total charge; exclude legacy clipping boundaries -2 and +3',
        'spin_provenance':'declared computational convention; original electronic state unknown',
        'coordinate_policy':'center then quantize to stored float32 before energy evaluation',
        'energy_policy':'store calculator scalar as float64; this cannot increase predictor internal precision',
        'potential_claim':'predictions of declared checkpoint, not original DFT energy labels'}
    calculator=load_calculator(args.checkpoint,args.device)
    rng=np.random.default_rng(args.seed)
    positions=[];types=[];charges=[];indices=[];energies=[];groups=[];errors=[]
    parents=[];parent_charges=[];multiplicities=[];electron_counts=[];excluded=[];offset=0
    args.out.parent.mkdir(parents=True,exist_ok=True)

    def save(complete):
        if not parents:return
        payload={'positions':torch.from_numpy(np.concatenate(positions)),
            'atom_types':torch.from_numpy(np.concatenate(types)),
            'atom_charges':torch.from_numpy(np.concatenate(charges)),
            'node_idx_array':torch.tensor(indices,dtype=torch.long),
            'energies':torch.tensor(energies,dtype=torch.float64),
            'group_id':torch.tensor(groups,dtype=torch.long),'K':len(args.sigmas),'sigmas':args.sigmas,
            'parent_source_indices':torch.tensor(parents,dtype=torch.long),
            'parent_total_charge':torch.tensor(parent_charges,dtype=torch.long),
            'parent_spin_multiplicity':torch.tensor(multiplicities,dtype=torch.long),
            'parent_electron_count':torch.tensor(electron_counts,dtype=torch.long),
            'energy_success':torch.tensor([error is None for error in errors],dtype=torch.bool),
            'evaluation_errors':errors,'excluded_source_rows':excluded,
            'provenance':protocol,'complete':complete}
        temporary=partial.with_suffix(partial.suffix+'.tmp')
        torch.save(payload,temporary)
        temporary.replace(partial)
        if complete:partial.replace(args.out)

    for source_index in range(args.start_index,end):
        lo,hi=map(int,src['node_idx_array'][source_index]);n=hi-lo
        if not 2<=n<=args.max_atoms:
            excluded.append({'source_index':source_index,'reason':'atom_count','n_atoms':n});continue
        atom_types=src['atom_types'][lo:hi].numpy().astype(np.int64)
        atom_charges=src['atom_charges'][lo:hi].numpy().astype(np.int64)
        if np.any(atom_charges[1:]!=0):raise ValueError(f'Source row {source_index} violates atom-zero charge convention')
        charge=int(atom_charges[0])
        if charge<=-2 or charge>=3:
            excluded.append({'source_index':source_index,'reason':'charge_clipping_boundary','recorded_charge':charge});continue
        if np.any(atom_types<0) or np.any(atom_types>=len(type_to_z)):raise ValueError('Invalid atom type index')
        numbers=type_to_z[atom_types];electrons=int(numbers.sum())-charge
        if electrons<1:raise ValueError('Nonpositive electron count')
        if args.spin_policy=='declared-singlet' and electrons%2:
            excluded.append({'source_index':source_index,'reason':'singlet_incompatible_electron_parity'});continue
        multiplicity=1+electrons%2 if args.spin_policy=='minimum-parity' else 1
        base=src['positions'][lo:hi].numpy().astype(np.float64)
        if not np.isfinite(base).all():raise ValueError('Non-finite source coordinates')
        group=len(parents);parents.append(source_index);parent_charges.append(charge)
        multiplicities.append(multiplicity);electron_counts.append(electrons)
        for sigma in args.sigmas:
            perturbed=base+rng.normal(scale=sigma,size=base.shape)
            stored=(perturbed-perturbed.mean(0,keepdims=True)).astype(np.float32)
            # Score the coordinates that will actually be read by the learner.
            atoms=ase.Atoms(numbers=numbers,positions=stored.astype(np.float64))
            atoms.info.update(charge=charge,spin=multiplicity);atoms.calc=calculator
            error=None
            try:
                energy=float(atoms.get_potential_energy())
                if not np.isfinite(energy):raise FloatingPointError('Non-finite potential energy')
            except Exception as exc:
                energy=float('nan');error=f'{type(exc).__name__}: {str(exc)[:500]}'
            positions.append(stored);types.append(atom_types.astype(np.int8));charges.append(atom_charges.astype(np.int8))
            indices.append([offset,offset+n]);offset+=n
            energies.append(energy);groups.append(group);errors.append(error)
        if len(parents)%args.checkpoint_every==0:
            save(False)
            print(json.dumps({'parents':len(parents),'source_index':source_index,
                              'failed_energies':sum(error is not None for error in errors)}),flush=True)
    if not parents:raise ValueError('No eligible parent remains after exclusions')
    save(True)
    print(json.dumps({'complete':True,'parents':len(parents),'virtual_geometries':len(energies),
        'failed_energies':sum(error is not None for error in errors),'excluded_rows':len(excluded),
        'output':str(args.out)}),flush=True)


if __name__=='__main__':main()
