#!/usr/bin/env python3
"""Replay legacy preprocessing and recover original OMol metadata only on exact matches.

Read-only inputs. Outputs are a resumable SQLite source index and dtype-preserving
NumPy sidecars; never overwrite the historical processed tensors. A partial
index is not a complete restored dataset. Source paths identify calculations,
not automatically unique chemical identities or independent trajectory groups.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import numpy as np
import torch


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write_json(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,required=True);p.add_argument('--processed',type=Path,required=True)
    p.add_argument('--archive-manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--max-atoms',type=int,default=200)
    p.add_argument('--limit',type=int);p.add_argument('--resume',action='store_true');p.add_argument('--checkpoint-every',type=int,default=20000)
    args=p.parse_args()
    if args.max_atoms!=200 or args.checkpoint_every<1:raise ValueError('This replay targets the archived max_atoms=200 protocol')
    args.out.mkdir(parents=True,exist_ok=True);database=args.out/'source_index.sqlite'
    if database.exists() and not args.resume:raise FileExistsError('Use --resume for the same verified source')
    if args.resume and not database.exists():raise FileNotFoundError(database)
    from fairchem.core.datasets import AseDBDataset
    raw=AseDBDataset({'src':str(args.raw)})
    paths={name:args.processed/f'{name}_data_processed.pt' for name in ['train','val']}
    legacy={name:torch.load(str(path),map_location='cpu',mmap=True,weights_only=False) for name,path in paths.items()}
    lengths={name:len(value['node_idx_array']) for name,value in legacy.items()};expected=sum(lengths.values())
    print(json.dumps({'phase':'hash_inputs','raw_records':len(raw),'expected_accepted':expected}),flush=True)
    protocol={'raw_directory':str(args.raw.resolve()),'raw_records':len(raw),'archive_manifest_sha256':sha(args.archive_manifest),
        'processed_sha256':{name:sha(path) for name,path in paths.items()},'expected_processed_counts':lengths,
        'seed':args.seed,'max_atoms':args.max_atoms,'script_sha256':sha(__file__),
        'comparison':'bitwise centered float32 coordinates, float32 forces/energy, atom types, legacy clipped atom-zero charge',
        'scope':'deterministic source replay, not a chemical identity split certificate'}
    mask=np.zeros(expected,dtype=bool);mask[np.random.default_rng(args.seed).permutation(expected)[:lengths['val']]]=True
    connection=sqlite3.connect(database)
    connection.execute('PRAGMA journal_mode=WAL');connection.execute('PRAGMA synchronous=FULL')
    connection.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
    connection.execute('CREATE TABLE IF NOT EXISTS records (raw_index INTEGER PRIMARY KEY, accepted_index INTEGER UNIQUE NOT NULL, split TEXT NOT NULL, processed_index INTEGER NOT NULL, source TEXT, reference_source TEXT, data_id TEXT, sid TEXT, charge INTEGER, spin INTEGER, n_atoms INTEGER NOT NULL, energy_eV REAL, UNIQUE(split,processed_index))')
    columns={'raw_indices':np.int64,'energy_float64':np.float64,'total_charge':np.int16,'spin_multiplicity':np.int16,
             'charge_known':np.bool_,'spin_known':np.bool_}
    if args.resume:
        old=json.loads(connection.execute('SELECT value FROM metadata WHERE key="protocol"').fetchone()[0])
        if old!=protocol:raise ValueError('Input or replay protocol changed; refusing resume')
        state=json.loads(connection.execute('SELECT value FROM metadata WHERE key="state"').fetchone()[0])
        arrays={name:np.lib.format.open_memmap(args.out/f'{name}.npy',mode='r+') for name in columns}
    else:
        state={'next_raw_index':0,'accepted':0,'processed_seen':{'train':0,'val':0},'skips':{},'charge_counts':{},'spin_counts':{},
               'legacy_charge_clipped':0,'missing_charge':0,'missing_spin':0,'nonfinite_labels':0,'complete':False}
        arrays={name:np.lib.format.open_memmap(args.out/f'{name}.npy',mode='w+',dtype=dtype,shape=(expected,)) for name,dtype in columns.items()}
        np.save(args.out/'train_accepted_indices.npy',np.flatnonzero(~mask));np.save(args.out/'val_accepted_indices.npy',np.flatnonzero(mask))
        connection.execute('INSERT INTO metadata VALUES (?,?)',('protocol',json.dumps(protocol)))
        connection.execute('INSERT INTO metadata VALUES (?,?)',('state',json.dumps(state)));connection.commit()
    if state['complete']:print(json.dumps(state),flush=True);connection.close();return
    buffer=[];start=time.perf_counter();starting_raw=state['next_raw_index']
    def checkpoint():
        for value in arrays.values():value.flush()
        connection.executemany('INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',buffer)
        connection.execute('UPDATE metadata SET value=? WHERE key="state"',(json.dumps(state),));connection.commit();buffer.clear()
        report={'protocol':protocol,**state,'seconds_this_invocation':time.perf_counter()-start}
        write_json(args.out/'progress.json',report)
        print(json.dumps({k:v for k,v in report.items() if k not in ['protocol','charge_counts','spin_counts']}),flush=True)
    def skip(reason):state['skips'][reason]=state['skips'].get(reason,0)+1
    end=min(len(raw),args.limit) if args.limit is not None else len(raw)
    for raw_index in range(state['next_raw_index'],end):
        state['next_raw_index']=raw_index+1
        try:atoms=raw.get_atoms(raw_index)
        except Exception:skip('read_error');continue
        n=len(atoms);numbers=atoms.numbers
        if n<2 or n>args.max_atoms:skip('size');continue
        if (numbers<1).any() or (numbers>83).any():skip('element');continue
        if atoms.calc is None or 'energy' not in atoms.calc.results or 'forces' not in atoms.calc.results:skip('missing_labels');continue
        try:energy=float(atoms.get_potential_energy());forces=atoms.get_forces().astype(np.float32)
        except Exception:skip('missing_labels');continue
        if forces.shape!=(n,3):skip('missing_labels');continue
        accepted=state['accepted']
        if accepted>=expected:raise ValueError('More accepted raw records than legacy preprocessing')
        split='val' if mask[accepted] else 'train';index=state['processed_seen'][split];old=legacy[split]
        lo,hi=map(int,old['node_idx_array'][index]);positions=atoms.positions.astype(np.float32);positions-=positions.mean(0,keepdims=True)
        raw_charge=atoms.info.get('charge');charge=int(raw_charge) if raw_charge is not None else None
        raw_spin=atoms.info.get('spin');spin=int(raw_spin) if raw_spin is not None else None
        charges=np.zeros(n,dtype=np.int8);charges[0]=np.clip(charge if charge is not None else 0,-2,3)
        checks={'n_atoms':hi-lo==n,'positions':np.array_equal(positions,old['positions'][lo:hi].numpy(),equal_nan=True),
            'forces':np.array_equal(forces,old['forces'][lo:hi].numpy(),equal_nan=True),
            'atom_types':np.array_equal((numbers-1).astype(np.int8),old['atom_types'][lo:hi].numpy()),
            'atom_charges':np.array_equal(charges,old['atom_charges'][lo:hi].numpy()),
            'energy':np.array_equal(np.array(energy,dtype=np.float32),old['energies'][index].numpy(),equal_nan=True)}
        if not all(checks.values()):
            state['next_raw_index']=raw_index
            state['error']={'raw_index':raw_index,'accepted_index':accepted,'split':split,'processed_index':index,'checks':checks}
            checkpoint();raise ValueError('Raw replay mismatch; no metadata assigned to the mismatching record')
        for name,value in [('raw_indices',raw_index),('energy_float64',energy),('total_charge',charge or 0),
                           ('spin_multiplicity',spin or 0),('charge_known',charge is not None),('spin_known',spin is not None)]:
            arrays[name][accepted]=value
        buffer.append((raw_index,accepted,split,index,atoms.info.get('source'),atoms.info.get('reference_source'),
            atoms.info.get('data_id'),str(atoms.info.get('sid')),charge,spin,n,energy))
        state['accepted']+=1;state['processed_seen'][split]+=1
        state['legacy_charge_clipped']+=int(charge is not None and (charge<-2 or charge>3))
        state['missing_charge']+=int(charge is None);state['missing_spin']+=int(spin is None)
        state['nonfinite_labels']+=int(not np.isfinite(energy) or not np.isfinite(forces).all())
        for name,value in [('charge_counts',charge),('spin_counts',spin)]:
            key=str(value);state[name][key]=state[name].get(key,0)+1
        if len(buffer)>=args.checkpoint_every:checkpoint()
    if state['next_raw_index']==len(raw):
        if state['accepted']!=expected or state['processed_seen']!=lengths:raise ValueError('Final replay count mismatch')
        state['complete']=True
    checkpoint();connection.close()


if __name__=='__main__':main()
