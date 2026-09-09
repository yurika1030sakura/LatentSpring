"""Read only fully verified raw metadata; historical charge features stay unchanged."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from cfm_mol.electronic_conditioning import attach_electronic_state


class ElectronicMetadata:
    def __init__(self,directory,split,atomic_numbers_by_type):
        self.directory=Path(directory)
        raw=(self.directory/'progress.json').read_bytes();self.state=json.loads(raw)
        if not self.state['complete']:raise ValueError('Electronic metadata replay is incomplete')
        if split not in ['train','val']:raise ValueError('Use a real split; the legacy test duplicates val')
        self.progress_sha256=hashlib.sha256(raw).hexdigest();self.split=split
        self.indices=np.load(self.directory/f'{split}_accepted_indices.npy',mmap_mode='r')
        if len(self.indices)!=self.state['processed_seen'][split]:raise ValueError('Metadata split length mismatch')
        self.values={name:np.load(self.directory/f'{name}.npy',mmap_mode='r') for name in
                     ['total_charge','spin_multiplicity','charge_known','spin_known','raw_indices','energy_float64']}
        if any(len(value)!=self.state['accepted'] for value in self.values.values()):raise ValueError('Metadata array length mismatch')
        self.atomic_numbers=torch.tensor(atomic_numbers_by_type,dtype=torch.long)

    def verify_processed_file(self,path):
        digest=hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
        if digest.hexdigest()!=self.state['protocol']['processed_sha256'][self.split]:
            raise ValueError('Electronic metadata belongs to a different processed tensor file')

    def attach(self,graph,processed_indices,kT_eV):
        rows=np.asarray(processed_indices,dtype=np.int64)
        if rows.shape!=(graph.batch_size,) or (rows<0).any() or (rows>=len(self.indices)).any():raise ValueError('Invalid metadata row selection')
        selected=self.indices[rows]
        if not self.values['charge_known'][selected].all() or not self.values['spin_known'][selected].all():
            raise ValueError('Missing original charge/spin metadata')
        types=graph.ndata['a_1_true'].argmax(-1).cpu()
        if (types>=len(self.atomic_numbers)).any():raise ValueError('Unrecognized atomic species')
        return attach_electronic_state(graph,self.values['total_charge'][selected].copy(),
            self.values['spin_multiplicity'][selected].copy(),kT_eV,
            atomic_numbers=self.atomic_numbers[types].to(graph.device))
