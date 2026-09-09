#!/usr/bin/env python3
"""Audit official validation overlap and freeze outcome-independent candidate panels.

This checks whole-system compositions and explicit calculation/reference links.
It does not infer complete chemical or trajectory identities from path names.
Candidates are ranked by a fixed hash, never by energies or method outcomes.
"""
import argparse
from collections import Counter
import hashlib
import heapq
import json
from pathlib import Path
import sqlite3
import time

import numpy as np
import torch


def fingerprint(value):
    prefix='s3://opencatalysisdata/archive/hot/'
    if value.startswith(prefix):value=value[len(prefix):]
    return hashlib.sha256(value.encode()).digest()


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def composition(types):
    count=np.bincount(types,minlength=83)
    if len(count)!=83 or count.max()>255:raise ValueError('Unsupported composition encoding')
    return count.astype(np.uint8).tobytes()


def write(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)


class HashReservoir:
    def __init__(self,capacity):self.capacity=capacity;self.rows={};self.heap=[]
    def add(self,key,rank,row):
        old=self.rows.get(key)
        if old is not None and old[0]<=rank:return
        while self.heap and (self.heap[0][1] not in self.rows or self.rows[self.heap[0][1]][0]!=-self.heap[0][0]):heapq.heappop(self.heap)
        if old is None and len(self.rows)>=self.capacity:
            if rank>=-self.heap[0][0]:return
            _,removed=heapq.heappop(self.heap);del self.rows[removed]
        self.rows[key]=(rank,row);heapq.heappush(self.heap,(-rank,key))
        if len(self.heap)>3*self.capacity:
            self.heap=[(-rank,key) for key,(rank,_) in self.rows.items()];heapq.heapify(self.heap)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-val',type=Path,required=True);p.add_argument('--processed',type=Path,required=True)
    p.add_argument('--training-index',type=Path,required=True);p.add_argument('--validation-manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--capacity-per-stratum',type=int,default=8)
    p.add_argument('--seed',type=int,default=20270918);p.add_argument('--limit',type=int)
    args=p.parse_args()
    if args.capacity_per_stratum<1:raise ValueError('Positive reservoir capacity required')
    args.out.mkdir(parents=True,exist_ok=True);output=args.out/'audit.json'
    if output.exists():raise FileExistsError(output)
    start=time.perf_counter();training_compositions=set();old_development_compositions=set()
    checked_files={}
    for split,destination in [('train',training_compositions),('val',old_development_compositions)]:
        data=torch.load(str(args.processed/f'{split}_data_processed.pt'),map_location='cpu',mmap=True,weights_only=False)
        types=data['atom_types'].numpy()
        for lo,hi in data['node_idx_array'].numpy():destination.add(composition(types[lo:hi]))
        print(json.dumps({'phase':'old_compositions','split':split,'unique':len(destination),'seconds':time.perf_counter()-start}),flush=True)
        del data,types
        checked_files[split]=file_hash(args.processed/f'{split}_data_processed.pt')
    source_links=set()
    con=sqlite3.connect('file:'+str(args.training_index.resolve())+'?mode=ro&immutable=1',uri=True)
    indexed_state=json.loads(con.execute('SELECT value FROM metadata WHERE key="state"').fetchone()[0])
    if not indexed_state['complete']:raise ValueError('Training source replay incomplete')
    indexed_protocol=json.loads(con.execute('SELECT value FROM metadata WHERE key="protocol"').fetchone()[0])
    if checked_files!=indexed_protocol['processed_sha256']:raise ValueError('Processed data differs from the verified training index')
    for source,reference in con.execute('SELECT source,reference_source FROM records'):
        for value in [source,reference]:
            if value:source_links.add(fingerprint(value))
    con.close()
    from fairchem.core.datasets import AseDBDataset
    raw=AseDBDataset({'src':str(args.raw_val)})
    source_manifest=json.loads(args.validation_manifest.read_text())
    if not source_manifest['complete']:raise ValueError('Official validation download incomplete')
    if source_manifest.get('raw_records',len(raw))!=len(raw):raise ValueError('Validation count changed')
    counts=Counter();strata={};domain_counts=Counter();size_counts=Counter()
    report={'complete':False,'scope':__doc__,'source_raw_records':len(raw),'raw_directory':str(args.raw_val.resolve()),
        'old_training_unique_compositions':len(training_compositions),'old_development_unique_compositions':len(old_development_compositions),
        'training_source_link_hashes':len(source_links),'validation_archive_sha256':source_manifest['archive_sha256'],
        'validation_manifest_sha256':hashlib.sha256(args.validation_manifest.read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'processed_input_sha256':checked_files,
        'partition_rule':'SHA256(seed || elemental-count vector) mod 5: zero=new development, otherwise=reserved evaluation',
        'selection_rule':'minimum SHA256 ranks within composition/electronic-state and stratum; no outcome-based ranking',
        'capacity_per_stratum':args.capacity_per_stratum,'seed':args.seed,'max_atoms':200,
        'remaining_limitations':['explicit source links do not exhaust parent-trajectory relations',
            'candidate selection is not evidence of method quality','official validation is not relabeled as an official test set']}
    def save(complete=False):
        report.update(complete=complete,counts=dict(counts),domains=dict(domain_counts),sizes=dict(size_counts),seconds=time.perf_counter()-start)
        write(output,report)
    save()
    end=min(len(raw),args.limit) if args.limit else len(raw)
    for index in range(end):
        if index and index%20000==0:
            save();print(json.dumps({'scanned':index,'eligible':counts['eligible'],'seconds':time.perf_counter()-start}),flush=True)
        counts['scanned']+=1
        try:atoms=raw.get_atoms(index)
        except Exception:counts['read_errors']+=1;continue
        n=len(atoms)
        if not 2<=n<=200:counts['outside_atom_range']+=1;continue
        numbers=atoms.numbers
        if (numbers<1).any() or (numbers>83).any():counts['unsupported_element']+=1;continue
        key=composition(numbers-1)
        if key in training_compositions:counts['overlap_old_training_composition']+=1;continue
        if key in old_development_compositions:counts['overlap_old_development_composition']+=1;continue
        info=atoms.info;source=info.get('source');reference=info.get('reference_source')
        if not source:counts['missing_source']+=1;continue
        if any(value and fingerprint(value) in source_links for value in [source,reference]):
            counts['overlap_old_explicit_source_link']+=1;continue
        if info.get('charge') is None or info.get('spin') is None:counts['missing_electronic_state']+=1;continue
        charge=int(info['charge']);spin=int(info['spin']);electrons=int(numbers.sum())-charge
        if spin<1 or electrons<1 or electrons<spin-1 or (electrons-spin+1)%2:counts['invalid_electron_parity']+=1;continue
        if atoms.calc is None or 'energy' not in atoms.calc.results or 'forces' not in atoms.calc.results:
            counts['missing_labels']+=1;continue
        energy=float(atoms.get_potential_energy());forces=atoms.get_forces()
        if not np.isfinite(energy) or not np.isfinite(forces).all():counts['nonfinite_labels']+=1;continue
        counts['eligible']+=1
        partition='new_development' if int.from_bytes(hashlib.sha256(str(args.seed).encode()+key).digest()[:8],'big')%5==0 else 'reserved_evaluation'
        size=next(label for limit,label in [(12,'2-12'),(24,'13-24'),(40,'25-40'),(80,'41-80'),(200,'81-200')] if n<=limit)
        domain=str(info.get('data_id','unknown'));domain_counts[domain]+=1;size_counts[size]+=1;counts[partition]+=1
        stratum=(partition,size,domain,'neutral' if charge==0 else 'charged','singlet' if spin==1 else 'open_shell')
        rank=int.from_bytes(hashlib.sha256(str(args.seed).encode()+key+source.encode()+f'|{charge}|{spin}'.encode()).digest()[:8],'big')
        row={'raw_index':index,'source':source,'reference_source':reference,'data_id':domain,'n_atoms':n,
            'atomic_numbers':numbers.tolist(),'charge':charge,'spin_multiplicity':spin,'energy_eV':energy,
            'composition_hex':key.hex(),'partition':partition,'stratum':list(stratum),'selection_rank':rank}
        reservoir=strata.setdefault(stratum,HashReservoir(args.capacity_per_stratum))
        reservoir.add((key,charge,spin),rank,row)
    candidates=[row for pool in strata.values() for _,row in pool.rows.values()]
    candidates.sort(key=lambda row:(row['partition'],row['stratum'],row['selection_rank']))
    for partition in ['new_development','reserved_evaluation']:
        rows=[row for row in candidates if row['partition']==partition]
        write(args.out/f'{partition}.json',{'complete':end==len(raw),'role':partition,'seed':args.seed,
            'source_archive_sha256':source_manifest['archive_sha256'],'rows':rows,
            'use_policy':'no method-outcome queries before protocol freeze' if partition=='reserved_evaluation' else 'development only'})
    report['candidate_counts']={p:sum(row['partition']==p for row in candidates) for p in ['new_development','reserved_evaluation']}
    save(end==len(raw));print(json.dumps({k:v for k,v in report.items() if k not in ['scope']}),flush=True)


if __name__=='__main__':main()
