#!/usr/bin/env python3
"""Select eight condition strata by fixed hash, before method outcome queries."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20270919);args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    data=json.loads(args.candidates.read_text())
    if not data['complete'] or data['role']!='new_development':raise ValueError('Only completed development candidates are eligible')
    rows=[];used=set();counts={}
    for lo,hi in [(2,12),(13,24)]:
        for neutral,singlet in [(True,True),(True,False),(False,True),(False,False)]:
            candidates=[(i,row) for i,row in enumerate(data['rows']) if lo<=row['n_atoms']<=hi
                and (row['charge']==0)==neutral and (row['spin_multiplicity']==1)==singlet
                and row['composition_hex'] not in used]
            if not candidates:raise ValueError('No candidate for required stratum')
            def rank(item):
                i,row=item;return hashlib.sha256(f'{args.seed}|{i}|{row["source"]}'.encode()).hexdigest()
            index,row=min(candidates,key=rank);row=dict(row);row['candidate_index']=index
            row['panel_stratum']=[lo,hi,'neutral' if neutral else 'charged','singlet' if singlet else 'open_shell']
            rows.append(row);used.add(row['composition_hex']);counts[str(row['panel_stratum'])]=len(candidates)
    report={'complete':True,'role':'new_development','seed':args.seed,'source_candidates':str(args.candidates.resolve()),
        'source_candidates_sha256':hashlib.sha256(args.candidates.read_bytes()).hexdigest(),
        'selection':'One minimum fixed hash per size/charge/spin stratum, excluding already selected compositions; no method/energy ranking.',
        'candidate_counts_at_selection':counts,'rows':rows}
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
