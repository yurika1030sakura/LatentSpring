#!/usr/bin/env python3
"""Freeze additional proposal-training compositions without energy/geometry ranking."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(candidates, exclusions, seed=25101):
    allowed={1,6,7,8,9,15,16,17,35,53}
    halogens={9,17,35,53}
    rows=[];used=set(exclusions);counts={}
    for lo,hi in [(8,12),(13,24)]:
        eligible=[]
        for i,row in enumerate(candidates):
            z=row['atomic_numbers']
            if (row['partition']=='new_development' and lo<=len(z)<=hi
                    and row['charge']==0 and row['spin_multiplicity']==1
                    and set(z)<=allowed and 6 in z and z.count(1)>=2
                    and set(z)&halogens and sum(a not in halogens|{1} for a in z)>=2
                    and row['composition_hex'] not in used):
                rank=hashlib.sha256(f'{seed}|{i}|{row["source"]}'.encode()).hexdigest()
                eligible.append((rank,i,row))
        counts[f'{lo}-{hi}']=len({r['composition_hex'] for _,_,r in eligible})
        selected=[]
        for rank,i,row in sorted(eligible):
            if row['composition_hex'] in used:continue
            new={k:v for k,v in row.items() if k!='energy_eV'}
            new.update(candidate_index=i,panel_stratum=[lo,hi,'neutral','singlet'],training_selection_hash=rank)
            selected.append(new);used.add(row['composition_hex'])
            if len(selected)==4:break
        if len(selected)!=4:raise ValueError('Insufficient disjoint training compositions in a size stratum')
        rows.extend(selected)
    return rows,counts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();root=args.project
    cp=root/'research/evidence/official_development_candidates.json'
    op=root/'research/evidence/development_panel_v1.json'
    ep=root/'research/evidence/transfer_development_panel_v1.json'
    candidates=json.loads(cp.read_text());old=json.loads(op.read_text());evaluation=json.loads(ep.read_text())
    assert candidates['complete'] and candidates['role']=='new_development'
    assert evaluation['source_candidates_sha256']==sha(cp)
    exclusions={r['composition_hex'] for panel in [old,evaluation] for r in panel['rows']}
    rows,counts=select(candidates['rows'],exclusions)
    result=dict(complete=True,role='new_development',intended_use='proposal_training',
        frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seed=25101,
        source_candidates_sha256=sha(cp),previous_panel_sha256=sha(op),
        excluded_evaluation_manifest_sha256=sha(ep),eligible_compositions_by_size=counts,rows=rows,
        selection='Four minimum fixed hash ranks per size8-12/13-24, distinct neutral singlet C/H/halogen compositions with at least two possible heavy anchors. Exclude old eight and evaluated six compositions. No energy or geometry ranking; reference energies omitted.',
        purpose='Broaden future proposal-training coverage; generated states are training inputs, not independent evaluation. No existing failed student is promoted by preparing this source.',
        reserved_outcomes_allowed=False)
    if args.out.exists():raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(eligible=counts,rows=[{k:r[k] for k in ['candidate_index','n_atoms','atomic_numbers']} for r in rows])))


if __name__=='__main__':main()
