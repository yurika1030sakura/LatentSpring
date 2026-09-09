#!/usr/bin/env python3
"""Report contact and diversity diagnostics for every stored generated geometry."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import statistics as st
from rdkit import Chem
from cfm_mol.geometry_diagnostics import distance_profile,profile_rms,contact_summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples',type=Path,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();periodic=Chem.GetPeriodicTable();sources=[];rows=[];diversity=[]
    for path in args.samples:
        report=json.loads(path.read_text())
        if not report['complete']:raise ValueError('Incomplete sampling panel')
        sources.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        refs={r['validation_index']:r for r in report['references']}
        for arm in report['arms']:
            profiles={}
            for sample in arm['samples']:
                parent=sample['validation_index'];symbols=refs[parent]['symbols']
                z=[periodic.GetAtomicNumber(s) for s in symbols]
                r=[periodic.GetRcovalent(s) for s in symbols]
                summary=contact_summary(sample['positions'],r)
                reference=contact_summary(refs[parent]['positions'],r)
                rows.append({'source':str(path),'arm':arm['name'],'parent':parent,
                    'sample':sample['sample_id'],**summary,'reference':reference})
                profiles.setdefault(parent,[]).append(distance_profile(sample['positions'],z))
            for parent,items in profiles.items():
                pairwise=[profile_rms(left,right) for left,right in itertools.combinations(items,2)]
                diversity.append({'source':str(path),'arm':arm['name'],'parent':parent,
                    'sample_count':len(items),'pair_count':len(pairwise),
                    'median_pair_profile_rms_A':st.median(pairwise) if pairwise else None})
    summaries=[]
    for source,arm in sorted({(r['source'],r['arm']) for r in rows}):
        selected=[r for r in rows if r['source']==source and r['arm']==arm]
        profiles=[r['median_pair_profile_rms_A'] for r in diversity if r['source']==source and r['arm']==arm]
        summaries.append({'source':source,'arm':arm,'attempted':len(selected),
            'geometries_with_overlap':sum(r['overlap_pairs']>0 for r in selected),
            'geometries_with_multiple_contact_components':sum(r['contact_components']>1 for r in selected),
            'geometries_with_more_components_than_reference':sum(r['contact_components']>r['reference']['contact_components'] for r in selected),
            'median_parent_pair_profile_rms_A':st.median(v for v in profiles if v is not None) if any(v is not None for v in profiles) else None})
    output={'sources':sources,'rows':rows,'per_parent_diversity':diversity,'summaries':summaries,
        'contact_factor':1.25,'overlap_factor':.6,
        'radii_source':'RDKit GetRcovalent',
        'limitations':['Contact graphs are distance heuristics, not inferred chemical bonds or validity',
            'Element-pair distance profiles are incomplete invariants, not aligned molecular RMSD',
            'Variation among four samples does not establish ensemble coverage or correct mode weights',
            'All supplied generated samples are retained; these diagnostics do not select outputs']}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
