"""Screen a fixed training-only reference subset for the recovery-field teacher."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import warnings
import torch
from cfm_mol.chemical_geometry_review import molecule_from_coordinates,geometry_diagnostics,contact_diagnostics
from scripts.research.run_matched_generators import write
from scripts.research.train_electronic_fm import sha


def check_reference(item):
    index,positions,condition=item
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',FutureWarning)
        row=dict(index=index,composition_hex=condition['composition_hex'],**contact_diagnostics(positions,condition['numbers']))
        try:
            row.update(geometry_diagnostics(molecule_from_coordinates(positions,condition['numbers'],condition['charge'])))
            row['eligible']=bool(row['closed_shell_geometry_pass'] and row['contact_components']==1 and not row['severe_overlap'])
        except Exception as exc:row.update(eligible=False,error=type(exc).__name__+': '+str(exc))
        return row


def main():
    p=argparse.ArgumentParser()
    for key in ['project','protocol','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();root=a.project.resolve();spec=json.loads(a.protocol.read_text());ph=sha(a.protocol)
    a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'complete.json').exists():
        old=json.loads((a.out/'complete.json').read_text());assert old['complete'] and old['protocol_sha256']==ph;return
    torch.set_num_threads(1);datafile=root/spec['data'];assert sha(datafile)==spec['data_sha256']
    data=torch.load(datafile,map_location='cpu',weights_only=False)['training']
    ranked=sorted(range(len(data)),key=lambda i:hashlib.sha256(f'{spec["reference_rank_seed"]}:{i}'.encode()).hexdigest())[:spec['reference_candidates']]
    items=[(i,data[i]['positions'].tolist(),data[i]['condition']) for i in ranked]
    with ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(check_reference,items,chunksize=16))
    indices=[r['index'] for r in rows if r['eligible']]
    write(a.out/'reference_checks.json',dict(protocol_sha256=ph,rows=rows))
    assert len(indices)>=spec['minimum_qualified_references'],len(indices)
    write(a.out/'complete.json',dict(complete=True,protocol_sha256=ph,examined=len(rows),qualified=len(indices),indices=indices,
        checks_sha256=sha(a.out/'reference_checks.json'),data_sha256=spec['data_sha256'],new_esen_queries=0,new_gfn2_attempts=0,
        scope='Training reference selection uses inferred-graph geometry checks. The network receives only coordinates, elements and progress, not bond labels. All selection outcomes are retained.'))
    print(json.dumps(dict(phase='reference_screen',examined=len(rows),qualified=len(indices))),flush=True)


if __name__=='__main__':main()
