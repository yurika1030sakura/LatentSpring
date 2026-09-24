"""Apply the same geometric screen to all archived additional-baseline outputs."""
import argparse,gzip,json,warnings,hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch
from cfm_mol.chemical_geometry_review import molecule_from_coordinates,geometry_diagnostics,contact_diagnostics
from scripts.research.run_matched_generators import write
from scripts.research.train_electronic_fm import sha


def assess_file(task):
    path,entry,target,fit,arm,forces,success=task
    assert sha(Path(path))==entry['sha256']
    saved=torch.load(path,map_location='cpu',weights_only=False);c=saved['condition'];rows=[]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',FutureWarning)
        for j,(position,original) in enumerate(zip(saved['positions'].numpy(),entry['records'])):
            row=dict(target=target,fit=fit,arm=arm,condition=entry['condition_index'],sample=j,
                graph=bool(original['graph_supported']),smiles=original.get('smiles'),success=bool(success[j]),
                force=float(forces[j]) if success[j] else None,source_sha256=entry['sha256'],
                **contact_diagnostics(position,c['numbers']))
            if row['graph']:
                try:
                    row.update(geometry_diagnostics(molecule_from_coordinates(position,c['numbers'],c['charge'])))
                    assert row['smiles']==original['smiles']
                except Exception as exc:row.update(closed_shell_geometry_pass=False,error=type(exc).__name__+': '+str(exc))
            else:row['closed_shell_geometry_pass']=False
            row['geometry_force']=bool(row['closed_shell_geometry_pass'] and row['success'] and row['force']<=5)
            rows.append(row)
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=a.project.resolve();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    audit_file=root/'research/evidence/other_baseline_transfer_audit_v1.json';audit=json.loads(audit_file.read_text());assert audit['complete']
    arrays_file=audit_file.with_suffix('.npz');assert sha(arrays_file)==audit['arrays_sha256'];source=dict(np.load(arrays_file));tasks=[];provenance={}
    targets=['edm','gaussian_fm','harmonic_fm']
    for target in targets:
        for fit in [0,1]:
            folder=root/f'runs/other_baseline_transfer_v1/{target}/s{fit}'
            complete=folder/'complete.json';assert sha(complete)==audit['provenance'][str(complete.relative_to(root))]
            d=json.loads(complete.read_text());report=json.loads((folder/'generation/generation.json').read_text())
            assert sha(folder/'generation/generation.json')==d['generation_sha256'];provenance[str(complete.relative_to(root))]=sha(complete)
            for row in report['rows']:
                arm=['parent','transferred'].index(row['method']);ci=row['condition_index']
                expected=source[target+'_graph'][fit,arm,ci]
                np.testing.assert_array_equal(expected,[v['graph_supported'] for v in row['records']])
                tasks.append((str(folder/'generation'/row['file']),row,target,fit,arm,
                    source[target+'_force'][fit,arm,ci],source[target+'_success'][fit,arm,ci]))
    records=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for rows in pool.map(assess_file,tasks,chunksize=4):records.extend(rows)
    assert len(records)==12288
    encoded=''.join(json.dumps(row,allow_nan=False)+'\n' for row in records).encode();(a.out/'records.jsonl.gz').write_bytes(gzip.compress(encoded,mtime=0))
    summary={};out_arrays={}
    for target in targets:
        summary[target]={};geometry=np.zeros((2,2,64,16),dtype=bool)
        for row in records:
            if row['target']==target:geometry[row['fit'],row['arm'],row['condition'],row['sample']]=row['closed_shell_geometry_pass']
        out_arrays[target+'_geometry']=geometry
        out_arrays[target+'_geometry_force']=geometry&source[target+'_success']&(source[target+'_force']<=5)
        for arm,name in enumerate(['parent','transferred']):
            rows=[v for v in records if v['target']==target and v['arm']==arm]
            summary[target][name]=dict(attempted=len(rows),graph=sum(v['graph'] for v in rows),
                geometry=sum(v['closed_shell_geometry_pass'] for v in rows),geometry_force=sum(v['geometry_force'] for v in rows),
                heavy_disconnected=sum(v['heavy_components']>1 for v in rows),
                failure_counts={k:sum(v['graph'] and not v.get(k,False) for v in rows) for k in ['planar_groups_pass','bond_lengths_within_bounds','bond_angles_within_bounds','no_internal_clash']})
    np.savez_compressed(a.out/'audit.npz',**out_arrays)
    report=dict(complete=True,summary=summary,provenance=provenance,source_audit_sha256=sha(audit_file),source_arrays_sha256=sha(arrays_file),
        arrays_sha256=sha(a.out/'audit.npz'),records_sha256=sha(a.out/'records.jsonl.gz'),archived_outputs_reassessed=len(records),new_generation_outputs=0,new_gfn2_attempts=0,new_esen_queries=0,
        scope='The same declared PoseBusters geometry subset and zero-radical requirement on every archived attempt. No coordinate changes, new scores, or outcome selection.')
    write(a.out/'audit.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
