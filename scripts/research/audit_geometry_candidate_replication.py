"""Combine the two fixed-benchmark fits and three subsequent geometry replications."""
import argparse,gzip,json
from pathlib import Path
import numpy as np
from cfm_mol.replication_statistics import paired_intervals
from scripts.research.run_matched_generators import write
from scripts.research.train_electronic_fm import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=a.project.resolve()
    first_protocol=root/'research/evidence/geometry_primary_confirmation_v1.json';last_protocol=root/'research/evidence/geometry_candidate_replication_v1.json'
    first=json.loads(first_protocol.read_text());last=json.loads(last_protocol.read_text())
    original_file=root/'research/evidence/seed_replication_audit_v2.json';original=json.loads(original_file.read_text());original_arrays=dict(np.load(original_file.with_suffix('.npz')));mi=original['methods'].index('fm_physical')
    records=[json.loads(line) for line in gzip.decompress((root/'research/evidence/round2_chemical_geometry_records_v1.jsonl.gz').read_bytes()).decode().splitlines()]
    shape=(5,2,64,16);arrays={name:np.zeros(shape,dtype=bool) for name in ['graph','geometry','geometry_force','graph_force','success','unique_geometry']};arrays['force']=np.full(shape,np.nan)
    arrays['graph'][:,0]=original_arrays['graph'][:,mi];arrays['success'][:,0]=original_arrays['success'][:,mi];arrays['force'][:,0]=original_arrays['force'][:,mi]
    for row in records:
        if row['method']=='fm_physical':arrays['geometry'][row['fit'],0,row['condition'],row['sample']]=row['closed_shell_geometry_pass']
    provenance={};paths=[]
    for fit in range(5):
        folder=root/f'runs/geometry_primary_confirmation_v1/s{fit}/full_geometry_physics/evaluation' if fit<2 else root/f'runs/geometry_candidate_replication_v1/s{fit}/evaluation'
        donefile=folder/'complete.json'
        if not donefile.exists():raise RuntimeError(f'Fit{fit} incomplete: {donefile}')
        done=json.loads(donefile.read_text());assert done['complete'] and done['fit']==fit and done['attempted']==1024
        assert done['protocol_sha256']==sha(first_protocol if fit<2 else last_protocol)
        for name,key in [('geometry.json','geometry_sha256'),('xtb/results.json','physical_sha256'),('generation/generation.json','generation_sha256')]:assert sha(folder/name)==done[key]
        generation=json.loads((folder/'generation/generation.json').read_text())
        for row in generation['rows']:assert sha(folder/'generation'/row['file'])==row['sha256']
        physical=json.loads((folder/'xtb/results.json').read_text())['rows'];geom=json.loads((folder/'geometry.json').read_text())['rows'];lookup={(row['condition_index'],row['sample_index']):row for row in physical};assert len(lookup)==1024
        seen={ci:set() for ci in range(64)}
        for row in sorted(geom,key=lambda row:(row['condition'],row['sample'])):
            ci,j=row['condition'],row['sample'];score=lookup[ci,j];assert row['graph']==score['graph'];slot=(fit,1,ci,j)
            arrays['graph'][slot]=row['graph'];arrays['geometry'][slot]=row['closed_shell_geometry_pass'];arrays['success'][slot]=score['success'];arrays['force'][slot]=score['rms_force'] if score['success'] else np.nan
            if row['closed_shell_geometry_pass']:
                arrays['unique_geometry'][slot]=row['smiles'] not in seen[ci];seen[ci].add(row['smiles'])
        baseline_rows=sorted((row for row in records if row['method']=='fm_physical' and row['fit']==fit),key=lambda row:(row['condition'],row['sample']));seen={ci:set() for ci in range(64)}
        for row in baseline_rows:
            if row['closed_shell_geometry_pass']:
                ci,j=row['condition'],row['sample'];arrays['unique_geometry'][fit,0,ci,j]=row['smiles'] not in seen[ci];seen[ci].add(row['smiles'])
        provenance[str(donefile.relative_to(root))]=sha(donefile);paths.append(str(folder.relative_to(root)))
    low=arrays['success']&(arrays['force']<=5);arrays['geometry_force']=arrays['geometry']&low;arrays['graph_force']=arrays['graph']&low
    strata=np.array([entry['condition']['n_atoms']>28 for entry in first['conditions']]);metrics=['graph','geometry','geometry_force','graph_force','unique_geometry']
    summary={};contrasts={}
    for index,name in enumerate(['original','geometric_candidate']):
        summary[name]=dict(attempted=5120,**{metric:dict(count=int(arrays[metric][:,index].sum()),rate=float(arrays[metric][:,index].mean()),by_fit=arrays[metric][:,index].mean((1,2)).tolist()) for metric in metrics})
    for label,selected in [('all_five',slice(None)),('three_additional',slice(2,5))]:
        contrasts[label]={metric:paired_intervals((arrays[metric][selected,1].astype(float)-arrays[metric][selected,0]).mean(-1),strata=strata) for metric in metrics}
    a.out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.out.with_suffix('.npz'),**arrays)
    result=dict(complete=True,summary=summary,contrasts=contrasts,methods=['original','geometric_candidate'],size_counts={'17-28':int((~strata).sum()),'29-40':int(strata.sum())},provenance=provenance,evaluation_paths=paths,source_baseline_audit_sha256=sha(original_file),arrays_sha256=sha(a.out.with_suffix('.npz')),additional_replication_gate=bool(min(contrasts['three_additional']['geometry_force']['by_fit'])>0 and contrasts['three_additional']['geometry_force']['crossed_fit_composition_ci95'][0]>0),scope='All five trained candidates retained. Fits0/1 informed selection on a separate development panel; fits2/3/4 repeat the frozen recipe. Existing64-composition benchmark reused. No output optimization, energy filtering or new force-teacher queries.')
    write(a.out,result);print(json.dumps(dict(summary=summary,additional_replication_gate=result['additional_replication_gate'],additional_joint=contrasts['three_additional']['geometry_force']),indent=2),flush=True)


if __name__=='__main__':main()
