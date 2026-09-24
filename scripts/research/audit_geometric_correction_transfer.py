"""Summarize frozen two-field transfers and the corresponding source ablation."""
import argparse,gzip,json
from pathlib import Path
import numpy as np
from cfm_mol.replication_statistics import paired_intervals
from scripts.research.run_matched_generators import write
from scripts.research.train_electronic_fm import sha


def main():
    p=argparse.ArgumentParser()
    for key in ['project','protocol','run','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();root=a.project.resolve();a.run=a.run.resolve();spec=json.loads(a.protocol.read_text());ph=sha(a.protocol)
    complete=json.loads((a.run/'complete.json').read_text());assert complete['complete'] and complete['protocol_sha256']==ph
    other=dict(np.load(root/'research/evidence/other_baseline_transfer_audit_v1.npz'))
    geometry=dict(np.load(root/'runs/baseline_chemical_geometry_v1/results/audit.npz'))
    seed_audit=json.loads((root/'research/evidence/seed_replication_audit_v2.json').read_text());seed_arrays=dict(np.load(root/'research/evidence/seed_replication_audit_v2.npz'))
    records=[json.loads(line) for line in gzip.decompress((root/'research/evidence/round2_chemical_geometry_records_v1.jsonl.gz').read_bytes()).decode().splitlines()]
    strata=np.array([entry['condition']['n_atoms']>28 for entry in spec['conditions']]);all_arrays={};summary={};contrasts={};provenance={}
    metrics=['graph','geometry','graph_force','geometry_force']
    for target in spec['targets']:
        arrays={key:np.zeros((2,2,64,16),dtype=bool) for key in ['graph','geometry','success']};arrays['force']=np.full((2,2,64,16),np.nan)
        if target=='gaga':
            index=seed_audit['methods'].index('gaga_parent')
            for key in ['graph','success','force']:arrays[key][:,0]=seed_arrays[key][:2,index]
            for row in records:
                if row['method']=='gaga_parent' and row['fit']<2:arrays['geometry'][row['fit'],0,row['condition'],row['sample']]=row['closed_shell_geometry_pass']
        else:
            for key in ['graph','success','force']:arrays[key][:,0]=other[target+'_'+key][:,0]
            arrays['geometry'][:,0]=geometry[target+'_geometry'][:,0]
        for fit in [0,1]:
            folder=a.run/target/f's{fit}'/'evaluation';donefile=folder/'complete.json';done=json.loads(donefile.read_text())
            assert done['complete'] and done['protocol_sha256']==ph and done['attempted']==1024
            assert done['geometry_head_state_sha256']==spec['geometry_heads'][fit]['ema_state_sha256']
            for file,key in [('generation/generation.json','generation_sha256'),('geometry.json','geometry_sha256'),('xtb/results.json','physical_sha256')]:assert sha(folder/file)==done[key]
            gen=json.loads((folder/'generation/generation.json').read_text())
            for row in gen['rows']:assert sha(folder/'generation'/row['file'])==row['sha256']
            g=json.loads((folder/'geometry.json').read_text())['rows'];scores=json.loads((folder/'xtb/results.json').read_text())['rows'];lookup={(row['condition_index'],row['sample_index']):row for row in scores};assert len(lookup)==len(g)==1024
            for row in g:
                ci,j=row['condition'],row['sample'];score=lookup[ci,j];slot=(fit,1,ci,j);assert row['graph']==score['graph']
                arrays['graph'][slot]=row['graph'];arrays['geometry'][slot]=row['closed_shell_geometry_pass'];arrays['success'][slot]=score['success'];arrays['force'][slot]=score['rms_force'] if score['success'] else np.nan
            provenance[str(donefile.relative_to(root))]=sha(donefile)
        low=arrays['success']&(arrays['force']<=5);arrays['graph_force']=arrays['graph']&low;arrays['geometry_force']=arrays['geometry']&low
        for key,value in arrays.items():all_arrays[target+'_'+key]=value
        summary[target]={name:dict(attempted=2048,**{metric:dict(count=int(arrays[metric][:,i].sum()),rate=float(arrays[metric][:,i].mean()),by_fit=arrays[metric][:,i].mean((1,2)).tolist()) for metric in metrics}) for i,name in enumerate(['parent','geometric_physical'])}
        contrasts[target]={metric:paired_intervals((arrays[metric][:,1].astype(float)-arrays[metric][:,0]).mean(-1),strata=strata,seed=78601,repetitions=10000) for metric in metrics}
    source_contrasts={}
    for metric in metrics:
        gauss=all_arrays['gaussian_fm_'+metric];harmonic=all_arrays['harmonic_fm_'+metric]
        source_contrasts[metric]={name:paired_intervals(delta.mean(-1),strata=strata,seed=78602,repetitions=10000) for name,delta in [
            ('source_only',harmonic[:,0].astype(float)-gauss[:,0]),('source_with_correction',harmonic[:,1].astype(float)-gauss[:,1]),
            ('correction_with_gaussian',gauss[:,1].astype(float)-gauss[:,0]),('correction_with_harmonic',harmonic[:,1].astype(float)-harmonic[:,0]) ]}
    a.out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.out.with_suffix('.npz'),**all_arrays)
    result=dict(complete=True,protocol_sha256=ph,summary=summary,contrasts=contrasts,source_contrasts=source_contrasts,
        provenance=provenance,arrays_sha256=sha(a.out.with_suffix('.npz')),new_generation_outputs=8192,new_gfn2_attempts=8192,
        new_optimizer_steps=0,new_esen_queries=0,scope=spec['scope'])
    write(a.out,result);print(json.dumps(dict(summary=summary,source_contrasts=source_contrasts),indent=2),flush=True)


if __name__=='__main__':main()
