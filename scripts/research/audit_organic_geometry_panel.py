"""Audit every raw organic-panel output, including inferred charge patterns."""
import argparse,gzip,json
from pathlib import Path
import numpy as np
from rdkit import Chem
from cfm_mol.replication_statistics import paired_intervals
from scripts.research.run_matched_generators import write
from scripts.research.train_electronic_fm import sha


def main():
    p=argparse.ArgumentParser()
    for key in ['project','protocol','run','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();root=a.project.resolve();run=a.run.resolve();spec=json.loads(a.protocol.read_text());ph=sha(a.protocol)
    done=json.loads((run/'complete.json').read_text());assert done['complete'] and done['protocol_sha256']==ph
    arrays={};summary={};provenance={};records=[];candidates=[];n=len(spec['conditions'])
    metrics=['graph','geometry','graph_force','geometry_force','uncharged_geometry_force','no_charged_carbon_geometry_force']
    for target in spec['targets']:
        x={k:np.zeros((2,n,16),dtype=bool) for k in metrics};x['force']=np.full((2,n,16),np.nan)
        for fit in [0,1]:
            folder=run/target/f's{fit}'/'evaluation';receipt=json.loads((folder/'complete.json').read_text())
            assert receipt['complete'] and receipt['protocol_sha256']==ph and receipt['attempted']==n*16
            for file,key in [('generation/generation.json','generation_sha256'),('geometry.json','geometry_sha256'),('xtb/results.json','physical_sha256')]:assert sha(folder/file)==receipt[key]
            gen=json.loads((folder/'generation/generation.json').read_text());files={}
            for row in gen['rows']:
                path=folder/'generation'/row['file'];assert sha(path)==row['sha256'];files[row['condition_index']]=str(path.relative_to(root))
            scores=json.loads((folder/'xtb/results.json').read_text())['rows'];lookup={(r['condition_index'],r['sample_index']):r for r in scores}
            geometry=json.loads((folder/'geometry.json').read_text())['rows'];assert len(lookup)==len(geometry)==n*16
            for row in geometry:
                ci,j=row['condition'],row['sample'];score=lookup[ci,j];assert row['graph']==score['graph'];slot=(fit,ci,j)
                x['graph'][slot]=row['graph'];x['geometry'][slot]=row['closed_shell_geometry_pass']
                low=score['success'] and score['rms_force']<=5
                if score['success']:x['force'][slot]=score['rms_force']
                x['graph_force'][slot]=row['graph'] and low;x['geometry_force'][slot]=row['closed_shell_geometry_pass'] and low
                cc=None
                if row['graph']:
                    mol=Chem.MolFromSmiles(row['smiles']);assert mol is not None
                    cc=sum(atom.GetAtomicNum()==6 and atom.GetFormalCharge()!=0 for atom in mol.GetAtoms())
                uncharged=row.get('charged_atoms')==0
                x['uncharged_geometry_force'][slot]=x['geometry_force'][slot] and uncharged
                x['no_charged_carbon_geometry_force'][slot]=x['geometry_force'][slot] and cc==0
                rec=dict(row,target=target,fit=fit,force=float(x['force'][slot]) if score['success'] else None,
                         charged_carbon_atoms=cc,file=files[ci]);records.append(rec)
                if target=='full_geometry_physics' and x['uncharged_geometry_force'][slot]:candidates.append(rec)
            provenance[str((folder/'complete.json').relative_to(root))]=sha(folder/'complete.json')
        summary[target]={metric:dict(count=int(x[metric].sum()),attempted=int(x[metric].size),rate=float(x[metric].mean()),by_fit=x[metric].mean((1,2)).tolist()) for metric in metrics}
        arrays.update({target+'_'+k:v for k,v in x.items()})
    contrasts={target:{metric:paired_intervals((arrays['full_geometry_physics_'+metric].astype(float)-arrays[target+'_'+metric]).mean(-1),seed=79361,repetitions=10000) for metric in metrics} for target in spec['targets'] if target!='full_geometry_physics'}
    a.out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.out.with_suffix('.npz'),**arrays)
    recordfile=a.out.with_suffix('.jsonl.gz');recordfile.write_bytes(gzip.compress(('\n'.join(json.dumps(r) for r in records)+'\n').encode(),mtime=0))
    write(a.out,dict(complete=True,protocol_sha256=ph,summary=summary,contrasts=contrasts,provenance=provenance,
        arrays_sha256=sha(a.out.with_suffix('.npz')),records_sha256=sha(recordfile),new_generation_outputs=1920,new_gfn2_attempts=1920,
        new_optimizer_steps=0,new_esen_queries=0,scope=spec['scope'],charge_diagnostic='Descriptive filters, not a complete chemistry-validity definition. Charged functional groups can be chemically legitimate.'))
    write(run/'uncharged_illustration_candidates.json',dict(selection='All geometry-and-force passing full-model outputs with zero assigned atom charges; illustration only.',rows=candidates))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
