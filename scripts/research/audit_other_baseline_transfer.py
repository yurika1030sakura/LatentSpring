"""Summarize every frozen additional-target transfer, without target selection."""
import argparse
import json
from pathlib import Path
import numpy as np
from cfm_mol.replication_statistics import paired_intervals
from scripts.research.train_electronic_fm import sha
from scripts.research.run_matched_generators import write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','protocol','run','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();spec=json.loads(a.protocol.read_text());ph=sha(a.protocol)
    summary={};contrasts={};provenance={};all_arrays={}
    strata=np.array([c['n_atoms']>28 for c in spec['test_rows']])
    for target in spec['targets']:
        fits=[]
        for si in spec['fits']:
            folder=a.run/target/f's{si}';file=folder/'complete.json';d=json.loads(file.read_text())
            assert d['complete'] and d['protocol_sha256']==ph and d['fit']==si and d['target']==target
            assert sha(folder/'audit.npz')==d['arrays_sha256']
            assert sha(folder/'generation/generation.json')==d['generation_sha256']
            assert sha(folder/'xtb/results.json')==d['physical_sha256']
            assert d['head_state_sha256']==spec['heads'][si]['head_state_sha256']
            assert d['new_parent_trajectories']==d['new_gfn2_attempts']==2048
            fits.append(dict(np.load(folder/'audit.npz')));provenance[str(file.relative_to(a.project))]=sha(file)
        arrays={k:np.stack([r[k] for r in fits]) for k in fits[0]}
        joint=arrays['graph']&arrays['success']&(arrays['force']<=5)
        for k,x in arrays.items():all_arrays[target+'_'+k]=x
        summary[target]={label:dict(attempted=2048,graph=float(arrays['graph'][:,ai].mean()),joint=float(joint[:,ai].mean()),
            graph_by_fit=arrays['graph'][:,ai].mean((1,2)).tolist(),joint_by_fit=joint[:,ai].mean((1,2)).tolist(),
            numerical_failures=int((~arrays['success'][:,ai]).sum())) for ai,label in enumerate(['parent','transferred'])}
        contrasts[target]={}
        for label,values in [('graph',arrays['graph']),('joint',joint)]:
            delta=(values[:,1].astype(float)-values[:,0]).mean(-1)
            stats=paired_intervals(delta,strata=strata)
            rng=np.random.default_rng(79201);ci=rng.integers(64,size=(20000,64));fi=rng.integers(2,size=(20000,2))
            crossing=delta[fi[:,:,None],ci[:,None,:]].mean((1,2));composition=delta.mean(0)[ci].mean(1)
            alpha=.05/len(spec['targets'])
            stats['bonferroni_three_targets_composition_ci']=np.quantile(composition,[alpha/2,1-alpha/2]).tolist()
            stats['bonferroni_three_targets_crossed_ci']=np.quantile(crossing,[alpha/2,1-alpha/2]).tolist()
            contrasts[target][label]=stats
    out=dict(complete=True,protocol_sha256=ph,summary=summary,contrasts=contrasts,provenance=provenance,
        new_parent_trajectories=12288,new_gfn2_attempts=12288,new_esen_queries=0,new_optimizer_steps=0,verification_replays=48,
        scope=spec['scope'],transfer_supported={t:bool(min(contrasts[t]['joint']['by_fit'])>0 and
            contrasts[t]['joint']['bonferroni_three_targets_crossed_ci'][0]>0) for t in spec['targets']})
    np.savez_compressed(a.out.with_suffix('.npz'),**all_arrays);out['arrays_sha256']=sha(a.out.with_suffix('.npz'))
    write(a.out,out);print(json.dumps(out,indent=2))


if __name__=='__main__':main()
