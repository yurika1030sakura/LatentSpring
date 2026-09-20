"""Audit original-versus-both comparisons for the two structured diffusion targets."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from cfm_mol.replication_statistics import paired_intervals
from scripts.research.train_electronic_fm import sha
from scripts.research.run_matched_generators import write


def main():
    q=argparse.ArgumentParser(description=__doc__)
    for k in ['project','protocol','run','out']:q.add_argument('--'+k,type=Path,required=True)
    a=q.parse_args();p=json.loads(a.protocol.read_text());ph=sha(a.protocol);summary={};contrasts={};provenance={};released={}
    strata=[c['n_atoms']>28 for c in p['test_rows']]
    for target in p['targets']:
        info=p['baseline_audits'][target];bf=a.project/info['path'];assert sha(bf)==info['sha256']
        baseline_audit=json.loads(bf.read_text());assert baseline_audit['complete']
        assert sha(a.project/info['arrays'])==info['arrays_sha256'];allbase=dict(np.load(a.project/info['arrays']))
        if target=='edm':baseline={k:allbase['edm_'+k][:,0] for k in ['graph','success','force','energy']}
        else:
            mi=baseline_audit['methods'].index('gaga_parent');baseline={k:allbase[k][:2,mi] for k in ['graph','success','force','energy']}
        new=[]
        for si in p['fits']:
            folder=a.run/target/f's{si}';done=json.loads((folder/'complete.json').read_text())
            assert done['complete'] and done['protocol_sha256']==ph and done['fit']==si and done['target']==target
            for filename,key in [('audit.npz','arrays_sha256'),('generation/generation.json','generation_sha256'),('xtb/results.json','physical_sha256'),('training/complete.json','training_sha256')]:assert sha(folder/filename)==done[key]
            assert done['head_state_sha256']==p['heads'][si]['head_state_sha256']
            training=json.loads((folder/'training/complete.json').read_text());ck=folder/'training/last.ckpt'
            assert training['steps']==30000 and sha(ck)==training['checkpoint_sha256']
            saved=torch.load(ck,map_location='cpu',weights_only=False)
            assert saved['global_step']==30000 and saved['protocol_sha256']==ph
            assert all(int(v['step'])==30000 for v in saved['optimizer_state_dict']['state'].values())
            assert training['ema_state_sha256']==done['model_state_sha256']
            original=p['parents'][si][target];assert sha(folder/'training/batch_indices.npy')==original['baseline_batch_indices_sha256']
            log=[json.loads(line) for f in sorted((folder/'training').glob('metrics_attempt*.jsonl')) for line in f.read_text().splitlines()]
            assert log and log[-1]['step']==30000 and all(np.isfinite([r['loss'],r['gradient_norm']]).all() for r in log)
            new.append(dict(np.load(folder/'audit.npz')));provenance[str((folder/'complete.json').relative_to(a.project))]=sha(folder/'complete.json')
        new={k:np.stack([f[k] for f in new]) for k in new[0]}
        bjoint=baseline['graph']&baseline['success']&(baseline['force']<=5);njoint=new['graph']&new['success']&(new['force']<=5)
        summary[target]={label:dict(attempted=2048,graph=float(v['graph'].mean()),joint=float(j.mean()),
            graph_by_fit=v['graph'].mean((1,2)).tolist(),joint_by_fit=j.mean((1,2)).tolist(),failures=int((~v['success']).sum()))
            for label,v,j in [('baseline',baseline,bjoint),('both',new,njoint)]}
        contrasts[target]={}
        for label,left,right in [('graph',new['graph'],baseline['graph']),('joint',njoint,bjoint)]:
            delta=(left.astype(float)-right).mean(-1);s=paired_intervals(delta,strata=strata)
            rng=np.random.default_rng(81301);ci=rng.integers(64,size=(20000,64));fi=rng.integers(2,size=(20000,2))
            draws=delta[fi[:,:,None],ci[:,None,:]].mean((1,2))
            s['bonferroni_two_targets_crossed_ci']=np.quantile(draws,[.0125,.9875]).tolist();contrasts[target][label]=s
        for name,v in [('baseline',baseline),('both',new)]:
            for k,x in v.items():released[target+'_'+name+'_'+k]=x
    np.savez_compressed(a.out.with_suffix('.npz'),**released)
    result=dict(complete=True,protocol_sha256=ph,summary=summary,contrasts=contrasts,provenance=provenance,
        arrays_sha256=sha(a.out.with_suffix('.npz')),new_parent_updates=120000,new_head_updates=0,new_training_example_forwards=3840000,
        diagnostic_validation_forwards=192,new_evaluation_outputs=4096,new_gfn2_attempts=4096,new_esen_queries=0,verification_replays=64,
        transfer_supported={t:bool(min(contrasts[t]['joint']['by_fit'])>0 and contrasts[t]['joint']['bonferroni_two_targets_crossed_ci'][0]>0) for t in p['targets']},scope=p['scope'])
    write(a.out,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
