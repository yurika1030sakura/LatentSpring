"""Separate composition uncertainty from variation across independent fitted models."""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.research.train_electronic_fm import sha
from scripts.research.run_matched_generators import write


def paired_intervals(delta,*,seed=65101,repetitions=20000,strata=None):
    """Rows are fits and columns are compositions; preserve both pairing axes."""
    x=np.asarray(delta,dtype=float)
    if x.ndim!=2 or not x.size or not np.isfinite(x).all():raise ValueError('Finite fit-by-composition differences required')
    rng=np.random.default_rng(seed);nf,nc=x.shape;comp=[];cross=[]
    for start in range(0,repetitions,2000):
        n=min(2000,repetitions-start);ci=rng.integers(nc,size=(n,nc));fi=rng.integers(nf,size=(n,nf))
        comp.extend(x.mean(0)[ci].mean(-1));cross.extend(x[fi[:,:,None],ci[:,None,:]].mean((1,2)))
    result=dict(mean=float(x.mean()),by_fit=x.mean(1).tolist(),composition_ci95=np.quantile(comp,[.025,.975]).tolist(),
        crossed_fit_composition_ci95=np.quantile(cross,[.025,.975]).tolist(),fits=nf,compositions=nc)
    if strata is not None:
        strata=np.asarray(strata);assert strata.shape==(nc,);groups=[np.where(strata==v)[0] for v in np.unique(strata)];result['by_size_bin']=[dict(compositions=len(ids),mean=float(x[:,ids].mean()),by_fit=x[:,ids].mean(1).tolist()) for ids in groups]
        values=[]
        for start in range(0,repetitions,2000):
            n=min(2000,repetitions-start);fi=rng.integers(nf,size=(n,nf));means=[]
            for ids in groups:
                ci=ids[rng.integers(len(ids),size=(n,len(ids)))];means.append(x[fi[:,:,None],ci[:,None,:]].mean((1,2)))
            values.extend(np.mean(means,axis=0))
        result['equal_stratum_mean']=float(np.mean([x[:,ids].mean() for ids in groups]));result['equal_stratum_crossed_ci95']=np.quantile(values,[.025,.975]).tolist()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','protocol','audits','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.project=a.project.resolve();a.audits=a.audits.resolve();assert not a.out.exists();campaign=json.loads(a.protocol.read_text());ph=sha(a.protocol);perfit=[];provenance={};audits=[]
    panel=json.loads((a.project/campaign['panel']).read_text());strata=np.array([0 if r['n_atoms']<=28 else 1 for r in panel['rows']]);assert (strata==0).sum()==20 and (strata==1).sum()==44
    for si in range(5):
        f=a.audits/f's{si}.json';d=json.loads(f.read_text());assert d['complete'] and d['fit']==si and d['campaign_sha256']==ph
        assert sha(f.with_suffix('.npz'))==d['arrays_sha256'];perfit.append(dict(np.load(f.with_suffix('.npz'))));audits.append(d);provenance[str(f.relative_to(a.project))]=sha(f)
    methods=audits[0]['methods'];assert all(d['methods']==methods for d in audits);arrays={k:np.stack([d[k] for d in perfit]) for k in perfit[0]}
    graph=arrays['graph'];success=arrays['success'];force=arrays['force'];joint=graph&success&(force<=5);summary={};thresholds=[.5,1.,2.,3.,5.,10.,20.]
    for mi,m in enumerate(methods):
        summary[m]=dict(attempted=int(graph[:,mi].size),graph_rate=float(graph[:,mi].mean()),joint_rate=float(joint[:,mi].mean()),
            graph_by_fit=graph[:,mi].mean((1,2)).tolist(),joint_by_fit=joint[:,mi].mean((1,2)).tolist(),
            gfn2_failures_by_fit=(~success[:,mi]).sum((1,2)).tolist(),
            force_threshold_curve=[dict(threshold=t,pooled=float((graph[:,mi]&success[:,mi]&(force[:,mi]<=t)).mean()),by_fit=(graph[:,mi]&success[:,mi]&(force[:,mi]<=t)).mean((1,2)).tolist()) for t in thresholds])
    comparisons=[]
    for name in ['fm','gaga']:
        comparisons.extend([(name+'_physical_minus_parent',name+'_physical',name+'_parent'),(name+'_hydrogen_minus_physical',name+'_hydrogen',name+'_physical'),(name+'_hydrogen_minus_radial',name+'_hydrogen',name+'_radial')])
    comparisons.extend([('fm_minus_gaga_physical','fm_physical','gaga_physical'),('fm_minus_gaga_hydrogen','fm_hydrogen','gaga_hydrogen')]);contrasts={}
    for name,left,right in comparisons:
        li,ri=methods.index(left),methods.index(right);result={}
        for tag,ids in [('all_five',[0,1,2,3,4]),('new_three',[2,3,4])]:
            subset={}
            for metric,value in [('graph',graph),('joint',joint)]:subset[metric]=paired_intervals((value[ids,li].astype(float)-value[ids,ri]).mean(-1),strata=strata)
            same=arrays['coordinate_hash'][ids,li]==arrays['coordinate_hash'][ids,ri];ok=success[ids,li]&success[ids,ri];missing=(~same)&(~ok)
            energy=dict(complete=not bool(missing.any()),missing_changed_pairs=int(missing.sum()),identity_zero_pairs=int(same.sum()),failed_identity_cancellations=int((same&~ok).sum()))
            delta=np.where(same,0.,arrays['energy'][ids,li]-arrays['energy'][ids,ri])
            if not missing.any():energy.update(paired_intervals(delta.mean(-1),strata=strata))
            subset['energy']=energy;result[tag]=subset
        contrasts[name]=result
    primary=contrasts['fm_physical_minus_parent']['new_three']['joint'];he=contrasts['fm_hydrogen_minus_radial']['new_three'];energy=he['energy']
    gate=primary['composition_ci95'][0]>0 and min(primary['by_fit'])>0
    hgate=energy['complete'] and energy['composition_ci95'][1]<0 and max(energy['by_fit'])<0 and he['joint']['mean']>=0 and he['graph']['mean']>=0
    target=a.out.with_suffix('.npz');np.savez_compressed(target,**arrays)
    costs={k:sum(d['costs'][k] for d in audits) for k in audits[0]['costs']}
    costs.update(new_parent_optimizer_steps=135000,new_parent_training_forward_examples=5760000,new_h_optimizer_steps=30000,new_h_training_forward_examples=960000)
    write(a.out,dict(complete=True,campaign_sha256=ph,summary=summary,contrasts=contrasts,primary_new_fit_replication_gate=bool(gate),
        primary_new_fit_crossed_ci_positive=bool(primary['crossed_fit_composition_ci95'][0]>0),hydrogen_new_fit_physical_gate=bool(hgate),
        methods=methods,provenance=provenance,arrays_sha256=sha(target),costs=costs,size_counts={'17-28':20,'29-40':44},
        scope='Frozen five-fit comparison on64 new compositions. New-three-fit analysis is the independent-training replication; old fits informed prior method development. Crossed intervals separately reflect limited five/three-fit variability. Every output, failure and fitted model is retained. No comprehensive superiority claim is automatic.'))
    print(json.dumps(dict(primary=primary,hydrogen=he,primary_new_fit_replication_gate=bool(gate),hydrogen_new_fit_physical_gate=bool(hgate))),flush=True)

if __name__=='__main__':main()
