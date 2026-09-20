"""Paired composition and fitted-model uncertainty; NumPy is the only dependency."""
import numpy as np


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

