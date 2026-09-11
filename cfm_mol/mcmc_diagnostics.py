"""Conservative multi-chain diagnostics; passing numbers are not a mixing proof."""
import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata


def _rhat(x):
    m,n=x.shape
    within=x.var(axis=1,ddof=1).mean();between=x.mean(1).var(ddof=1)
    if within<=0:return None
    return float(np.sqrt(((n-1)/n*within+between)/within))


def diagnose_chains(values):
    x=np.asarray(values,dtype=float)
    if x.ndim!=2 or min(x.shape)<4 or not np.isfinite(x).all():
        raise ValueError('Require finite[chains,draws] with at least four of each')
    m,n=x.shape;half=n//2
    split=np.concatenate([x[:,:half],x[:,-half:]],axis=0)
    def normalized(a):
        ranks=rankdata(a.reshape(-1),method='average').reshape(a.shape)
        return ndtri((ranks-.375)/(a.size+.25))
    rank_rhat=_rhat(normalized(split));folded_rhat=_rhat(normalized(abs(split-np.median(split))))
    rhats=[r for r in [rank_rhat,folded_rhat] if r is not None]
    centered=x-x.mean(1,keepdims=True)
    fft=np.fft.rfft(centered,n=2*n,axis=1)
    acov=np.fft.irfft(fft*np.conjugate(fft),n=2*n,axis=1)[:,:n]/n
    within=x.var(axis=1,ddof=1).mean();var_plus=(n-1)/n*within+x.mean(1).var(ddof=1)
    ess=None
    if var_plus>0:
        rho=1-(within-acov.mean(0))/var_plus;rho[0]=1.
        total=0.;previous=np.inf
        for i in range(0,n-1,2):
            pair=float(rho[i]+rho[i+1])
            if pair<=0:break
            pair=min(pair,previous);total+=pair;previous=pair
        tau=max(1.,-1+2*total)
        ess=float(m*n/tau)
    return dict(chains=m,draws_per_chain=n,mean=float(x.mean()),
        split_rank_rhat=max(rhats) if rhats else None,ess=ess,
        scope='Finite-chain diagnostic; requires separate initialization and mode-coverage checks')
